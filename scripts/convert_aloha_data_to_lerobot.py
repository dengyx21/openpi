"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    decode_workers: int = 8
    decode_chunk_size: int = 64


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    cameras: list[str] | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    if cameras is None:
        cameras = [
            "cam_high",
            "cam_low",
            "cam_left_wrist",
            "cam_right_wrist",
        ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _jpeg = TurboJPEG()
    def _decode_image(data: bytes) -> np.ndarray:
        """Decode a single compressed image using PyTurboJPEG."""
        return _jpeg.decode(data, pixel_format=TJPF_RGB)
except (ImportError, RuntimeError):
    import cv2
    import numpy as np
    def _decode_image(data: bytes) -> np.ndarray:
        """Decode a single compressed image from bytes to a numpy array."""
        img_array = np.frombuffer(data, np.uint8)
        return cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def iter_episode_frames(
    ep_path: Path,
    cameras: list[str],
    decode_workers: int = 8,
    chunk_size: int = 64,
):
    """Generator that yields frames from an episode one at a time.

    Compressed images are decoded in parallel chunks so that neither
    single-threaded decoding nor loading all frames into RAM becomes
    the bottleneck.
    """
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        num_frames = state.shape[0]

        velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
        effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None

        # Separate cameras by storage type
        uncompressed: dict[str, h5py.Dataset] = {}
        compressed: dict[str, h5py.Dataset] = {}
        for camera in cameras:
            dataset = ep[f"/observations/images/{camera}"]
            if dataset.ndim == 4:
                uncompressed[camera] = dataset
            else:
                compressed[camera] = dataset

        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            for chunk_start in range(0, num_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_frames)
                chunk_len = chunk_end - chunk_start

                # Decode compressed images in this chunk, one camera at a time
                # (each camera saturates the thread pool, so inter-camera
                # parallelism wouldn't help)
                chunk_imgs: dict[str, np.ndarray] = {}
                for camera, dataset in compressed.items():
                    raw_chunk = [bytes(d) for d in dataset[chunk_start:chunk_end]]
                    chunk_imgs[camera] = np.array(list(pool.map(_decode_image, raw_chunk)))

                # Uncompressed images: slice directly from HDF5
                for camera, dataset in uncompressed.items():
                    chunk_imgs[camera] = dataset[chunk_start:chunk_end]

                # Yield frames in this chunk
                for i in range(chunk_len):
                    frame: dict = {}
                    frame["observation.state"] = state[chunk_start + i]
                    frame["action"] = action[chunk_start + i]
                    for camera in cameras:
                        frame[f"observation.images.{camera}"] = chunk_imgs[camera][i]
                    if velocity is not None:
                        frame["observation.velocity"] = velocity[chunk_start + i]
                    if effort is not None:
                        frame["observation.effort"] = effort[chunk_start + i]
                    yield frame


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    cameras: list[str],
    episodes: list[int] | None = None,
    decode_workers: int = 8,
    decode_chunk_size: int = 64,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        for frame in iter_episode_frames(
            ep_path, cameras,
            decode_workers=decode_workers,
            chunk_size=decode_chunk_size,
        ):
            frame["task"] = task
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    resume: bool = False,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if not resume and (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        raise ValueError(
            f"raw_dir {raw_dir} does not exist. Provide a local directory of episode_*.hdf5 files."
        )

    # Sort by the episode number in the filename for correct resumption
    import re
    def extract_ep_num(f: Path) -> int:
        match = re.search(r"episode_(\d+)", f.stem)
        return int(match.group(1)) if match else 0
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"), key=extract_ep_num)

    if not hdf5_files:
        raise ValueError(f"No episode_*.hdf5 files found under {raw_dir}.")

    cameras = get_cameras(hdf5_files)

    if resume and (HF_LEROBOT_HOME / repo_id).exists():
        print(f"Resuming conversion. Loading existing dataset at {HF_LEROBOT_HOME / repo_id}")
        dataset = LeRobotDataset(
            repo_id,
            root=HF_LEROBOT_HOME / repo_id,
            video_backend=dataset_config.video_backend,
        )
        if dataset_config.image_writer_processes or dataset_config.image_writer_threads:
            dataset.start_image_writer(dataset_config.image_writer_processes, dataset_config.image_writer_threads)
        print(f"Loaded dataset with {dataset.num_episodes} already processed episodes.")
    else:
        dataset = create_empty_dataset(
            repo_id,
            robot_type="mobile_aloha" if is_mobile else "aloha",
            mode=mode,
            has_effort=has_effort(hdf5_files),
            has_velocity=has_velocity(hdf5_files),
            cameras=cameras,
            dataset_config=dataset_config,
        )

    # Automatically skip already processed episodes when resuming if not specificly overridden
    if episodes is None:
        start_idx = dataset.num_episodes if resume and (HF_LEROBOT_HOME / repo_id).exists() else 0
        episodes_to_append = list(range(start_idx, len(hdf5_files)))
    else:
        episodes_to_append = episodes

    if not episodes_to_append:
        print("No new episodes to process.")
        return

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        cameras=cameras,
        episodes=episodes_to_append,
        decode_workers=dataset_config.decode_workers,
        decode_chunk_size=dataset_config.decode_chunk_size,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
