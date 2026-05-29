uv run --no-sync scripts/convert_aloha_data_to_lerobot.py \
    --raw-dir /home/dyx/ocean/huggingface_cache/hub/datasets--dengyixuan--clothes/snapshots/9a941f44717df35743328adabf2eb12f2328b07e \
    --repo-id dyx/clothes \
    --task "fold the clothes"  \
    --resume 
    # --dataset-config.decode-workers 48 \
    # --dataset-config.decode-chunk-size 256 \
    # --dataset-config.image-writer-processes 24 \
    # --dataset-config.image-writer-threads 4