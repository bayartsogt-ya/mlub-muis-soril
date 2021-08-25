python train.py \
    --model-name-or-path bayartsogt/mongolian-roberta-large \
    --num-epochs 15 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --roberta-large-optimizer \
    --truncate-length 30 \
    --max-len 150 \
    --push-to-hub \
    --submit-to-kaggle