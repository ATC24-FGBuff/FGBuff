#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H n15:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python train_gtopk.py \
        --cuda \
        --data /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/data/wikitext-103 \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 5000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 60 \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
