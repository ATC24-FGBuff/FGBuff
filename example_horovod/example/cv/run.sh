#!/bin/bash

func() {
    echo "Usage:"
    echo "run.sh [-d dataset] [-m model] [-c compressor] [-e epochs]"
    echo "dataset:      cifar100, imagenet"
    echo "model:        resnet50, resnet101, vgg16, vgg19"
    echo "compressor:   baseline, actopk, allchanneltopk, globaltopk, dgc, gaussiank, redsync, sidco"
    exit -1
}


while getopts 'h:d:m:c:e:' OPT; do
    case $OPT in
        d) dataset="$OPTARG";;
        m) model="$OPTARG";;
        c) compressor="$OPTARG";;
        e) epochs="$OPTARG";;
        h) func;;
        ?) func;;
    esac
done

# echo "$dataset $model $compressor $epochs"

HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H n15:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python zxj/pytorch_${dataset}_${model}_${compressor}.py --epochs ${epochs}