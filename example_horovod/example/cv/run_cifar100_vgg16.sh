# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/*  user@node22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/cv/



scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/




# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 bash run.sh
# dnn="${dnn:-resnet20}"
# source exp_configs/$dnn.conf
# nworkers="${nworkers:-4}"
# density="${density:-0.001079929467}"

# compressors = {
#         None: 'none',
#         'none': '',
#         'topk': topk,
#         'topkef': topkef, #TopK with error-feedback
#         'gaussian': gaussiank, #GaussianK with error-feedback
#         'dgc': dgc,
#         'redsync' :redsync,
#         'sidco': sidcoexp,
#         'randomk': randomk,
#     }

density="${density:-0.1}"
threshold="${threshold:-8192}"
compressor="${compressor:-topkef}"
max_epochs="${max_epochs:-80}"
memory="${memory:-residual}"

nwpernode=4
nstepsupdate=1
PY=python


HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  pytorch_cifar100_vgg16.py  --epochs $max_epochs --density $density --compressor $compressor --threshold $threshold

