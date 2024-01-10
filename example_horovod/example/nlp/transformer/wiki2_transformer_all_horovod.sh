# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/
# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/
# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/
# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/
# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/
# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/
# scp /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/*   user@n22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/example/nlp/transformer/


scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/*  user@node22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/horovod/

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
density="${density:-0.01}"
threshold="${threshold:-8192}"
# compressor="${compressor:-topkef}"
compressor="${compressor:-gaussian}"
max_epochs="${max_epochs:-120}"
memory="${memory:-residual}"

nwpernode=4
nstepsupdate=1
PY=python


# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  wiki2_transformer_topk.py  --epochs $max_epochs --density $density --compressor $compressor --threshold $threshold

# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  wiki2_transformer_topk.py  --epochs $max_epochs --density 0.1 --compressor gaussian --threshold $threshold
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  wiki2_transformer_topk.py  --epochs $max_epochs --density 0.05 --compressor gaussian --threshold $threshold
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  wiki2_transformer_topk.py  --epochs $max_epochs --density 0.01 --compressor gaussiank --threshold $threshold


