
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/
# scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/*  user@node22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/

scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/*  user@node22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/example_dear/


scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/
scp -r  /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/*   user@node22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/Bayesian/


# compressors = {
#         None: NoneCompressor,
#         'none': NoneCompressor,
#         'topk': TopKCompressor,
#         'topkef': EFTopKCompressor,
#         'eftopk': EFTopKCompressor, #TopK with error-feedback
#         'gaussian': GaussianCompressor, #GaussianK with error-feedback
#         'dgc': DgcCompressor,
#         'redsync' :RedSyncCompressor,
#         'randomk': RandomKCompressor,
#         'sidco': ExpCompressor,
#         # 'signum': SignCompressor,
#         # 'efsignum': EFSignCompressor,
#     }
density="${density:-0.01}"
# threshold="${threshold:-8192}"
threshold="${threshold:-67108864}"
compressor="${compressor:-topkef}"
# compressor="${compressor:-gaussian}"
max_epochs="${max_epochs:-200}"
memory="${memory:-residual}"

nwpernode=4
nstepsupdate=1
PY=python


HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  wiki2_lstm_topk.py  --epochs $max_epochs --density $density --compressor $compressor --threshold $threshold


