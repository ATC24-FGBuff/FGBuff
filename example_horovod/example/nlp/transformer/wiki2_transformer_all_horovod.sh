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
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  wiki2_transformer_topk.py  --epochs $max_epochs --density 0.01 --compressor gaussiank --threshold $threshold


