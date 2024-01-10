scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n16:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n17:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n18:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n19:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n20:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n21:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/
scp -r /home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/* user@n22:/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/omgs/example/nlp/transformer/

HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H n15:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python transformer.py --epochs 120