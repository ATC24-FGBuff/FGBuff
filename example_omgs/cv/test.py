import time
import torch
import numpy as np
import sys
sys.path.append("../..") 
import utils_model

from profiling import CommunicationProfiler
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
# import horovod.torch as hvd
import os
import math
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 环境变量HOROVOD_FUSION_THRESHOLD实际上以字节为单位.
# 然而, 当使用horovodrun时, 有一个--fusion-threshold-mb以MB为单位的参数.
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'


import sys
sys.path.append("../..") 
import hv_distributed_optimizer_omgs as hvd
from compression import compressors
from utils_model import get_network

import timeit
import numpy as np
from profiling_omgs import benchmark



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar100 + ResNet-50 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Horovod
# parser.add_argument('--net', default='resnet50',type=str, required=True, help='net type')
# parser.add_argument('--model-net', default='resnet50',type=str, help='net type')
parser.add_argument('--model-net', default='resnet50',type=str, help='net type')

# parser.add_argument('--model-net', default='vgg16',type=str, help='net type')



parser.add_argument('--train-dir', default=os.path.expanduser('~/cifar100/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/cifar100/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./pytorch_checkpoints/cifar100_resnet50/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')


# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')

parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')


# Gradient Merging
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
# parser.add_argument('--batch-size', type=int, default=32,
#                     help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=20,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=50,
                    help='number of benchmark iterations')

# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')

# parser.add_argument('--use-adasum', action='store_true', default=False,
#                     help='use adasum algorithm to do reduction')

parser.add_argument('--mgwfbp', action='store_true', default=True, help='Use MG-WFBP')
parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')

# 设置合并的阈值大小,default=23705252为ResNet50所有层梯度元素数量的总和
# parser.add_argument('--threshold', type=int, default=536870912, help='Set threshold if mgwfbp is False')
# parser.add_argument('--threshold', type=int, default=671080, help='Set threshold if mgwfbp is False')
parser.add_argument('--threshold', type=int, default=2370520, help='Set threshold if mgwfbp is False')

# parser.add_argument('--threshold', type=int, default=1000520, help='Set threshold if mgwfbp is False')


# parser.add_argument('--threshold', type=int, default=23705252, help='ResNet-50 Set threshold if mgwfbp is False')

parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')

# Baseline
# parser.add_argument('--compressor', type=str, default='none', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
# parser.add_argument('--density', type=float, default=1, help='Density for sparsification')

# Top-k + EF
parser.add_argument('--compressor', type=str, default='eftopk', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=0.1, help='Density for sparsification')
# parser.add_argument('--density', type=float, default=0.0101, help='Density for sparsification')
# parser.add_argument('--density', type=float, default=0.0099, help='Density for sparsification')


# Gaussiank + EF
# parser.add_argument('--compressor', type=str, default='gaussian', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
# parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')





# sub_buffer = utils.optimal_gradient_merging_0101(11111, 'resnet50', density=0.1)




def test_benchmark():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.set_default_dtype(torch.float16)
    os.environ['HOROVOD_NUM_NCCL_STREAMS'] = str(args.nstreams)
    
    # allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        
    cudnn.benchmark = True
    model=get_network(args)
    
    
    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()
            
    image_size = 224
    if args.model == 'inception_v3':
        image_size = 227
    data = torch.randn(args.batch_size, 3, image_size, image_size)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    
    if args.mgwfbp:

        seq_layernames, layerwise_times, _ = benchmark(model, (data, target), F.cross_entropy, task='cifar100')
        # layerwise_times =hvd.broadcast(layerwise_times,root_rank=0)
        # layerwise_times = comm.bcast(layerwise_times, root=0)
    else:
        seq_layernames, layerwise_times = None, None
    
    print('seq_layernames= ', seq_layernames)
    print('seq_layernames= ', seq_layernames)
    
   
   
test_benchmark()
    
    
