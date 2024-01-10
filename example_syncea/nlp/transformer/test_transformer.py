# coding: utf-8
import argparse
import time
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import horovod.torch as hvd

import datahelper
import model
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--dataset', type=str, default='/data/dataset/nlp/transformer/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

# horovod 0
hvd.init()


# horovod 1
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    # print(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)


# if args.cuda:
device = torch.device("cuda")

###############################################################################
# Load data
###############################################################################

print(args.dataset)
corpus = datahelper.Corpus(args.dataset)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
    # return data.to(device)

size = hvd.size()

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)

minibatch_size = train_data.size()[0] // size
train_datas = torch.chunk(train_data, size , dim=0)
train_data = train_datas[hvd.local_rank()]

val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

lr =  1
# print(hvd.size())

best_val_loss = None
optimizer = optim.SGD(model.parameters(), lr=lr)

from grace_dll.torch.helper import grace_from_params
params = {'compressor': 'topkkexi', 'memory': 'residual', 'communicator': 'allgather'}
# params = {'compressor': 'none', 'memory': 'none', 'communicator': 'allreduce'}
# params = {'compressor': 'qsgd', 'memory': 'none', 'communicator': 'allreduce'}
# params = {'compressor': 'qsgd', 'memory': 'none', 'communicator': 'allreduce'}
grc = grace_from_params(params)


# optimizer = hvd.DistributedOptimizer(optimizer, grc, named_parameters=model.named_parameters())
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x): # x是step次数
        if x >= warmup_iters:
            return 0.95**(x-warmup_iters)
        alpha = float(x) / warmup_iters # 当前进度 0-1
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


lr_step = lr_scheduler.ExponentialLR(optimizer, gamma=0.95) 
warmup_factor = 1. / 1000
warmup_iters = 10
lr_step = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


def train(optimizer):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    optimizer.iteration = len(range(0, train_data.size(0) - 1, args.bptt))

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        optimizer.iteration -= 1
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        # for name, parms in model.named_parameters():
        #     print('\nBefore backward\n')
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     # print('-->grad_requirs:',parms.requires_grad)
        #     # print('-->grad_value:',parms.grad)
        #     print('-->grad_value:',parms.grad.shape)
        #     print("===========================")

        loss.backward()
        
        for name, parms in model.named_parameters():	
            print('\nAfter backward\n')
            print('-->name:', name)
            # print('-->para:', parms)
            print('-->grad_requirs:',parms.requires_grad)
            # print('-->grad_value:',parms.grad)
            print('-->grad_value:',parms.grad.shape)
            print("===========================")




        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)
        optimizer.step()

        total_loss += loss.item()

        # if batch % args.log_interval == 0 and batch > 0 and hvd.local_rank()==0:
        #     cur_loss = total_loss / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
        #             'loss {:5.2f} | ppl {:8.2f}'.format(
        #         epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
        #         elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()
        if args.dry_run:
            break
        
        break
    lr_step.step()



if hvd.rank() == 0:
    for name,parameters in model.named_parameters():

        print(name,':',parameters.size())
# for name, parms in model.named_parameters():
#     print('\nBefore backward\n')
#     print('-->name:', name)
#     print('-->para:', parms)
#     print('-->grad_requirs:',parms.requires_grad)
#     print('-->grad_value:',parms.grad)
#     print("===========================")
#
# train_loss.backward()
#
# for name, parms in model.named_parameters():	
#     print('\nAfter backward\n')
#     print('-->name:', name)
#     print('-->para:', parms)
#     print('-->grad_requirs:',parms.requires_grad)
#     print('-->grad_value:',parms.grad)
#     print("===========================")

# train(optimizer)