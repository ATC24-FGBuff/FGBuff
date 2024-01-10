import torch

from adtopk_lib import Compressor
import random
import numpy as np
import horovod.torch as hvd
import math




class TopKCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]

        # self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        # self.strided_sample = strided_sample
        # self.compress_upper_bound = compress_upper_bound
        # self.compress_lower_bound = compress_lower_bound
        # self.max_adaptation_iters = max_adaptation_iters
        # self.resample = resample

        self.attributes = {}
        self.tensor={}

        # self.residuals={{}}
        # for i in range(hvd.size()):
        #     self.residuals[str(i)]={}
   
    def initialize(self, named_parameters):
        # if hvd.rank() == 0:
        #     print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            
            afa=0.2
            thres_global=None
            compression_global=None
            indices_global= None
            values_global=None
            indices_channel_1=None
            values_channel_1=None
            tensor_original=None
            tensor_mean_global=None
            tensor_mean_channel=None
            tensors_aggregated=None
            scale=None
            tensors_aggregated_mean=None
            tensors_residuals=None
            self.compress_ratio=0.01
            sign=-1
            # self.attributes[name] ={'numel':numel,'shape': shape, 'compress_ratio':self.compress_ratio,'rank':self.rank,'thres_global':thres_global,'afa':afa,\
            #     'compression_global':compression_global,'indices_global':indices_global,'values_global':values_global,\
            #         'indices_channel_1':indices_channel_1,'values_channel_1':values_channel_1,\
            #             'tensor_original':tensor_original,'tensor_mean_global':tensor_mean_global,'tensor_mean_channel':tensor_mean_channel,\
            #                 'tensors_aggregated':tensors_aggregated,'scale':scale,'tensors_aggregated_mean':tensors_aggregated_mean,\
            #                     'tensors_residuals':tensors_residuals,'sign':sign} 
            
            
            # self.residuals[str(hvd.rank())][name]=None
            # for i in range(hvd.size()): 
            #     self.residuals[str(i)]={}

    # tensor稀疏化得到top-k的稀疏值
    def sparsify(self,tensor, compress_ratio,epoch, name):
        tensor_flatten = tensor.flatten()
        numel = tensor.numel()

        if self.compress_ratio<1:
            k= max(1, int(numel * compress_ratio))
            _, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
        
            return values_flatten_global, indices_flatten_global
        
        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values = tensor
        indices = torch.arange(0,numel).cuda(tensor.device)
        return values, indices


    # tensor反稀疏化
    def desparsify(self,tensors, numel,shape,name):
        values, indices = tensors
        if values.numel()==numel:
            return values

        else:
            tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(0, indices, values)


        return tensor_decompressed

    # 抽象方法重载compress
    def compress(self, tensor, name):


        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch, name)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def show_sparsify(self, tensor):
        # if self.rank==0:
        print('----------'+str(self.rank)+'----------')
        print(tensor.shape)
        tensor = tensor.flatten()
        print(tensor)

    def decompress(self, tensors, ctx,name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed = self.desparsify(tensors, numel,shape,name)
        # if self.rank==0:
        return tensor_decompressed.view(shape)

    def decompress_add(self, tensors, ctx, name):
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        # 返回一个形状为为size,类型为torch.dtype,里面的每一个值都是0的tensor
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        
        # 填充稀疏值
        # if hvd.rank() == 0:
        #     print('values: ', values, 'indices: ', indices)
        # [a,b,    c,d]  [0,1,    0,2]
        # [c, b ,d ][a+c, b,d ]
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
