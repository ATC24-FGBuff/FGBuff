import torch

from adtopk_lib import Compressor
import random
import numpy as np
import horovod.torch as hvd
import math
import time
import scipy.stats as stats


class ImbalanceTopkTimeCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]

        self.attributes = {}
        self.thresholds = {}
        self.tensor={}
        self.indices={}
        
        self.topk_time=[]
        self.threshold_time=[]
        self.detopk_time=[]

        self.bias_gaussiank=[]
        self.bias_dgc=[]
        self.bias_redsync=[]


    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            
            self.thresholds[name]=0.0
            self.indices[name]=None
            
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
            self.attributes[name] ={'numel':numel,'shape': shape, 'compress_ratio':self.compress_ratio,'rank':self.rank,'thres_global':thres_global,'afa':afa,\
                'compression_global':compression_global,'indices_global':indices_global,'values_global':values_global,\
                    'indices_channel_1':indices_channel_1,'values_channel_1':values_channel_1,\
                        'tensor_original':tensor_original,'tensor_mean_global':tensor_mean_global,'tensor_mean_channel':tensor_mean_channel,\
                            'tensors_aggregated':tensors_aggregated,'scale':scale,'tensors_aggregated_mean':tensors_aggregated_mean,\
                                'tensors_residuals':tensors_residuals,'sign':sign} 


    def sparsify(self,tensor, compress_ratio, epoch, name):

        tensor_flatten = tensor.flatten()
       
        
        numel = tensor.numel()
     
        time_start_=time.time()
        if tensor.dim()==2:
            values=tensor_flatten
            indices=torch.arange(0,numel).cuda(tensor.device)
            return values, indices
        
   
        
        thres=self.gaussiank_threshold_estimation(tensor_flatten, self.compress_ratio)
        # thres=self.dgc_threshold_estimation(tensor, compress_ratio)
        
        mask = tensor_flatten.abs() >=thres
        indices_flatten_global, = torch.where(mask)
     


        values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
        e_topk_time=time.time()-time_start_
   
        self.topk_time.append(e_topk_time)
        
        return values_flatten_global, indices_flatten_global
        
    def desparsify(self,tensors, numel,shape,name):
        values, indices = tensors
        if values.numel()==numel:
            return values

        else:
            tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(0, indices, values)
            return tensor_decompressed
            

    
    def compress(self, tensor, name):
        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch, name)
        self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']

        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx,name):

        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed = self.desparsify(tensors, numel,shape,name)
        return tensor_decompressed.view(shape)
    
    def decompress_add(self, tensors, ctx, name):
        if ctx==None:
            tensor, = tensors
            return tensor

        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values

        else:
            tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)


            return tensor_decompressed.view(shape)


            if values.dim()==1:
                tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
            else:

                # case-1
                tensor_decompressed = torch.zeros(
                    shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

                # if hvd.rank() == 0:
                #     print(tensor_decompressed.shape, indices.shape)
                size = hvd.size()
                sizes = [tensor_decompressed.shape[0]] * size
                indices_list = indices.split(sizes)
                indices = torch.concatenate(indices_list,axis = 1)
                values_list = values.split(sizes)
                values = torch.concatenate(values_list, axis = 1)
                tensor_decompressed = tensor_decompressed.scatter_add(1, indices, values)

        return tensor_decompressed.view(shape)




    
    # 高斯阈值估计Gaussiank
    def gaussiank_threshold_estimation(self, tensor, compress_ratio):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * compress_ratio), 1)

        std = torch.std(tensor)
        mean = torch.mean(tensor)
        left_thres, right_thres = self.gen_threshold_from_normal_distribution(1-compress_ratio, float(mean), float(std))

        # tensor_flatten = tensor.flatten().cuda()
        # abs_tensor_tensor_flatten = torch.abs(tensor_flatten)
        # one_indexes = abs_tensor_tensor_flatten > right_thres
        # loops = 0
        # while loops < 5:
        #     one_indexes = abs_tensor_tensor_flatten > right_thres
        #     selected = one_indexes.sum()
    
        #     if selected < 2*k/3:
        #         right_thres *= 0.5
        #     elif selected > 4*k/3:
        #         right_thres *= 1.5
        #     else:
        #         break
        #     loops += 1

        # abs_tensor = torch.abs(tensor)
        # loops = 0
        # while loops < 5:
        #     one_indexes = abs_tensor > right_thres
        #     indexes = one_indexes.nonzero().data.squeeze().view(-1)
        #     if indexes.numel() < 2*k/3:
        #         right_thres *= 0.5
        #     elif indexes.numel() > 4*k/3:
        #         right_thres *= 1.5
        #     else:
        #         break
        #     loops += 1
        # indices = indexes 
        # values = tensor.data[indexes]

        return right_thres

    def gen_threshold_from_normal_distribution(self, p_value, mu, sigma):
        zvalue = stats.norm.ppf((1-p_value)/2)
        return mu+zvalue*sigma, mu-zvalue*sigma


    # DGC阈值估计
    def dgc_threshold_estimation(self, tensor, compress_ratio):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        k = max(1, int(numel * compress_ratio * 0.01))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        thr = vals.min()
        
        # thr = vals.max()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(2):
            if selected > 1.3 * numel * compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        return thr

    # Redsync阈值估计
    def redsync_threshold_estimation(self, tensor, compress_ratio):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * compress_ratio), 1)

        tensor_flatten = tensor.flatten().cuda()

        l = 0.0
        r = 1.0
        thres = 0.0
        eps = 0.2
        abs_tensor = torch.abs(tensor_flatten)
        mean_val = torch.mean(abs_tensor)
        max_val = torch.max(abs_tensor)

        one_indexes = abs_tensor > thres
        while r - l > eps:
            tmp_ratio = l + (r-l)/2
            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = abs_tensor > thres
            # indexes = one_indexes.nonzero().data.squeeze().view(-1)
            # nnz = indexes.numel()
            nnz = one_indexes.sum()

            if nnz > k and 2*k > nnz:
                break
            elif nnz < k/2:
                r = tmp_ratio
            else:
                l = tmp_ratio
        
        return thres


