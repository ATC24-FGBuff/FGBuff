import torch

from adtopk_lib import Compressor
import random
import horovod.torch as hvd
import numpy as np
import scipy.stats as stats


class GaussiankCompressor(Compressor):

    def __init__(self, compress_ratio, rank, epoch=0):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0


    def gen_threshold_from_normal_distribution(self, p_value, mu, sigma):
        zvalue = stats.norm.ppf((1-p_value)/2)
        return mu+zvalue*sigma, mu-zvalue*sigma


    # tensor反稀疏化
    def desparsify(self, tensors, numel, shape):
        values, indices = tensors
        # if True:
        if values.numel()==numel:
            return values
        else:
            tensor_decompressed = torch.zeros(
                numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(0, indices, values)

            return tensor_decompressed


    # 抽象方法重载compress
    def compress(self, tensor, name):
        numel = tensor.numel()
        shape = tensor.size()
        
        
        k = max(int(numel * self.compress_ratio), 1)

        std = torch.std(tensor)
        mean = torch.mean(tensor)
        left_thres, thr = self.gen_threshold_from_normal_distribution(1- self.compress_ratio, float(mean), float(std))

        tensor_flatten = tensor.flatten().cuda()
        # abs_tensor_tensor_flatten = torch.abs(tensor_flatten)
        
        # one_indexes = abs_tensor_tensor_flatten > right_thres
        # loops = 0
        # while loops <  2:
        # while loops <  5:
        #     one_indexes = abs_tensor_tensor_flatten > right_thres
        #     selected = one_indexes.sum()
            
        #     # print(f"loops: {loops}, selected: {selected}")
        #     # print(f"loops: {loops}, one_indexes: {one_indexes}")
        #     # indexes = one_indexes.nonzero().data.squeeze().view(-1)
        #     # print(f"indexes: {indexes}")

        #     if selected < 2*k/3:
        #         right_thres *= 0.5
        #     elif selected > 4*k/3:
        #         right_thres *= 1.5
        #     else:
        #         break
        #     loops += 1
        # # indexes = indexes 
        # indices, = torch.where(one_indexes)
        # indices = indices.cuda(tensor.device)
        mask = tensor_flatten.abs() >= thr
        selected = mask.sum()

        # for _ in range(2):
        #     if selected > 1.3 * k:
        #         thr = 1.3 * thr
        #     elif selected < 0.7 * numel * k:
        #         thr = 0.7 * thr
        #     else:
        #         break
        #     mask = tensor_flatten.abs() >= thr
        #     selected = mask.sum()
        for _ in range(5):
            if selected > 1.2 * k:
                thr = 1.2 * thr
            elif selected < 0.8 * numel * k:
                thr = 0.8 * thr
            else:
                break
            mask = tensor_flatten.abs() >= thr
            selected = mask.sum()
            
        # indexes = indexes[0:k]
        indices, = torch.where(mask)
        
        values = tensor_flatten[indices]

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

        tensors = values, indices
        ctx = numel, shape
        return tensors, ctx
    

    def decompress(self, tensors, ctx, name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        tensor_decompressed =self.desparsify(tensors, numel, shape) 

        return tensor_decompressed.view(shape)
    

    def decompress_add(self, tensors, ctx, name):
        
        
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        values, indices = tensors
        # if values.numel()==numel:
        #     return values
        # 返回一个形状为为size,类型为torch.dtype,里面的每一个值都是0的tensor
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
