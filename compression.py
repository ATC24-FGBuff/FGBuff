# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time
import math
import utils_optimizer
from scipy import stats
import hv_distributed_optimizer as hvd

class NoneCompressor():
    def __init__(self):
        self.name = 'none'

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=1.0):
        # tensor = tensor.flatten().cuda()
        # numel = tensor.numel()
        # values=tensor
        # indexes=torch.arange(0,numel).cuda(tensor.device)
        # return tensor, indexes, values
    
        return tensor, tensor.dtype

    def decompress(self, tensor, ctc):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {}
        
        self.c = 0
        self.t = 0.
        self.name = 'topk'
        self.zc = None
        self.current_ratio = 1
        
        # 实时更新
        self.epoch=0
        self.iteration=0
        
        self.attributes = {}
        self.thresholds = {}
        self.tensor={}
        self.indices={}
        
        self.topk_time=[]
        self.threshold_time=[]
        self.detopk_time=[]
        
        

    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data):
        if name not in self.zero_conditions:
            self.zero_conditions[name] = torch.ones(data.numel(), dtype=torch.float32, device=data.device) 
        zero_condition = self.zero_conditions[name]
        zero_condition.fill_(1.0)
        zero_condition[self.indexes[name]] = 0.0
        self.zc = zero_condition

    def clear(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 
    
    # Traditional Compression
    def compress(self, tensor, name=None, group_size=None, sigma_scale=2.5, ratio=0.01):
        time_start_=time.time()
        with torch.no_grad():
            
            # if name not in self.residuals:
            #     self.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()

            self.current_ratio = ratio
            
            if ratio==1:
            # if ratio ==1 or 'fc' in name:
                numel = tensor.numel()
                values =tensor
                indexes=torch.arange(0,numel).cuda(tensor.device)
                return tensor, indexes, values
            
            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)

            k = max(int(numel * ratio), 1)
            values_, indexes = torch.topk(torch.abs(tensor.data), k=k)

            values = tensor.data[indexes]
            
            
            # self.residuals[name].data = tensor.data + 0.0 
            # self.residuals[name].data[indexes] = 0.
            
            e_topk_time=time.time() - time_start_
            self.topk_time.append(e_topk_time)

            return tensor, indexes, values
    
    def compress_layer_wise(self, tensor, name=None, group_size=None, sigma_scale=2.5, ratio=0.01):
        time_start_=time.time()
        with torch.no_grad():
            
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            # numel = tensor.numel()
            
            self.current_ratio = ratio
            if ratio==1:
                numel = tensor.numel()
                values=tensor
                indexes=torch.arange(0,numel).cuda(tensor.device)
                return tensor, indexes, values
            
            
            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)
            
            tensor_abs=torch.abs(tensor.data)

            pre_size=0
            values = None
            indexes = None  
            indexes_arr=[]
            
            for i, s in enumerate(group_size):
                k = max(int(s * ratio), 1)
                values_, indexes_ = torch.topk(tensor_abs[pre_size:pre_size+s], k=k)
                # if group_dim[i]!=2:
                #     k = max(int(s * ratio), 1)
                #     values_, indexes_ = torch.topk(tensor_abs[pre_size:pre_size+s], k=k)
                # else:
                #     indexes_=torch.arange(0,s).cuda(tensor.device)
                
                indexes_+=pre_size
                pre_size+=s
                indexes_arr.append(indexes_)
            
            indexes = torch.cat(indexes_arr, dim=0)
            values = tensor.data[indexes]
                      
            
            # for i, s in enumerate(group_size):
            #     # layer_name=names[i]
            #     numel=s
            #     # if numel<10000:
            #     #     indexes_=torch.tensor(range(s)).cuda()
            #     # else:
            #     #     k = max(int(numel * ratio), 1)
            #     #     values_, indexes_ = torch.topk(torch.abs(tensor.data[pre_size:pre_size+s]), k=k)
            #     # indexes_+=pre_size
            #     # pre_size+=s  
            #     k = max(int(numel * ratio), 1)
            #     values_, indexes_ = torch.topk(tensor_abs[pre_size:pre_size+numel], k=k)
            #     indexes_+=pre_size
            #     pre_size+=numel
                
            #     if indexes==None:
            #         indexes=indexes_
            #     else:
            #         indexes=torch.concatenate((indexes,indexes_), dim=0)
            # values = tensor.data[indexes]

            # self.residuals[name].data = tensor.data + 0.0 
            # self.residuals[name].data[indexes] = 0.            
            
            e_topk_time=time.time()-time_start_
            self.topk_time.append(e_topk_time)
            
            return tensor, indexes, values

    
    def compress_layer_wise_selective(self, tensor, name=None, group_size=None, group_dim=None,sigma_scale=2.5, ratio=0.01):
        time_start_=time.time()
        with torch.no_grad():
            
            if ratio==1:
                numel = tensor.numel()
                values=tensor
                indexes=torch.arange(0,numel).cuda(tensor.device)
                return tensor, indexes, values
            
            
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
                      
            self.current_ratio = ratio
            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)
            
            tensor_abs=torch.abs(tensor.data)

            pre_size=0
            values = None
            indexes = None
            indexes_arr=[]
            
            for i, s in enumerate(group_size):
                
                if group_dim[i]!=2:
                    k = max(int(s * ratio), 1)
                    values_, indexes_ = torch.topk(tensor_abs[pre_size:pre_size+s], k=k)
                else:
                    indexes_=torch.arange(0,s).cuda(tensor.device)
                
                indexes_+=pre_size
                pre_size+=s
                indexes_arr.append(indexes_)
            
            indexes = torch.cat(indexes_arr, dim=0)
            values = tensor.data[indexes]

            # self.residuals[name].data = tensor.data + 0.0 
            # self.residuals[name].data[indexes] = 0.


            e_topk_time=time.time()-time_start_
            self.topk_time.append(e_topk_time)

            return tensor, indexes, values
    
    
    # Block-based compression
    def compress_block(self, tensor, name=None, group_size=None, group_dim=None, sigma_scale=2.5, ratio=0.01):
        time_start_ = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)

            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)
            
            tensor_abs=torch.abs(tensor.data)
            values = None
            indexes = None
            
            numel = tensor.numel()            
            
            self.current_ratio = ratio
            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)
            pre_size=0
            indexes_arr=[]
            k = max(int(numel * ratio), 1)
            values_, indexes_ = torch.topk(tensor_abs, k=k)
            
            indexes_arr.append(indexes_)
            for i, s in enumerate(group_size):                
                if group_dim[i]==2:
                    indexes_dim=torch.arange(0,s).cuda(tensor.device)
                    indexes_dim+=pre_size
                    indexes_arr.append(indexes_dim)
                pre_size+=s  
            indexes = torch.cat(indexes_arr, dim=0)
      
            values = tensor.data[indexes]
            # self.residuals[name].data = tensor.data + 0.0 
            # self.residuals[name].data[indexes] = 0.
            # self.values[name] = values
            # self.indexes[name] = indexes
            # self._process_data_after_residual(name, tensor.data)
            
            e_topk_time=time.time()-time_start_
            self.topk_time.append(e_topk_time)
            
            return tensor, indexes, values
    
    # 优化压缩
    def compress_block_opt(self, tensor, name=None, group_size=None, group_dim=None, sigma_scale=2.5, ratio=0.01):
        time_start_ = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)

            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)
            
            tensor_abs=torch.abs(tensor.data)
            numel = tensor.numel()        
            
            self.current_ratio = ratio
            
            if 'fc' in name:
                # case-1, fc层全传输
                # indexes= torch.arange(0, numel).cuda(tensor.device)
                
                # case-2, fc层部分传输
                # ratio=0.1
                # k = max(int(numel * ratio), 1)
                # values_, indexes = torch.topk(tensor_abs, k=k)
                # values = tensor.data[indexes]
                
                
                ratio=0.3
                k = max(int(numel * ratio), 1)
                indexes_bias= torch.arange(0, 100).cuda(tensor.device)                
                values_weight, indexes_weight= torch.topk(tensor_abs[100:], k=k)
                indexes= torch.concatenate((indexes_bias, indexes_weight+100), dim=0)
                values = tensor.data[indexes]
                
                
                # ratio=0.1
                # k = max(int(2048 * ratio), 1)
                # indexes_bias= torch.arange(0, 100).cuda(tensor.device)             
                # values_weight, indexes_weight= torch.topk(tensor_abs[100:].reshape([100, 2048]), k=k,dim=1)
                # tensor_decompressed = torch.zeros(
                #     [100, 2048], dtype=values_weight.dtype, layout=values_weight.layout, device=values_weight.device).cuda()
                # tensor_decompressed.scatter_(1, indexes_weight, values_weight)
                # tensor_=tensor_decompressed.flatten()
                # one_indexes = tensor_ > 0
                # indexes_weight_ = one_indexes.nonzero().data.squeeze().view(-1)                
                # indexes= torch.concatenate((indexes_bias, indexes_weight_+100), dim=0)

                return tensor, indexes, values
            
            k = max(int(numel * ratio), 1)
            values, indexes = torch.topk(tensor_abs, k=k)
            values = tensor.data[indexes]
            
            e_topk_time=time.time()-time_start_
            self.topk_time.append(e_topk_time)
            return tensor, indexes, values
        
        
            
            # self._process_data_before_selecting(name, tensor.data)
            # tensor.add_(self.residuals[name].data)
            pre_size=0
            indexes_arr=[]
            k = max(int(numel * ratio), 1)
            values_, indexes_ = torch.topk(tensor_abs, k=k)
            
            indexes_arr.append(indexes_)
            for i, s in enumerate(group_size):                
                if group_dim[i]==2:
                    indexes_dim=torch.arange(0,s).cuda(tensor.device)
                    indexes_dim+=pre_size
                    indexes_arr.append(indexes_dim)
                pre_size+=s  
            indexes = torch.cat(indexes_arr, dim=0)
      
            values = tensor.data[indexes]
            # self.residuals[name].data = tensor.data + 0.0 
            # self.residuals[name].data[indexes] = 0.
            # self.values[name] = values
            # self.indexes[name] = indexes
            # self._process_data_after_residual(name, tensor.data)
            
            e_topk_time=time.time()-time_start_
            self.topk_time.append(e_topk_time)
            
            return tensor, indexes, values


    # def get_residuals(self, name, like_tensor):
    #     if name not in self.residuals:
    #         self.residuals[name] = torch.zeros_like(like_tensor.data)
    #     return self.residuals[name]

    # def add_residuals(self, included_indexes, name):
    #     with torch.no_grad():
    #         residuals = self.residuals[name]
    #         if type(included_indexes) is np.ndarray:
    #             indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
    #         else:
    #             indexes_t = included_indexes
    #         values = self.values[name]
    #         values.data[indexes_t] = 0.0
    #         residuals.data[self.indexes[name]] += values.data

    def decompress(self, tensor, original_tensor_size):
        return tensor


class EFTopKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'eftopk'

    def compress(self, tensor, name=None, group_size=None, sigma_scale=2.5, ratio=0.01):
        time_start_=time.time()
        with torch.no_grad():            

            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)

            # top-k solution
            numel = tensor.numel()

            self.current_ratio = ratio
            
            if ratio ==1:
            # if ratio ==1 or 'fc' in name:
                numel = tensor.numel()
                values =tensor
                indexes=torch.arange(0,numel).cuda(tensor.device)
                return tensor, indexes, values
            
            tensor.add_(self.residuals[name].data)
            

            k = max(int(numel * ratio), 1)
            _, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.
            
            e_topk_time=time.time() - time_start_
            self.topk_time.append(e_topk_time)

            return tensor, indexes, values




class GaussianCompressor(TopKCompressor):
    """
    """

    def __init__(self):
        super().__init__()
        self.name = 'gaussian'
        self.iterations = {}
        self.sparsities = []

    def compress(self, tensor, name=None,  group_size=None, sigma_scale=3, ratio=0.01):
        time_start_=time.time()
        with torch.no_grad():            
            
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
                
            # classifier.6
            if ratio ==1:
            # if ratio ==1 or 'fc' in name or 'classifier.6' in name:
                numel = tensor.numel()
                values =tensor
                indexes=torch.arange(0,numel).cuda(tensor.device)
                return tensor, indexes, values
            
            tensor.add_(self.residuals[name].data)
            
            
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, thr = utils_optimizer.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            # abs_tensor = torch.abs(tensor)
            # loops = 0
            # while loops < 5:
            # # while loops < 2:
            #     one_indexes = abs_tensor > right_thres
            #     indexes = one_indexes.nonzero().data.squeeze().view(-1)
            #     if indexes.numel() < 2*k/3:
            #         right_thres *= 0.7
            #     elif indexes.numel() > 4*k/3:
            #         right_thres *= 1.3
            #     else:
            #         break
            #     loops += 1
            mask = tensor.abs() >= thr
            selected = mask.sum()

            # for _ in range(5):
            #     if selected > 1.3 * k:
            #         thr = 1.3 * thr
            #     elif selected < 0.7 * numel * k:
            #         thr = 0.7 * thr
            #     else:
            #         break
            #     mask = tensor.abs() >= thr
            #     selected = mask.sum()
            
            for _ in range(5):
                if selected > 1.2 * k:
                    thr = 1.2 * thr
                elif selected < 0.8 * numel * k:
                    thr = 0.8 * thr
                else:
                    break
                mask = tensor.abs() >= thr
                selected = mask.sum()
            
            
            # indexes = indexes[0:k]
            indexes, = torch.where(mask)
            values = tensor.data[indexes]
            
            #print('gaussion vs topk: ', indexes.numel(), k)
            
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            # self.values[name] = values
            # self.indexes[name] = indexes
            # self._process_data_after_residual(name, tensor)
            e_topk_time=time.time() - time_start_
            self.topk_time.append(e_topk_time)
            
            return tensor, indexes, values


class DgcCompressor(TopKCompressor):

    def __init__(self):
        super().__init__()
        self.name='dgc'

    def compress(self, tensor, name=None,  group_size=None, sigma_scale=3, ratio=0.01):
        with torch.no_grad():
            if name not in self.residuals:
                    self.residuals[name] = torch.zeros_like(tensor.data)
            tensor.add_(self.residuals[name].data)
            
            
            shape = tensor.size()
            tensor = tensor.flatten()
            numel = tensor.numel()
            compress_ratio=ratio

            sample_shape = [max(1, int(numel * compress_ratio))]
            sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
            sample_tensor = tensor[sample_index]

            k = max(1, int(numel * compress_ratio * compress_ratio))
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

            indices, = torch.where(mask)
            values = tensor[indices]
            
            
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indices] = 0.0

            # self.values[name] = values
            # self.indexes[name] = indices
            # self._process_data_after_residual(name, tensor)
            return tensor,indices, values
    
class RedSyncCompressor(TopKCompressor):

    def __init__(self):
        super().__init__()
        self.name="redsync"

    
    def compress(self, tensor, name=None,  group_size=None, sigma_scale=3, ratio=0.01):
        with torch.no_grad():
            if name not in self.residuals:
                    self.residuals[name] = torch.zeros_like(tensor.data)
            tensor.add_(self.residuals[name].data)
            numel = tensor.numel()
            compress_ratio=ratio
            
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
            indices, = torch.where(one_indexes)
            indices = indices.cuda(tensor.device)
            values = tensor_flatten[indices]
            
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indices] = 0.0

            # self.values[name] = values
            # self.indexes[name] = indices
            # self._process_data_after_residual(name, tensor)
            return tensor,indices,values
 
class RandomKCompressor(TopKCompressor):
    def __init__(self):
        super().__init__()
        self.global_step = 0
        self.name='randomk'
    
    def compress(self, tensor, name=None,  group_size=None, sigma_scale=3, ratio=0.01):
        with torch.no_grad():
            if name not in self.residuals:
                    self.residuals[name] = torch.zeros_like(tensor.data)
            tensor.add_(self.residuals[name].data)
            
            h = sum(bytes(name, encoding='utf8'), self.global_step)
            self.global_step += 1
            torch.manual_seed(h)
            tensor = tensor.flatten()
            numel = tensor.numel()
            k = max(1, int(numel * ratio))
            indices = torch.randperm(numel, device=tensor.device)[:k]
            values = tensor[indices]
            
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indices] = 0.0

            # self.values[name] = values
            # self.indexes[name] = indices
            # self._process_data_after_residual(name, tensor)
            return tensor,indices,values
    
    
class ExpCompressor(TopKCompressor):
    def __init__(self):
        super().__init__()
        self.i_ratio = 0.25
        self.stages = 1

    
    def compress(self, tensor, name=None,  group_size=None, sigma_scale=3, ratio=0.01):
        with torch.no_grad():
            if name not in self.residuals:
                    self.residuals[name] = torch.zeros_like(tensor.data)
            tensor.add_(self.residuals[name].data)
            numel = tensor.numel()
            tensor_flatten = tensor.flatten().cuda()

            # -------------EXP compress-------------
            i_ratio = self.i_ratio
            stages = self.stages

            ada_stages = 0
            if stages < 0 or i_ratio == 0.0:
                ada_stages = stages
                stages = ExpCompressor.cur_stages

            t_norm = tensor.norm(2)
            ExpCompressor.norm = t_norm
            # abs_norm_tensor = tensor.abs() / t_norm
            abs_norm_tensor = tensor_flatten.abs() / t_norm
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            t_mean = torch.mean(abs_norm_tensor)

            # if stages == 1 or ratio >= NoneCompressor.first_ratio:
            if stages == 1 or ratio >= 0.25:
                threshold = -t_mean * math.log(ratio)
            else:
                # threshold = -t_mean * math.log(NoneCompressor.first_ratio)
                threshold = -t_mean * math.log(0.25)

            # r_ratio = ratio / NoneCompressor.first_ratio
            r_ratio = ratio / 0.25
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    # indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    indexes, = torch.where(one_indexes)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    t_mean = torch.mean(abs_norm_tensor)

                    threshold = -(t_mean - t_min) * math.log(i_ratio) + t_min
                    if i == 1 and stages == 0:
                        threshold = -(t_mean - t_min) * math.log(r_ratio / math.pow(i_ratio, loop - 1)) + t_min
                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            # indexes = one_indexes.nonzero().data.squeeze().view(-1)
            indexes, = torch.where(one_indexes)

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                ExpCompressor.adapt_stages(actual_ratio, ratio, ada_stages)

            indices = indexes.cuda(tensor.device)
            values = tensor_flatten[indices]
            
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indices] = 0.0

            # self.values[name] = values
            # self.indexes[name] = indices
            # self._process_data_after_residual(name, tensor)


            return tensor, indices,values   
    
    
    
compressors = {
        None: NoneCompressor,
        'none': NoneCompressor,
        'topk': TopKCompressor,
        'topkef': EFTopKCompressor,
        'eftopk': EFTopKCompressor, #TopK with error-feedback
        'gaussian': GaussianCompressor, #GaussianK with error-feedback
        'dgc': DgcCompressor,
        'redsync' :RedSyncCompressor,
        'randomk': RandomKCompressor,
        'sidco': ExpCompressor,
        # 'signum': SignCompressor,
        # 'efsignum': EFSignCompressor,
    }


