import horovod.torch as hvd 

def get_compressor(params):
    comp = params.get('compressor', 'none')
    world_size = hvd.size()     
    # 获取当前的rank     
    cur_rank=hvd.rank()

    if comp == 'dgc':
        from adtopk_lib.compressor.dgc import DgcCompressor
        density = params.get('density', 0.3)
        compressor = DgcCompressor(density)
    elif comp == 'none':
        from adtopk_lib.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif comp == 'topk':
        from adtopk_lib.compressor.topk import TopKCompressor
        density = params.get('density', 0.01)
        compressor = TopKCompressor(density,rank=cur_rank)
    
    elif comp == 'gaussiank':
        from adtopk_lib.compressor.gaussiank import GaussiankCompressor
        density = params.get('density', 0.01)
        compressor = GaussiankCompressor(density,rank=cur_rank)
    
    elif comp == 'redsync':
        from adtopk_lib.compressor.redsync import RedSyncCompressor
        density = params.get('density', 0.01)
        compressor = RedSyncCompressor(density,rank=cur_rank)
    elif comp == 'redsynctrim':
        from adtopk_lib.compressor.redsync import RedSyncTrimCompressor
        density = params.get('density', 0.01)
        compressor = RedSyncTrimCompressor(density,rank=cur_rank)
    
    elif comp == 'sidcoexp':
        from adtopk_lib.compressor.sidco import ExpCompressor
        density = params.get('density', 0.01)
        compressor = ExpCompressor(density)
    elif comp == 'sidcogp':
        from adtopk_lib.compressor.sidco import GParetoCompressor
        density = params.get('density', 0.01)
        compressor = GParetoCompressor(density)
    elif comp == 'sidcogam':
        from adtopk_lib.compressor.sidco import GammaGParetoCompressor
        density = params.get('density', 0.01)
        compressor = GammaGParetoCompressor(density)

    elif comp == 'topkef':
        from adtopk_lib.compressor.topkef import TopKEFCompressor
        # density = params.get('density', 0.3)
        density = params.get('density', 0.1)
        model_named_parameters = params.get('model_named_parameters')
        # density = params.get('density', 0.001)
        compressor = TopKEFCompressor(density,rank=cur_rank)
        compressor.initialize(model_named_parameters)
    

  
    
    elif comp == 'randomk':
        from adtopk_lib.compressor.randomk import RandomKCompressor
        # density = params.get('density', 0.3)
        density = params.get('density', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        # density = params.get('density', 0.001)
        compressor = RandomKCompressor(density,rank=cur_rank)
        # compressor.initialize(model_named_parameters)

    

    
    elif comp == 'imbalancetopktime':
        from adtopk_lib.compressor.imbalancetopktime import ImbalanceTopkTimeCompressor
        density = params.get('density', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = ImbalanceTopkTimeCompressor(density, rank=hvd.rank())
        compressor.initialize(model_named_parameters)
        

    else:
        raise NotImplementedError(compressor)
    
    return compressor

def get_memory(params):
    
    mem = params.get('memory', 'none') 
    if mem == 'dgc':
        from adtopk_lib.memory.dgc import DgcMemory
        momentum = params.get('momentum', 0.9)
        gradient_clipping = params.get('gradient_clipping', False)
        memory = DgcMemory(momentum, gradient_clipping)
    elif mem == 'none':
        from adtopk_lib.memory.none import NoneMemory
        memory = NoneMemory()
    # elif mem == 'powersgd':
    #     from adtopk_lib.memory.powersgd import PowerSGDMemory
    #     compress_rank = params.get('compress_rank', 1)
    #     memory = PowerSGDMemory(compressor.q_memory, compress_rank)
    elif mem == 'residual':
        from adtopk_lib.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'residualgtopk':
        from adtopk_lib.memory.residualgtopk import ResidualGlobalTopkMemory
        memory = ResidualGlobalTopkMemory()
    else:
        raise NotImplementedError(mem)

    return memory



def get_communicator(params):
    # 获取当前的进程数
    world_size = hvd.size()
    # 获取当前的rank
    cur_rank=hvd.rank()
    # 获得当前rank     
    rank=params.get('rank', 0)     
    cur_epoch=params.get('cur_epoch')
    
    # communicator默认采用的是allreduce的方法
    comm = params.get('communicator', 'allreduce')
    
    compressor = get_compressor(params)
    memory = get_memory(params)

    # 梯度交换的通信方法
    if comm == 'allreduce':
        from adtopk_lib.communicator.allreduce import Allreduce
        return Allreduce(compressor, memory)
    elif comm == 'allgather':
        from adtopk_lib.communicator.allgather import Allgather
        return Allgather(compressor, memory, world_size)
    else:
        raise NotImplementedError(comm)


    