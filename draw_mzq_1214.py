# import horovod
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np   
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import random
import scipy as sp
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab
import torchvision.models as models
import torch
import torch.nn.functional as F
import scipy.stats
import torchvision.models as models

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt 
import numpy as np


#设置图例并且设置图例的字体及大小     
font1 = {'family' : 'Times New Roman',              
         'weight' : 'normal',              
         'size'   : 14,     }          
#设置横纵坐标的名称以及对应字体格式     
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',              
         'size'   : 14,     }     
font_size=20     
font_size_legend=18          
plt.rcParams['font.family'] = "Times New Roman" 

colors = ['#FA4F4F', '#F89B5C', '#FACD5D', '#C3F448', '#6AFBBC', '#62FAEE', '#34BCF9', '#4468FA', '#874DF8']
colors1 = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2', '#B883D4']

markers = ['.', 'o', 'D', '*', 'P', 'X', 'v', '^']
markers = ['.', '.', '.', '.', '.', '.', '.', '.']



# 绘制减小buffer-1的就绪等待时间的对比
def inter_worker_tensor_wait_time_1212():
    
    dir_path='/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/cifar100_resnet50_design_compare/'
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    
    buffer_size=[11235428, 6168576, 3089152, 2055936, 905472, 190976, 59712]
    
    # Compression ratio = 0.05
    wait_005_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_005/merge_number_noef_wait_005_e80_ytest_acc_1214.txt')
    wait_005_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_005/merge_number_noef_wait_005_e80_xtrain_time_1214.txt')
    
    # wait_005_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9/merge_number_noef_wait_005_9_e80_ytest_acc_1214.txt')
    # wait_005_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9/merge_number_noef_wait_005_9_e80_xtrain_time_1214.txt')
   
    wait_005_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9_1215_1/merge_number_noef_wait_005_9_1215_1_e80_ytest_acc_1214.txt')
    wait_005_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9_1215_1/merge_number_noef_wait_005_9_1215_1_e80_xtrain_time_1214.txt')
   
    
    # Compression ratio = 0.1
    wait_01_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_01/merge_number_noef_wait_01_e80_ytest_acc_1214.txt')
    wait_01_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_01/merge_number_noef_wait_01_e80_xtrain_time_1214.txt')
    
    wait_01_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_01_9/merge_number_noef_wait_01_9_e80_ytest_acc_1214.txt')
    wait_01_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_01_9/merge_number_noef_wait_01_9_e80_xtrain_time_1214.txt')


    # 0.0049215190264643
    # 0.00430134

    # tensor_np_gaussiank =  np.loadtxt(dir_path+"/average_bias_gaussiank_array_epoch_1.txt")
    # tensor_np_redsync =  np.loadtxt(dir_path+"/average_bias_redsync_array_epoch_1.txt")
    
    # x_arr=range(1, len(wait_005_acc)+1)

    label_font_size=26
    tick_font_size=24
    
    legend_font_size=18
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    
    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    # ax1.set_xlim(0.5, 7.5)
    # ax1.set_ylim(0, 0.05)
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    
    # plt.xticks(ticks, ticks, rotation=30, fontsize=15) 

    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Test Accuracy", fontsize=label_font_size)
    # ax1.set_ylabel("Synchronization Time", fontsize =label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index", fontsize =label_font_size)
    ax1.set_xlabel("Training Time (Sec)", fontsize=label_font_size)
    ax1.tick_params(labelsize =tick_font_size)
    
    
    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray', edgecolor='darkgray', alpha=1.0, label='Non-compression')
    # ax1.plot(x_arr, buffer_08, linewidth=2.0, label='Density=0.8')
    
    ax1.plot(wait_005_time, wait_005_acc, marker='o', linewidth=2.0, label='Buffer-1 Wait')
    ax1.plot(wait_005_9_time, wait_005_9_acc, marker='o', linewidth=2.0, label='Buffer-1 Wait Ahead')


    # ax1.plot(wait_01_time, wait_01_acc, marker='o', linewidth=2.0, label='Buffer-1 Wait')
    # ax1.plot(wait_01_9_time, wait_01_9_acc, marker='o', linewidth=2.0, label='Buffer-1 Wait Ahead')


    # ax1.plot(x_arr, buffer_01, marker='o', linewidth=2.0, label='Density=0.1')
    # ax1.plot(x_arr, buffer_005, marker='o', linewidth=2.0, label='Density=0.05')
    # ax1.plot(x_arr, buffer_001, marker='o', linewidth=2.0, label='Density=0.01')    
    # plt.axhline(y=0.0049251, color='black', linewidth=2.0, linestyle='--', label='Avg Backward')
    
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='ASC-WFBP')
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='Ours')
    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    plt.legend(loc = 0, ncol=1, columnspacing=0.3, labelspacing=0.3, fontsize=legend_font_size)

    # plt.legend(bbox_to_anchor=(0.5, -0.2),loc=8,ncol=4,fontsize=legend_font_size) # , borderaxespad=0
    
    # bbox_to_anchor 为相对于(0,0)坐标的位置
    # plt.legend(bbox_to_anchor=(0.5, 1.2),loc=9,ncol=4,columnspacing=0.8, labelspacing=0.8,fontsize=legend_font_size-2) # , borderaxespad=0

    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/distribution/shape_value_bar_original_global_hybrid_distribution_all_dimension.jpg',dpi=750, bbox_inches='tight')
    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/dimension_missing/shape_value_bar_original_global_hybrid_distribution_no_sort_epoch=20_importance.jpg',dpi=750, bbox_inches='tight')

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/cifar100_resnet50_design_compare/figures/'
    plt.savefig(dir_fig +'/merge_number_noef_wait_005_1215_1.jpg', dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig +'/merge_number_noef_wait_005_1215_1.pdf', dpi=750, bbox_inches='tight')
    
    # plt.savefig(dir_fig +'/merge_number_noef_wait_01.jpg', dpi=750, bbox_inches='tight')
    # plt.savefig(dir_fig +'/merge_number_noef_wait_01.pdf', dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return

# inter_worker_tensor_wait_time_1212()


# 绘制选择性压缩的收敛精度图
def inter_worker_tensor_selective_compression_acc_time_1216():
    
    dir_path='/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/cifar100_resnet50_design_compare/'
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    
    buffer_size=[11235428, 6168576, 3089152, 2055936, 905472, 190976, 59712]
    
    # Compression ratio = 0.05
    wait_005_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_005/merge_number_noef_wait_005_e80_ytest_acc_1214.txt')
    wait_005_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_005/merge_number_noef_wait_005_e80_xtrain_time_1214.txt')
    
    # wait_005_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9/merge_number_noef_wait_005_9_e80_ytest_acc_1214.txt')
    # wait_005_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9/merge_number_noef_wait_005_9_e80_xtrain_time_1214.txt')
   
    wait_005_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9_1215_1/merge_number_noef_wait_005_9_1215_1_e80_ytest_acc_1214.txt')
    wait_005_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_005_9_1215_1/merge_number_noef_wait_005_9_1215_1_e80_xtrain_time_1214.txt')
   
    
    # Compression ratio = 0.1
    wait_01_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_01/merge_number_noef_wait_01_e80_ytest_acc_1214.txt')
    wait_01_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_01/merge_number_noef_wait_01_e80_xtrain_time_1214.txt')
    
    wait_01_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_01_9/merge_number_noef_wait_01_9_e80_ytest_acc_1214.txt')
    wait_01_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_01_9/merge_number_noef_wait_01_9_e80_xtrain_time_1214.txt')
    
    # Compression ratio = 0.01
    wait_001_9_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_001_9/merge_number_noef_wait_001_9_e80_ytest_acc_1214.txt')
    wait_001_9_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_001_9/merge_number_noef_wait_001_9_e80_xtrain_time_1214.txt')
   
    wait_001_9_adap_acc  =  np.loadtxt(dir_path+'/merge_number_noef_wait_001_adap_1216/merge_number_noef_wait_001_adap_e80_ytest_acc_1214.txt')
    wait_001_9_adap_time =  np.loadtxt(dir_path+'/merge_number_noef_wait_001_adap_1216/merge_number_noef_wait_001_adap_e80_xtrain_time_1214.txt')
   

    # 0.0049215190264643
    # 0.00430134
    # x_arr=range(1, len(wait_005_acc)+1)
    
    label_font_size=26
    tick_font_size=24    
    legend_font_size=18
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    
    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    # ax1.set_xlim(0.5, 7.5)
    # ax1.set_ylim(0, 0.05)
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    
    # plt.xticks(ticks, ticks, rotation=30, fontsize=15) 
    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Test Accuracy", fontsize=label_font_size)
    # ax1.set_ylabel("Synchronization Time", fontsize =label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index", fontsize =label_font_size)
    ax1.set_xlabel("Training Time (Sec)", fontsize=label_font_size)
    ax1.tick_params(labelsize =tick_font_size)
    
    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray', edgecolor='darkgray', alpha=1.0, label='Non-compression')
    # ax1.plot(x_arr, buffer_08, linewidth=2.0, label='Density=0.8')
    
    ax1.plot(wait_005_time, wait_005_acc, marker='o', linewidth=2.0, label='Density=0.05 (All Buffers)')
    # ax1.plot(wait_005_9_time, wait_005_9_acc, marker='o', linewidth=2.0, label='Density=0.05')
    ax1.plot(wait_001_9_time, wait_001_9_acc, marker='o', linewidth=2.0, label='Density=0.01 (All Buffers)')
    ax1.plot(wait_001_9_adap_time, wait_001_9_adap_acc, marker='o', linewidth=2.0, label='Density=0.01 (Selective Buffers)')

    # ax1.plot(wait_01_time, wait_01_acc, marker='o', linewidth=2.0, label='Buffer-1 Wait')
    # ax1.plot(wait_01_9_time, wait_01_9_acc, marker='o', linewidth=2.0, label='Buffer-1 Wait Ahead')

    # ax1.plot(x_arr, buffer_01, marker='o', linewidth=2.0, label='Density=0.1')
    # ax1.plot(x_arr, buffer_005, marker='o', linewidth=2.0, label='Density=0.05')
    # ax1.plot(x_arr, buffer_001, marker='o', linewidth=2.0, label='Density=0.01')    
    # plt.axhline(y=0.0049251, color='black', linewidth=2.0, linestyle='--', label='Avg Backward')
    
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='ASC-WFBP')
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='Ours')
    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    plt.legend(loc = 0, ncol=1, columnspacing=0.3, labelspacing=0.3, fontsize=legend_font_size)
    # plt.legend(bbox_to_anchor=(0.5, -0.2),loc=8,ncol=4,fontsize=legend_font_size) # , borderaxespad=0

    # bbox_to_anchor 为相对于(0,0)坐标的位置
    # plt.legend(bbox_to_anchor=(0.5, 1.2),loc=9,ncol=4,columnspacing=0.8, labelspacing=0.8,fontsize=legend_font_size-2) # , borderaxespad=0

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/cifar100_resnet50_design_compare/figures/'
    plt.savefig(dir_fig +'/merge_number_noef_selective_compression_1216.jpg', dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig +'/merge_number_noef_selective_compression_1216.pdf', dpi=750, bbox_inches='tight')
    
    # plt.savefig(dir_fig +'/merge_number_noef_wait_01.jpg', dpi=750, bbox_inches='tight')
    # plt.savefig(dir_fig +'/merge_number_noef_wait_01.pdf', dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return

# inter_worker_tensor_selective_compression_acc_time_1216()



backward_time =  6.379696846008301
average_backward_time = 0.0012055
communication_time = [0.002100229263305664, 0.0020492076873779297, 0.002020597457885742, 0.0020248889923095703, 0.0020139217376708984, 0.0014421939849853516, 0.001302957534790039, 0.0012891292572021484, 0.0012848377227783203, 0.0013036727905273438, 0.0012810230255126953, 0.0012843608856201172, 0.0012860298156738281, 0.0008804798126220703, 0.0008893013000488281, 0.000888824462890625, 0.0008270740509033203, 0.0008127689361572266, 0.0008003711700439453, 0.0008029937744140625, 0.0007991790771484375, 0.0007977485656738281, 0.0008103847503662109, 0.0008015632629394531, 0.0008006095886230469, 0.0007593631744384766, 0.0007104873657226562]


# 压缩时间减少了, 反向传播的计算时间也减少了.
backward_time =  3.8903868198394775
step_time =  4.639777421951294
communication_time = [0.0014655590057373047, 0.001436471939086914, 0.0018186569213867188, 0.0013897418975830078]


backward_time =  5.693884372711182
step_time =  4.9869091510772705
communication_time = [0.0016357898712158203, 0.0015342235565185547, 0.0014309883117675781, 0.001432180404663086, 0.0013685226440429688, 0.0013239383697509766, 0.0011334419250488281, 0.0011143684387207031, 0.0011146068572998047, 0.0011043548583984375]



backward_time =  6.136146783828735
step_time =  4.238052129745483
communication_time = [0.001354217529296875, 0.001791238784790039, 0.004488229751586914]


backward_time =  6.3147008419036865
step_time =  6.120984077453613
communication_time = [0.018946409225463867]


backward_time =  5.975486755371094
step_time =  4.545055150985718
communication_time = [0.006887912750244141, 0.0032160282135009766]


backward_time =  9.1145339012146
step_time =  22.56567358970642
communication_time =[161]

# number_buffer = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128]
# density = 0.01
epochs = 10
training_time = 158.72031831741333, 155.688462972641, 154.68866968154907, 153.32315754890442, 152.78905582427979, 152.49521827697754, 157.54267144203186, 162.47674345970154, 175.68011164665222, 220.16909861564636,
255.0232057571411, 413.89867973327637


# density = 0.05
# number_buffer = [1, 2, 3, 4, 5, 6, 7, 8, 16,  20,  32, 64, 128]
# per_number_buffer=[161, 81, 54, 41, 33, 27, 23, 21, 11] 
training_time = 233.52583241462708, 217.86393570899963, 199.04457807540894, 194.54913330078125, 193.9619541168213, 195.07850527763367, 197.93249106407166, 199.4943664073944, 203.54408502578735


# 33 = 197.93249106407166



# Density = 0.1
# number_buffer = [1, 2, 3, 4, 5, 6, 7, 8,   10,   12,  14,  16, 32,  64, 128]
# per_number_buffer=[161, 81, 54, 41, 33, 27, 23, 21,  17,  14,   11,  9, ] 6
training_time = 337.1375403404236, 321.04374170303345, 296.68427658081055, 292.11885261535645, 287.84505438804626, 288.1908504962921, 285.73701190948486, 283.7078912258148, 281.7317433357239, 282.8498909473419
289.4772641658783, 293.6497769355774, 324.4949188232422


# wait ahead number_buffer=7
training_time = 266.68830156326294, 266.01109409332275



# buffer划分数量的对比
def inter_worker_tensor_buffer_number_compare_1217():
    
    dir_path = '/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/cifar100_resnet50_design_compare/'
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    
    buffer_size = [11235428, 6168576, 3089152, 2055936, 905472, 190976, 59712]
    
    # compression ratio = 0.01, epochs = 10
    buffer_number_001 = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128]
    training_time_001 = [158.72031831741333, 155.688462972641, 154.68866968154907, 153.32315754890442, 152.78905582427979, 152.49521827697754, 156.54267144203186,162.47674345970154, 175.68011164665222, 220.16909861564636, 255.0232057571411, 413.89867973327637]
    x = range(len(training_time_001))
    
    # training_time_005 = [233.52583241462708, 217.86393570899963, 199.04457807540894, 194.54913330078125, 193.9619541168213, 195.07850527763367, 189.2605516910553, 193.4943664073944, 203.54408502578735]
    training_time_005 = [233.52583241462708, 217.86393570899963, 199.04457807540894, 194.54913330078125, 193.9619541168213, 194.07850527763367, 197.93249106407166, 199.4943664073944, 203.54408502578735]


    # 0.0049215190264643
    # 0.00430134
    # x_arr=range(1, len(wait_005_acc)+1)
    
    label_font_size=26
    tick_font_size=24    
    legend_font_size=18
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    
    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    # ax1.set_xlim(0.5, 7.5)
    # ax1.set_ylim(0, 0.05)
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # 调用xticks进行设置横坐标的显示值
    plt.xticks(x, buffer_number_001, fontsize=tick_font_size) 
    
    # plt.xticks(ticks, ticks, rotation=30, fontsize=15) 
    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Test Accuracy", fontsize=label_font_size)
    # ax1.set_ylabel("Synchronization Time", fontsize =label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index", fontsize =label_font_size)
    ax1.set_xlabel("Training Time (Sec)", fontsize=label_font_size)
    ax1.tick_params(labelsize =tick_font_size)
    

    ax1.plot(x[:9], training_time_001[:9], marker='o', linewidth=2.0, label='Density=0.01')
    
    ax1.plot(x[:9], training_time_005[:9], marker='o', linewidth=2.0, label='Density=0.05')

    # ax1.plot(x_arr, buffer_01, marker='o', linewidth=2.0, label='Density=0.1')
    # ax1.plot(x_arr, buffer_005, marker='o', linewidth=2.0, label='Density=0.05')
    # ax1.plot(x_arr, buffer_001, marker='o', linewidth=2.0, label='Density=0.01')    
    # plt.axhline(y=0.0049251, color='black', linewidth=2.0, linestyle='--', label='Avg Backward')
    
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='ASC-WFBP')
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='Ours')
    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    plt.legend(loc = 0, ncol=1, columnspacing=0.3, labelspacing=0.3, fontsize=legend_font_size)
    # plt.legend(bbox_to_anchor=(0.5, -0.2),loc=8,ncol=4,fontsize=legend_font_size) # , borderaxespad=0

    # bbox_to_anchor 为相对于(0,0)坐标的位置
    # plt.legend(bbox_to_anchor=(0.5, 1.2),loc=9,ncol=4,columnspacing=0.8, labelspacing=0.8,fontsize=legend_font_size-2) # , borderaxespad=0

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/buffer_number_compare_fig/'
    plt.savefig(dir_fig +'/resnet50_merge_number_buffer_compare_001_1217.jpg', dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig +'/resnet50_merge_number_buffer_compare_001_1217.pdf', dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()
    return

# inter_worker_tensor_buffer_number_compare_1217()








gradient_size = [100, 204800, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 2097152, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 32768, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 4096, 64, 64, 1728]

sub_buffer = [6, 19, 25, 28, 28, 28, 28]

communication_time = 15 

len_gradient_size = len(gradient_size)

# print(len(gradient_size))

gradient_sum = sum(gradient_size)
backward_time = 6.662214994430542

per_elements_backward_time = backward_time/gradient_sum

per_elements_backward_time_ = backward_time/len_gradient_size

per_elements_communication_time = communication_time/gradient_sum
# 最优梯度合并
def optimal_gradient_merging_1218():
    sub_buffer =[11, 20, 131]
    
    sub_buffer =[6, 25, 131]
    
    # [1,0]<stdout>:3618916
    # [1,0]<stdout>:11025920
    # [1,0]<stdout>:9060416
    
    # sum(sub_buffer[1:])
    sub_backward_time = sum(gradient_size[sub_buffer[1]:])* per_elements_backward_time
    
    sub_backward_time_ = sum(sub_buffer[1:]) * per_elements_backward_time_
    

    buffer_1_communication_time = sum(gradient_size[:sub_buffer[1]])* per_elements_communication_time

    buffer_2_communication_time = sum(gradient_size[sub_buffer[1]:sub_buffer[1]+sub_buffer[2]])* per_elements_communication_time
    
    buffer_2_backward_time = sum(gradient_size[sub_buffer[1]:sub_buffer[1]+sub_buffer[2]])* per_elements_backward_time


    print('sub_backward_time = ', sub_backward_time)
    print('sub_backward_time_ = ', sub_backward_time_)


    print('buffer_1_communication_time = ', buffer_1_communication_time)
    
    print('buffer_2_backward_time = ', buffer_2_backward_time)
    
    print('buffer_2_communication_time = ', buffer_2_communication_time)
    
    
    
    for i in sub_buffer:
        
        continue
        
    
    return


# optimal_gradient_merging_1218()


# 测量通信时间
x1 = [0.0014472007751464844, 0.008694887161254883, 0.02558279037475586, 0.03646540641784668]

# 明显是隐藏到了反向传播计算当中
x2 = [0.0014257431030273438, 0.008941173553466797, 0.024686574935913086, 0.03651714324951172]
# [1,0]<stdout>:1257572
# [1,0]<stdout>:6822912
# [1,0]<stdout>:6564352
# [1,0]<stdout>:9060416


x1 = [0.10688138008117676], [0.1131587028503418]
x2 = [0.10785865783691406]
# [1,0]<stdout>:23705252

# [1,0]<stdout>:21281892
# [1,0]<stdout>:2423360
x1 = [0.08834028244018555, 0.012144804000854492], [0.08702230453491211, 0.012292623519897461]
x1 = 0.0165673853183279

# 0.104907 = 0.08834028244018555 + 0.0165673853183279
# 0.11704738 = 0.104907 +0.012144804000854492
# 0.0331


# [1,0]<stdout>:17668196
# [1,0]<stdout>:5159936
# [1,0]<stdout>:877120
x1 = [0.06421184539794922, 0.021883487701416016, 0.004156827926635742], [0.06210184097290039, 0.022228717803955078, 0.0048177242279052734]
# 0.02206




gradient_size = [100, 204800, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 2097152, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 32768, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 4096, 64, 64, 1728]

communication_time = 0.10785865783691406
startup_time = 0.0008780956268310547
backward_time = 6.499021768569946/196
len_gradient_size = len(gradient_size)

# 反向传播时间按数量计算
per_elements_backward_time = backward_time/len_gradient_size

# 通信时间按数据量计算
per_elements_communication_time = (communication_time-startup_time)/gradient_sum

per_elements_backward_time_len = backward_time/len_gradient_size
per_elements_backward_time_size = backward_time/gradient_sum

buffer_compression_time= 0.06595400047302246



def communication_time_measure():
    
    
    group =[]
    group_size =0
    group_len =0
    
    min_diff =1
    index_i =0
    min_group_len =0
    
    # buffer_number = 1
    num =1
    for i, s in enumerate(gradient_size):
        group.append(s)
        group_len = len(group)
        group_size = sum(group)
        
        sun_backward_time_group= (len_gradient_size- group_len) * per_elements_backward_time
        communication_time_group = group_size * per_elements_communication_time + startup_time * num
        
        diff= abs(sun_backward_time_group - communication_time_group)
        if diff <min_diff:
            min_diff = diff
            index_i =i
            min_group_len = group_len
        
        print('Group len= ', group_len,',', diff)
    
    print('min_diff = ', min_diff)
    # print('index_i= ', index_i)
    print('min_group_len = ', min_group_len)
    backward_time_group= (group_len) * per_elements_backward_time
    print('backward_time_group = ', backward_time_group)
    
    
    # buffer_number = 2
    num = 2
    for i, s in enumerate(gradient_size):

        # buffer-1
        group.append(s)
        group_len = len(group)
        group_size = sum(group)
    
    
    
    group =[]
    group_len =0
    group_size =0
    
    min_diff = 1
    min_group_len =0
    min_group_len_array= []
    merging_time_array= []
    
    pre_merging_time =0
    print(gradient_size)
    for i, s in enumerate(gradient_size):
        group.append(s)
        group_len = len(group)
        group_size = sum(group)
        
        merging_time = startup_time + group_size * per_elements_communication_time + group_len * per_elements_backward_time_len + buffer_compression_time
        # merging_time = startup_time + group_size * per_elements_communication_time + group_size * per_elements_backward_time_size +buffer_compression_time
        
        layer_time = startup_time * group_len + group_size* per_elements_communication_time + buffer_compression_time* group_len
        
        # group_backward_time = group_size * per_elements_backward_time_size
        group_backward_time = group_len * per_elements_backward_time_len
        
        diff = merging_time -layer_time
        
        if diff<0:
            abs_diff= abs(diff)
            if abs_diff<min_diff:
                
                # buffer-1的通信时间小于buffer-2的反向传播时间
                if len(min_group_len_array)>1 and pre_merging_time> group_backward_time:
                    continue
                    
                
                min_diff = abs_diff
                min_group_len = group_len
                min_group_len_array.append(min_group_len)
                merging_time_array.append(merging_time)
                
                print('min_diff = ', min_diff)
                # print('index_i= ', index_i)
                print('min_group_len = ', min_group_len)
                
                group =[]
                group_len =0
                group_size =0
                min_diff = 1
                min_group_len =0
                
                merging_time_array_sum=sum(merging_time_array)
                
                sub_backward_time= sum(gradient_size[min_group_len_array[0]:])*per_elements_backward_time_size+buffer_compression_time
                
                if merging_time_array_sum>=sub_backward_time:
                    break
            
            pre_merging_time =  merging_time
        
        
                
                
            # print('Group len= ', group_len,',diff= ', diff)
    
    
    # print('min_diff = ', min_diff)
    # # print('index_i= ', index_i)
    # print('min_group_len = ', min_group_len)
    print('min_group_len_array = ', min_group_len_array)

    print('min_group_len_array_sum = ', sum(min_group_len_array))
    
    
    
    return


# communication_time_measure()


gradient_size = [100, 204800, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 2097152, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 32768, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 4096, 64, 64, 1728]

communication_time = 0.10785865783691406
startup_time = 0.0008780956268310547
backward_time = 6.499021768569946/196
# 反向传播时间按数量计算
per_elements_backward_time = backward_time/len_gradient_size
# 通信时间按数据量计算
per_elements_communication_time = (communication_time-startup_time)/gradient_sum
per_elements_backward_time_len = backward_time/len_gradient_size
per_elements_backward_time_size = backward_time/gradient_sum


buffer_compression_time= 0.06595400047302246/196


# 最优的梯度合并方案
def optimal_gradient_merging():
    group =[]
    group_len =0
    group_size =0
    
    min_diff = 1
    min_group_len =0
    min_group_len_array= []
    merging_time_array= []
    merging_communication_time_array = []
    group_backward_time_size_array = []
    
    pre_merging_time =0

    print(gradient_size)
    flag= False
    for i, s in enumerate(gradient_size):
        group.append(s)
        group_len = len(group)
        group_size = sum(group)
        
        
        # merging_time_len = startup_time + group_size * per_elements_communication_time + group_len * per_elements_backward_time_len 
        merging_time_size = startup_time + group_size * per_elements_communication_time   
        
        merging_time_wait_size = startup_time + group_size * per_elements_communication_time + group_size * per_elements_backward_time_size  
        
        
        merging_communication_time_size = startup_time + group_size * per_elements_communication_time
        
        layer_time = startup_time * group_len + group_size* per_elements_communication_time 
        # + buffer_compression_time*group_len
        
        group_backward_time_size = group_size * per_elements_backward_time_size 
        # group_backward_time_len = group_len * per_elements_backward_time_len+ buffer_compression_time
        
        diff = merging_time_wait_size -layer_time

        if diff < 0:
            # abs_diff = abs(diff)
            abs_diff = diff
            
            if abs_diff < min_diff:
            # if True:
                
                # buffer-1的通信时间小于buffer-2的反向传播时间, 
                # if len(min_group_len_array) >0  and pre_merging_time> group_backward_time_size:
                #     continue

                min_diff = abs_diff
                min_group_len = group_len
                min_group_len_array.append(min_group_len)
                merging_time_array.append(merging_time_size)
                merging_communication_time_array.append(merging_communication_time_size)
                group_backward_time_size_array.append(group_backward_time_size)

                # print('min_diff = ', min_diff)
                # print('index_i= ', index_i)
                # print('min_group_len = ', min_group_len)
                
                if flag:
                    last_min_group_len = len_gradient_size - sum(min_group_len_array) 
                    min_group_len_array.append(last_min_group_len)
                    
                    print('break!')
                    break

                group = []
                group_len = 0
                group_size = 0
                min_diff = 1
                min_group_len = 0
                
                # merging_time_array_sum= sum(merging_time_array)
                merging_communication_time_array_sum =sum(merging_communication_time_array)
                
                sub_backward_time= sum(gradient_size[min_group_len_array[0]:])*per_elements_backward_time_size
                # +buffer_compression_time*(len(min_group_len_array)-1)
                # sub_backward_time= (group_len- min_group_len_array[0])*per_elements_backward_time_len+ buffer_compression_time
                
                # if merging_time_array_sum>sub_backward_time:
                #     break

                # if merging_communication_time_array_sum > backward_time:    
                #     flag= True
                    
                pre_merging_time =  merging_time_size
        
            print('Group len= ', group_len,',diff= ', diff)
    
    
    # print('min_diff = ', min_diff)
    # # print('index_i= ', index_i)
    # print('min_group_len = ', min_group_len)
    print('min_group_len_array = ', min_group_len_array)

    print('min_group_len_array_sum = ', sum(min_group_len_array))

    print('len(min_group_len_array) = ', len(min_group_len_array))
    print('len(group_backward_time_size_array) = ', len(group_backward_time_size_array))
    print('len(merging_communication_time_array) = ', len(merging_communication_time_array))


    # 遍历贪心生成的group
    groups = []
    for i, group in enumerate(min_group_len_array):        
        group_backward_time_sum = sum(group_backward_time_size_array[1:])
        merging_communication_time_array_sum =sum(merging_communication_time_array[:i])
        
        groups.append(group)
         # 最小化非重叠buffer数量
        if merging_communication_time_array_sum > group_backward_time_sum:
            last_min_group_len = len_gradient_size - sum(groups) 
            groups.append(last_min_group_len)
            break

        # groups.append(group)     
        # 减少小buffer的数量
        # if len(groups)>1 and sum():
            
            # 进一步合并


    print('groups = ',groups)
    return


optimal_gradient_merging()





# 最优梯度合并对比图
def optimal_number_buffer_size_bar():
    dir_path='/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/bias_threshold/tensor_fusion/acc/'
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    
    # buffers= 5
    training_time = [270.99489521980286, 287.84505438804626, 258.62292742729187]
    
    label_font_size=26
    tick_font_size=24    
    legend_font_size=18
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax2 = fig.add_subplot(111) 
    
    
    ax2.set_ylim(200, 320)
    
    width=0.3
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  
    # ax2.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax2.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax2.yaxis.get_offset_text().set_fontsize(14)   #设置1e6的大小与位置
    
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    labels = ['Fixed Size', 'Fixed Number', 'Ours(Optimal)']
    
    # plt.xticks(ticks, labels, fontsize=tick_font_size)  # 调用xticks进行设置

    ax2.tick_params('y', labelsize=tick_font_size)  #刻度字体大小16
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    ticks = [1, 2, 3] 
    # labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    plt.xticks(ticks, labels, fontsize=tick_font_size)  # 调用xticks进行设置    

    ax2.bar(ticks, training_time, zorder=0, color=default_color[0],width = width, hatch='x', edgecolor='white')

    ax2.set_ylabel("Training Time (Sec)", fontsize=label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index",fontsize=label_font_size)
    ax2.set_xlabel("Merge Strategy", fontsize=label_font_size)
    
    ax2.grid(axis='y',linestyle='--',)
    
    
    # 显示数据标签
    for a,b in zip(ticks, training_time):
        plt.text(a,b,
                 round(b,2),
                 ha='center', 
                 va='bottom',
                 size=15,
                )
    
    # # 边框隐藏
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.spines['left'].set_visible(False)

    # plt.legend(loc = 0, ncol=1, columnspacing=0.3, labelspacing=0.3, fontsize=legend_font_size)
    # plt.legend(bbox_to_anchor=(0.5, -0.2),loc=8,ncol=4,fontsize=legend_font_size) # , borderaxespad=0

    # bbox_to_anchor 为相对于(0,0)坐标的位置
    # plt.legend(bbox_to_anchor=(0.5, 1.2),loc=9,ncol=4,columnspacing=0.8, labelspacing=0.8,fontsize=legend_font_size-2) # , borderaxespad=0
    
    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/mgwfbp/result/buffer_number_compare_fig/'
    plt.savefig(dir_fig+'/optimal_number_buffer_size.jpg', dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig+'/optimal_number_buffer_size.pdf', dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return


# optimal_number_buffer_size_bar()



# 测试DeAR
def test_dear():
    import os
    from setuptools import setup
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, _find_cuda_home
    os.system('pip uninstall -y comm_core')
    CUDA_DIR = _find_cuda_home()
    print('CUDA_DIR: ', CUDA_DIR)
    
    
    return


# test_dear()
