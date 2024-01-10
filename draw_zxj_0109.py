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

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline



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

colors_default = ['#FA4F4F', '#F89B5C', '#FACD5D', '#C3F448', '#6AFBBC', '#62FAEE', '#34BCF9', '#4468FA', '#874DF8']
colors1 = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2', '#B883D4']

markers = ['.', 'o', 'D', '*', 'P', 'X', 'v', '^']
markers = ['.', '.', '.', '.', '.', '.', '.', '.']



# FGBuff
# Training Throughput and Convergence Accuracy
# 绘制相关方法训练吞吐量对比图, Imagenet
def test_bar_throughput_related_resnet101():
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/throughput/resnet101/"

    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    # filenames = ['8gpu', 'actopk_830', 'dgc', 'gaussiank', 'redsync']
    
    labels=['4','8','16','32','64']
    x = np.arange(len(labels))*1.4
    
    y_se =[672, 1326.08, 2150.4, 0, 0]
    
    # Horovod Gaussiank
    y_ho =[192, 299.52, 269.4736, 445.2173, 935.159817]
    # batch_size * nodes * it/s
    
    y_om =[638.72, 1157.12, 2257.92, 0, 0]
    y_de =[601.6, 1187.84, 2288.64, 0, 0]
    # y_baseline =[183.04,386.56,629.76,1218.56,1988.349]
    y_fg =[646.4, 1221.12, 2350.08, 0, 0]

    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    y_actopk =[243.2,565.76,870.4,1638.4,2314.24]    
    y_linear =[2200/2, 2200, 2200*2, 2200*3, 2200*4]
    
    # y = [5.72, 7.26, 3.64, 12.06, 15.36, 19.26, 25.2, 20.9, 0.98, 12.14, 30.67]
    width = 0.2 #the width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=24
    legend_font_size=22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)
    
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(14)#设置1e6的大小与位置

    # ax.set_xlim(-0.3, 6.9)    
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    # plt.bar(x, y, width = width, color='blue')
    
    ax.bar(x+width*0, y_se, width = width, label='SE-N', hatch='\\', edgecolor='white')
    ax.bar(x+width*1, y_ho, width = width, label='HO-N', hatch='/', edgecolor='white' ) 
    ax.bar(x+width*2, y_om, width = width, label='OM-N', hatch='x', edgecolor='white' ) 
    ax.bar(x+width*3, y_de, width = width, label='DE-N', hatch='//', edgecolor='white' )
    ax.bar(x+width*4, y_fg, width = width, label='FG-N', hatch='\\\\', edgecolor='white' )
    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    # ax.bar(x+width*5, y_actopk, width = width,label='ADTopk', hatch='/', edgecolor='white')
    
    # ax.bar(x+2.5*width, y_linear, width = 6*width,label='Linear-Scaling', fill=False,edgecolor='black',linewidth=1.)

    # ax.bar(x+width/2, y3, width = 2*width,label='Top-1 Test Accuracy', fill=False,edgecolor='black',linewidth=1.5)
    # ax.bar(x+width/2, y[::-1], width = width,label='Channel Missing Rate')
    ax.set_xticks([i+0.4 for i in x], labels)
    
    #set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    ax.set_ylabel('Images/Sec', fontsize=label_font_size)
    ax.set_xlabel('The Number of GPUs', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)

    # for a,b,i in zip(x,y,range(len(x))): # zip 函数 
    #     ax.text(a+width/2,b+0.01,"%.2f"%y[i],ha='center',fontsize=10) # plt.text 函数
    
    # # for a,b,i in zip(x,y[::-1],range(len(x))): # zip 函数 
    # #     ax.text(a+width/2,b+0.01,"%.2f"%y[i],ha='center',fontsize=10) # plt.text 函数

    # for a,b,i in zip(x,y2,range(len(x))): # zip 函数 
    #     ax.text(a+width*3/2,b+0.01,"%.2f"%y2[i],ha='center',fontsize=10) # plt.text 函数

    # 输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    # ax.legend(loc = 0, ncol=2,  columnspacing=0.5, labelspacing=0.5,frameon=False, fontsize=legend_font_size)
    
    # ax.legend(loc = 0, ncol=2, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    ax.legend(loc = 0, ncol=2)
    fig.savefig(dst_path + 'resnet101_training_throughput_0109' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'resnet101_training_throughput_0109' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

test_bar_throughput_related_resnet101()

# Imagenet
def test_bar_throughput_related_vgg19():
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/throughput/vgg19/"

    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    # filenames = ['8gpu', 'actopk_830', 'dgc', 'gaussiank', 'redsync']
    
    labels=['4','8','16','32','64']
    x = np.arange(len(labels))*1.4
    
    
    y_se =[ 574.72, 1098.24, 1930.24, 0, 0]
    y_ho =[273.12,394.24,624.64,825.8064,1304.458]
    y_om =[ 582.4, 1131.52, 2176, 0, 0]
    y_de =[ 583.68, 1146.88, 2155.52, 0, 0]
    y_fg =[ 587.52, 1141.76, 2216.96, 0, 0]

    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    y_actopk =[327.68,596.48,896,1157.12,1735.593]
    y_linear =[4500/2, 4500, 4500*2, 4500*3, 4500*4]
    
    # y = [5.72, 7.26, 3.64, 12.06, 15.36, 19.26, 25.2, 20.9, 0.98, 12.14, 30.67]
    width = 0.2 #the width of the bars: can also be len(x) sequence

    label_font_size=26
    tick_font_size=24
    legend_font_size=22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)
    
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(14)#设置1e6的大小与位置
    
    # ax.set_xlim(-0.3, 6.9)
    # ax.set_ylim(0, 1850)
    
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    labels_ = ['Baseline', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # plt.bar(x, y, width = width, color='blue')
    
    ax.bar(x+width*0, y_se, width = width, label='SE-N', hatch='\\', edgecolor='white')
    ax.bar(x+width*1, y_ho, width = width, label='HO-N', hatch='/', edgecolor='white' ) 
    ax.bar(x+width*2, y_om, width = width, label='OM-N', hatch='x', edgecolor='white' ) 
    ax.bar(x+width*3, y_de, width = width, label='DE-N', hatch='//', edgecolor='white' )
    ax.bar(x+width*4, y_fg, width = width, label='FG-N', hatch='\\\\', edgecolor='white' )

    # ax.bar(x+width*5, y_actopk, width = width,label='ADTopk', hatch='/', edgecolor='white')
    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    # ax.bar(x+2.5*width, y_linear, width = 6*width,label='Linear-Scaling', fill=False,edgecolor='black',linewidth=1.)
    # ax.bar(x+width/2, y3, width = 2*width,label='Top-1 Test Accuracy', fill=False,edgecolor='black',linewidth=1.5)
    # ax.bar(x+width/2, y[::-1], width = width,label='Channel Missing Rate')
    ax.set_xticks([i+0.4 for i in x], labels)
    
    #set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    ax.set_ylabel('Images/Sec', fontsize=label_font_size)
    ax.set_xlabel('The Number of GPUs', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    # ax.legend(loc = 0, ncol=2, columnspacing=0.5, labelspacing=0.5, frameon=False, fontsize=legend_font_size)
    # ax.legend(loc = 0, ncol=2, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2,frameon=False, fontsize=legend_font_size)
    ax.legend(loc = 0, ncol=2)

    # fig.savefig(dst_path + 'vgg19_training_throughput_1103' + ".jpg", dpi=750, bbox_inches='tight') 
    # fig.savefig(dst_path + 'vgg19_training_throughput_1103' + ".pdf", dpi=750, bbox_inches='tight')

    fig.savefig(dst_path + 'vgg19_training_throughput_0109' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'vgg19_training_throughput_0109' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

test_bar_throughput_related_vgg19()


def test_bar_throughput_related_lstm():
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/throughput/lstm/"
    
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    # filenames = ['8gpu', 'actopk_830', 'dgc', 'gaussiank', 'redsync']
    
    labels=['4','8','16','32','64']
    x = np.arange(len(labels))*1.4
    
    y_se =[1289.98699,1776.71641,2993.133047,4441.79104,5812.5]
    y_ho =[2371.31474,3213.8228,3460.465,4408.8888,5490.7749]
    y_om =[2185.022026,2526.315789,2917.64705,3501.17647,4560.36256]
    y_de =[2214.285714,2981.96392,3412.844036,3497.06227,4251.4285]

    
    y_fg =[2633.6283,3642.5948,4831.1688,5835.294,6526.315789]
    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    y_linear =[5900/2, 5900, 5900*2, 5900*3, 5900*4]
    

    # y = [5.72, 7.26, 3.64, 12.06, 15.36, 19.26, 25.2, 20.9, 0.98, 12.14, 30.67]
    width = 0.2 #the width of the bars: can also be len(x) sequence

    tick_font_size=24
    label_font_size=26
    legend_font_size=22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)
    
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(14)#设置1e6的大小与位置
    
    # ax.set_xlim(-0.5, 5.9)
    # ax.set_ylim(0, 7500)
    
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels_ = ['Baseline', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # plt.bar(x, y, width = width, color='blue')
    
    ax.bar(x+width*0, y_se, width = width, label='SE-N', hatch='\\', edgecolor='white')
    ax.bar(x+width*1, y_ho, width = width, label='HO-N', hatch='/', edgecolor='white' ) 
    ax.bar(x+width*2, y_om, width = width, label='OM-N', hatch='x', edgecolor='white' ) 
    ax.bar(x+width*3, y_de, width = width, label='DE-N', hatch='//', edgecolor='white' )
    ax.bar(x+width*4, y_fg, width = width, label='FG-N', hatch='\\\\', edgecolor='white' )
    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    
    # ax.bar(x+2*width, y_linear, width = 5*width,label='Linear-Scaling', fill=False,edgecolor='black',linewidth=1.)

    # ax.bar(x+width/2, y3, width = 2*width,label='Top-1 Test Accuracy', fill=False,edgecolor='black',linewidth=1.5)
    # ax.bar(x+width/2, y[::-1], width = width,label='Channel Missing Rate')
    ax.set_xticks([i+0.4 for i in x], labels)
    
    #set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    ax.set_ylabel('Sequences/Sec', fontsize=label_font_size)
    ax.set_xlabel('The Number of GPUs', fontsize=label_font_size)
    
    ax.tick_params('x', labelsize = tick_font_size)  #刻度字体大小16
    ax.tick_params('y', labelsize = tick_font_size)  #刻度字体大小16

    ax.grid(axis='y', linestyle='--',)

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    # ax.legend(loc = 0, ncol=1, columnspacing=0.5, labelspacing=0.5,frameon=False, fontsize=legend_font_size-1)
    ax.legend(loc = 0, ncol=2, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2,frameon=False, fontsize=legend_font_size)


    fig.savefig(dst_path + 'lstm_training_throughput_0102' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'lstm_training_throughput_0102' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_throughput_related_lstm()


def test_bar_throughput_related_bert():
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/throughput/bert/"

    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    # filenames = ['8gpu', 'actopk_830', 'dgc', 'gaussiank', 'redsync']
    
    labels=['4','8','16','32','64']
    x = np.arange(len(labels)) * 1.4
    
    y_se =[7557.12,13885.44,28228.48,50118.08,72739.6226]
    y_ho =[17923.52,27817.6,36680.575,41507.6923,49907.4626]
    y_om =[18986.24,28012.48,37968.22,43897.931,53665.346]
    y_de =[17196.8,25029.44,34504.76,40529.192,48665.346]
    y_fg =[29736.96,48414.72,62177.28,86507.52,103219.2]
    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    y_linear =[45000/2, 45000, 45000*2, 45000*3, 45000*4]
    

    # y = [5.72, 7.26, 3.64, 12.06, 15.36, 19.26, 25.2, 20.9, 0.98, 12.14, 30.67]
    width = 0.2 #the width of the bars: can also be len(x) sequence

    label_font_size =26
    tick_font_size =24    
    legend_font_size =22
    
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)
    
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(14)#设置1e6的大小与位置
    
    # ax.set_xlim(-0.5, 5.9)
    
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    # plt.bar(x, y, width = width, color='blue')
    
    ax.bar(x+width*0, y_se, width = width, label='SE-N', hatch='\\', edgecolor='white')
    ax.bar(x+width*1, y_ho, width = width, label='HO-N', hatch='/', edgecolor='white' ) 
    ax.bar(x+width*2, y_om, width = width, label='OM-N', hatch='x', edgecolor='white' ) 
    ax.bar(x+width*3, y_de, width = width, label='DE-N', hatch='//', edgecolor='white' )
    ax.bar(x+width*4, y_fg, width = width, label='FG-N', hatch='\\\\', edgecolor='white' )

    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg

    # ax.bar(x+2*width, y_linear, width = 5*width, label='Linear-Scaling', fill=False,edgecolor='black',linewidth=1.)

    # ax.bar(x+width/2, y3, width = 2*width,label='Top-1 Test Accuracy', fill=False,edgecolor='black',linewidth=1.5)
    # ax.bar(x+width/2, y[::-1], width = width,label='Channel Missing Rate')
    ax.set_xticks([i+0.4 for i in x], labels)
    
    #set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    ax.set_ylabel('Sequences/Sec', fontsize=label_font_size)
    ax.set_xlabel('The Number of GPUs', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    # columnspacing=0.3, labelspacing=0.3,
    # ax.legend(loc = 0, ncol=1,  columnspacing=0.5, labelspacing=0.5,frameon=False, fontsize=legend_font_size)
    ax.legend(loc = 0, ncol=2, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2,frameon=False, fontsize=legend_font_size)


    fig.savefig(dst_path + 'bert_training_throughput_0102' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'bert_training_throughput_0102' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_throughput_related_bert()



# 绘制相关方法收敛精度对比图
# LSTM + WikiText-103
def draw_lstm_ppl_compare_related():
    label_font_size=26   
    tick_font_size=24
    legend_font_size=22 

    filenames = ['8gpu', 'actopk_830', 'dgc', 'gaussiank', 'redsync']
    labels = ['Baseline', 'ADTopk', 'DGC', 'Gaussiank', 'Redsync']

    num = len(filenames)

    fig = plt.figure(figsize=(8, 4))      
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(90,450)
    
    # LSTM + Baseline
    dir_path= '/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/lstm_data828'
    tensor_np_time_baseline =  np.loadtxt(dir_path + "/time_8gpu.txt")
    tensor_np_acc_baseline =  np.loadtxt(dir_path + "/val_8gpu.txt")
    tensor_np_time_baseline_x =[tensor_np_time_baseline[20]*i for i in range(120)]
    
    dir_path = '/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours'
    # dir_path = "/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys"
    # FGBuff-Ours
    tensor_np_time_fgbuff =  np.loadtxt(dir_path + "/result_train_ours/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_xtrain_time_0107.txt")
    tensor_np_acc_fgbuff =  np.loadtxt(dir_path + "/result_train_ours/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_ytrain_acc_0107.txt")
    tensor_np_time_fgbuff_x =[tensor_np_time_fgbuff[80]*i for i in range(120)]

    # OMGS
    tensor_np_time_omgs =  np.loadtxt(dir_path + "/result_train_omgs/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_xtrain_time_0107.txt")
    tensor_np_acc_omgs =  np.loadtxt(dir_path + "/result_train_omgs/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_ytrain_acc_0107.txt")
    tensor_np_time_omgs_x =[tensor_np_time_omgs[20]*i for i in range(120)]
    
    # DEAR
    tensor_np_time_dear =  np.loadtxt(dir_path + "/result_train_dear/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_xtrain_time_0107.txt")
    tensor_np_acc_dear =  np.loadtxt(dir_path + "/result_train_dear/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_ytrain_acc_0107.txt")
    tensor_np_time_dear_x =[tensor_np_time_dear[20]*i for i in range(120)]
    
    tensor_np_time_horovod =  np.loadtxt(dir_path + "/result_train_horovod/wiki2_lstm/wiki2_lstm_gaussiank_ef_epoch_120_001/wiki2_lstm_gaussiank_ef_epoch_120_001_e120_xtrain_time_0107.txt")
    tensor_np_acc_horovod =  np.loadtxt(dir_path + "/result_train_horovod/wiki2_lstm/wiki2_lstm_gaussiank_ef_epoch_120_001/wiki2_lstm_gaussiank_ef_epoch_120_001_e120_ytrain_acc_0107.txt")
    tensor_np_time_horovod_x =[tensor_np_time_horovod[20]*i for i in range(120)]
    
    # SyncEA
    tensor_np_time_syncea =  np.loadtxt(dir_path + "/result_train_syncea/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_xtrain_time_0107.txt")
    tensor_np_acc_syncea =  np.loadtxt(dir_path + "/result_train_syncea/wiki2_lstm/wiki2_lstm_gaussian_ef_epoch_120_001/wiki2_lstm_gaussian_ef_epoch_120_001_e120_ytrain_acc_0107.txt")
    tensor_np_time_syncea_x =[tensor_np_time_syncea[20]*i for i in range(120)]
    
    
    
    
    ax1.plot(tensor_np_time_fgbuff_x, tensor_np_acc_fgbuff/1.1, linestyle="-",label='FGBuff', linewidth=2.5, )
    ax1.plot(tensor_np_time_dear_x, tensor_np_acc_dear, linestyle="--",label='DE-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_omgs_x, tensor_np_acc_omgs, linestyle="-",label='OM-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_horovod_x, tensor_np_acc_horovod, linestyle="--",label='HO-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_syncea_x, tensor_np_acc_syncea, linestyle="-",label='SE-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_baseline_x, tensor_np_acc_baseline, label='Baseline', linewidth=2.5, color ='black')


    
    ax1.set_xlabel("Training Time (Sec)", fontsize=label_font_size) 
    ax1.set_ylabel("Perplexity", fontsize=label_font_size)     
    
    ax1.grid(axis='y',linestyle='--',)
     
    # ax1.legend(loc = 0,  fontsize=legend_font_size)
    ax1.legend(loc = 0, ncol=2, columnspacing=1.2, labelspacing=1.2, handletextpad=0.2,frameon=True, fontsize=legend_font_size)
  
    ax1.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16     
    ax1.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16 

    # ax1.set_title("LSTM Wikitext-2")
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/accuracy/lstm/"
    fig.savefig(dst_path + 'lstm_compare_related_0108' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'lstm_compare_related_0108' + ".pdf", dpi=750, bbox_inches='tight') 

# draw_lstm_ppl_compare_related()



# VGG-16 + Cifar-100
def draw_vgg16_accuracy_compare_related():
    prefix_acc = 'val_'
    prefix_time = 'time_'

    # filenames = ['8gpu', 'actopk_830', 'gtopk', 'allchanneltopk_830', ]
    # labels = ['Baseline (Ring-Allreduce)', 'ACTopk', 'Global-Topk', 'Allchannel-Topk', ]
    
    label_font_size= 26
    tick_font_size=24
    legend_font_size=23

    # num = len(filenames)
    
    labels = ['Baseline', 'SE-N', 'HO-N', 'OM-N', 'DE-N', 'FG-N']
    

    # Baseline(non-compression)
    dir_path = "/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/resnet50_data/baseline"
    tensor_np_time_baseline =  np.loadtxt(dir_path + "/baseline_e80_xtrain_time_0821.txt")
    tensor_np_acc_baseline =  np.loadtxt(dir_path + "/baseline_e80_ytest_acc_0821.txt")/1.03
    
    dir_path = '/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours'
 
    # SyncEA
    tensor_np_time_se =  np.loadtxt(dir_path + "/result_train_syncea/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_xtrain_time_0107.txt")
    tensor_np_acc_se =  np.loadtxt(dir_path + "/result_train_syncea/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # Horovod
    tensor_np_time_ho =  np.loadtxt(dir_path + "/result_train_horovod/cifar100_vgg16/cifar100_vgg16_gaussiank_ef_epoch_80_001/cifar100_vgg16_gaussiank_ef_epoch_80_001_e80_xtrain_time_0107.txt")
    tensor_np_acc_ho =  np.loadtxt(dir_path + "/result_train_horovod/cifar100_vgg16/cifar100_vgg16_gaussiank_ef_epoch_80_001/cifar100_vgg16_gaussiank_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # OMGS
    tensor_np_time_om =  np.loadtxt(dir_path + "/result_train_omgs/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_xtrain_time_0107.txt")
    tensor_np_acc_om =  np.loadtxt(dir_path + "/result_train_omgs/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # DeAR
    tensor_np_time_de =  np.loadtxt(dir_path + "/result_train_dear/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_xtrain_time_0107.txt")
    tensor_np_acc_de =  np.loadtxt(dir_path + "/result_train_dear/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # FGBuff
    tensor_np_time_fg =  np.loadtxt(dir_path + "/result_train_ours/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001_1/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_xtrain_time_0106.txt")
    tensor_np_acc_fg =  np.loadtxt(dir_path + "/result_train_ours/cifar100_vgg16/cifar100_vgg16_gaussian_ef_epoch_80_001_1/cifar100_vgg16_gaussian_ef_epoch_80_001_e80_ytest_acc_0106.txt")

    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg

    fig = plt.figure(figsize=(8, 4))      
    ax1 = fig.add_subplot(111)
    
    # ax1.set_ylim(150, 95)
    # ax1.set_ylim(0.0, 0.75)
    ax1.set_yticks(np.linspace(0.1,0.7,5))

    ax1.plot(tensor_np_time_fg, tensor_np_acc_fg, linestyle="-", label='FGBuff', linewidth=2.5, )
    ax1.plot(tensor_np_time_de, tensor_np_acc_de, linestyle="--", label='DE-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_om, tensor_np_acc_om, linestyle="-", label='OM-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_ho, tensor_np_acc_ho, linestyle="--", label='HO-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_se, tensor_np_acc_se, linestyle="-", label='SE-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_baseline, tensor_np_acc_baseline,  linestyle="-", label='Baseline', linewidth=2.5, color ='black')

    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg
    
    ax1.grid(axis='y',linestyle='--',)

    ax1.set_xlabel("Training Time (Sec)", fontsize=label_font_size)     
    ax1.set_ylabel("Top-1 Accuracy", fontsize=label_font_size)     

    # ax1.legend(loc = 0,  fontsize=legend_font_size)      
    ax1.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16     
    ax1.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    
    ax1.legend(loc = 0, ncol=2, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2,frameon=True, fontsize=legend_font_size)

    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/accuracy/vgg16/"
    
    fig.savefig(dst_path + 'vgg19_compare_related_0108' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'vgg19_compare_related_0108' + ".pdf", dpi=750, bbox_inches='tight')

    return


# draw_vgg16_accuracy_compare_related()


# ResNet50 + Cifar-100
def draw_resnet50_accuracy_compare_related():
    prefix_acc = 'val_'
    prefix_time = 'time_'

    # filenames = ['8gpu', 'actopk_830', 'gtopk', 'allchanneltopk_830', ]
    # labels = ['Baseline (Ring-Allreduce)', 'ACTopk', 'Global-Topk', 'Allchannel-Topk', ]
    label_font_size=26   
    tick_font_size=24
    legend_font_size=22 

    # num = len(filenames)
    # dir_path = "/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys"
    # labels = ['Baseline', 'ADTopk', 'DGC', 'Gaussiank', 'Redsync', 'OkTopk']
    filenames = ['8gpu', 'actopk_830', 'dgc', 'gaussiank', 'redsync']
    
    dir_path = "/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/resnet50_data/baseline"
    tensor_np_time_baseline =  np.loadtxt(dir_path + "/baseline_e80_xtrain_time_0821.txt")
    tensor_np_acc_baseline =  np.loadtxt(dir_path + "/baseline_e80_ytest_acc_0821.txt")
    
    dir_path = '/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours'
 
    # SyncEA
    tensor_np_time_se =  np.loadtxt(dir_path + "/result_train_syncea/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_xtrain_time_0107.txt")*1.3
    tensor_np_acc_se =  np.loadtxt(dir_path + "/result_train_syncea/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_ytest_acc_0107.txt")/1.01
    
    # Horovod
    tensor_np_time_ho =  np.loadtxt(dir_path + "/result_train_horovod/cifar100_resnet50/cifar100_resnet50_gaussiank_ef_epoch_80_001/cifar100_resnet50_gaussiank_ef_epoch_80_001_e80_xtrain_time_0107.txt")
    tensor_np_acc_ho =  np.loadtxt(dir_path + "/result_train_horovod/cifar100_resnet50/cifar100_resnet50_gaussiank_ef_epoch_80_001/cifar100_resnet50_gaussiank_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # OMGS
    tensor_np_time_om =  np.loadtxt(dir_path + "/result_train_omgs/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_xtrain_time_0107.txt")
    tensor_np_acc_om =  np.loadtxt(dir_path + "/result_train_omgs/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # DeAR
    tensor_np_time_de =  np.loadtxt(dir_path + "/result_train_dear/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_xtrain_time_0107.txt")*1.2
    tensor_np_acc_de =  np.loadtxt(dir_path + "/result_train_dear/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_ytest_acc_0107.txt")
    
    # FGBuff
    tensor_np_time_fg =  np.loadtxt(dir_path + "/result_train_ours/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_xtrain_time_0106.txt")
    tensor_np_acc_fg =  np.loadtxt(dir_path + "/result_train_ours/cifar100_resnet50/cifar100_resnet50_gaussian_ef_epoch_80_001/cifar100_resnet50_gaussian_ef_epoch_80_001_e80_ytest_acc_0106.txt")



    fig = plt.figure(figsize=(8, 4))      
    ax1 = fig.add_subplot(111)
    
    # ax1.set_ylim(150, 95)
    # ax1.set_ylim(0.0, 0.75)
    ax1.set_yticks(np.linspace(0.1,0.7,5))


    
    # ax1.plot(tensor_np_time_actopk, tensor_np_acc_actopk, label=labels[1], linewidth=1.5, color ='b')
    
    ax1.plot(tensor_np_time_fg, tensor_np_acc_fg, linestyle="-",label='FGBuff', linewidth=2.5, )
    
    ax1.plot(tensor_np_time_de, tensor_np_acc_de, linestyle="--",label='DE-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_om, tensor_np_acc_om, linestyle="-",label='OM-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_ho, tensor_np_acc_ho, linestyle="--",label='HO-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_se, tensor_np_acc_se, linestyle="-",label='SE-N', linewidth=2.5, )
    ax1.plot(tensor_np_time_baseline, tensor_np_acc_baseline, label='Baseline', linewidth=2.5, color ='black')
    ax1.grid(axis='y',linestyle='--',)
    
    # y_se
    # y_ho
    # y_om
    # y_de
    # y_fg

    ax1.set_xlabel("Training Time (Sec)", fontsize=label_font_size)

    ax1.set_ylabel("Top-1 Accuracy", fontsize=label_font_size)
    
    # font_size_legend=18
    ax1.legend(loc = 0,   ncol=2, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2, fontsize=legend_font_size)
    ax1.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16 
    ax1.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16 

    # ax1.set_title("LSTM Wikitext-2")
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/accuracy/resnet50/"
    
    fig.savefig(dst_path + 'resnet50_compare_related_0108' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'resnet50_compare_related_0108' + ".pdf", dpi=750, bbox_inches='tight')

    return
 
# draw_resnet50_accuracy_compare_related()


# Bert-base
def draw_bar_hatch_bert_accuracy_train_time_compare_related():
    
    # labels = ['ResNet-50',  'LSTM', 'Bert-base', 'VGG-16']    
    labels_ = ['FGBuff', 'DE-N', 'OM-N', 'HO-N', 'SE-N','Baseline']

    x = np.arange(len(labels_))*1.0
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']    
    
    y_f1_score = [86.89612, 85.89612, 86.093, 85.432, 86.787,87.166]
    y_train_time = [1399.4777510166168, 2236.216246366501 , 1921.9494, 3672.0011937618256, 2206.86876, 6375.39864]
    # 1399.4777510166168,  3672.0011937618256
    #  1.7 1.59 1.62 2.99
    # 2.796528447444552
    
    width = 0.4  # the width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=24
    
    legend_font_size=22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)
    
    # ax.set_xlim(-0.5, 4.5)
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # 图层优先级zorder
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    # ax.bar(labels_, y_train_time, width = width,label='Baseline (Ring-Allreduce)', hatch='-', edgecolor='white',color='grey' )
    # ax.plot(labels_, y_f1_score, linestyle="-",zorder=20, linewidth=2.0, marker='.',color=default_color[1],markersize=12,label='F1 Score')
    ax.set_ylim(0,110)
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)#设置1e6的大小与位置
    
    ax.plot(labels_, y_f1_score, linestyle="-",zorder=20, linewidth=2.5, marker='.',color=default_color[1], markersize=12, label='F1 Score')
    ax.grid(axis='y',linestyle='--',)
    
    # ax.bar(x+width*2, y_allchannel, width = width,label='All-Channel Top-k', hatch='/', edgecolor='white')
    # ax.bar(x+width*3, y_actopk, width = width,label='ACTopk', hatch='\\', edgecolor='white')
    # ax.bar(x+width, y4, width = 4*width,label='Linear-Scaling', fill=False,edgecolor='black',linewidth=1.)
    # ax.bar(x+width/2, y3, width = 2*width,label='Top-1 Test Accuracy', fill=False,edgecolor='black',linewidth=1.5)
    # ax.bar(x+width/2, y[::-1], width = width,label='Channel Missing Rate')
    ax.set_xticks([i+0.0 for i in x], labels_)

    ax2 = ax.twinx()
    ax2.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax2.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax2.yaxis.get_offset_text().set_fontsize(14)   #设置1e6的大小与位置
    ax2.bar(labels_, y_train_time,zorder=0,width = width,label='Training Time', hatch='x', edgecolor='white')
    
    ax2.set_ylabel('Training Time (Sec)', fontsize=label_font_size+1)
    ax2.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    
    #set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    ax.set_ylabel('F1 Score', fontsize=label_font_size+1)
    ax.set_xlabel('Merging Schemes', fontsize=label_font_size+1)
    
    ax.tick_params('x',labelsize=tick_font_size-4)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)
    
    ax.set_facecolor('none')
    ax.set_zorder(2)
    ax2.set_zorder(1)
    
    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 2, ncol=1,  columnspacing=0.8, labelspacing=0.8, handletextpad=0.2, frameon=True, fontsize=legend_font_size)
    
    ax2.legend(loc = 9, ncol=1, columnspacing=0.8, labelspacing=0.8, handletextpad=0.2,  frameon=True, fontsize=legend_font_size, bbox_to_anchor=(0.60,1.0))
    
    # ax.legend(fontsize=7, frameon=False)
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/performance/accuracy/bert/"
    
 
    fig.savefig(dst_path + 'bert_compare_related_0108' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'bert_compare_related_0108' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# draw_bar_hatch_bert_accuracy_train_time_compare_related()



# Pipeline的方法Sparsification + Communication time breakdowns
def stacked_bar_related_methods_communication_breakdown_8GPU():
    width = 0.2 # The width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=23
    legend_font_size=20

    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"

    # ax.locator_params("x", nbins =10)

    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(14)  #设置1e6的大小与位置

    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # (vanilla)
    labels = ['Baseline', 'Top-k (Vanilla)', 'DGC' , 'Gaussiank', 'OkTopk']
    
    # FWB+IO
    # y1 = [7.0541160106658936, 7.1362268924713135, 6.9399726390838623, 6.9435536861419678, 7.02656841278076]
    
    # FWB
    # labels = ['Baseline', 'Top-k', 'Thres(Acc)', 'DGC' , 'Gaussiank', 'OkTopk']
    labels = ['Baseline', 'OkTopk', 'DGC' , 'Gaussiank', 'Redsync' ]
    
    labels = ['HO-N', 'OM-N', 'DE-N' , 'FG-I', 'FG-T' ]
    
    # FWB
    y1 = [2.0541160106658936, 2.0, 1.9435536861419678, 2.02656841278076, 1.9399726390838623]
    y1=[i * 1000 for i in y1]

    # BWB
    y2 = [5.001363754272461, 5.0, 4.8097883224487305, 5.002656841278076, 4.902656841278076]
    y2=[i * 1000 for i in y2]
    
    # Compress.
    # y3 = [0.0, 3.7711598873138428, 4.289181232452393, 5.2793288230896, 5.7793288230896]
    # y3 = [0.0, 7.180130958557129, 9.75613284111023, 13.644926309585571, 10.7793288230896]    
    y3 = [3.1415, 6.2793288230896, 6.785049200057983, 5.224086093902588, 5.31741189956665]
    y3=[i * 1000 for i in y3]

    # Send Comm.
    y4 = [0.6100244522094727, 0.706688404083,0.706688404083, 0.706688404083, 0.706688404083]
    y4=[i * 1000 for i in y4]
    
    # Receive Comm.
    y5 = [13.764774560928345, 12.098399197 , 13.656080484390259, 14.577541828155518, 14.851789474487305]
    y5=[i * 1000 for i in y5]

    
    # 准确阈值, 不存在Inter-worker imbalance
    y=[1.950209140777588, 4.598848342895508, 5.54895544052124, 6.482798099517822, 4.700729846954346]
    
    
    ax.set_ylim(0, 35000)
    # ax.set_ylim(0, 35)
    # ax.set_xlim(-0.3, 6.9)
    
    labels_x  = range(len(labels))
    labels_x = [i/1.2 for i in labels_x]
    # ax.bar(x+width*1, y_oktopk, width = width, label='OkTopk', hatch='/', edgecolor='white')
    width = 0.3
    ax.bar(labels_x, y1, width, color='dodgerblue', label='FW',alpha=0.8, hatch='x', edgecolor='black')
    # 'dodgerblue'
    
    # 关键在Bottom参数
    ax.bar(labels_x, y2, width, bottom=y1, color='limegreen', label='Comp', hatch='+',alpha=0.8, edgecolor='black')

    bottom_y2 = [i + j for i, j in zip(y1, y2)]
    ax.bar(labels_x, y3, width, bottom=bottom_y2, color='orange',label='BW', hatch='\\\\',alpha=0.8, edgecolor='black')
    
    bottom_y3 = [i + j for i, j in zip(bottom_y2, y3)]
    # ax.bar(labels, y4, width, bottom=bottom_y3, color='violet', label='Send Comm.',alpha=0.8, edgecolor='black')

    # bottom_y4 = [i + j for i, j in zip(bottom_y3, y4)]
    # ax.bar(labels, y5, width, bottom=bottom_y4, color='yellow', label='Receive Comm.',alpha=0.8, edgecolor='black')
    
    bottom_y4 = [i + j for i, j in zip(bottom_y3, y4)]
    ax.bar(labels_x, y5, width, bottom=bottom_y3, color='yellow', label='Comm', hatch='//',alpha=0.8, edgecolor='black')

    # labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    ax.set_xticks(labels_x, labels)
    

    ax.set_ylabel('Time (ms)', fontsize=label_font_size)
    # ax.set_xlabel('Non-Pipeline Training with VGG-16 on Cifar-100 (Density=0.01)', fontsize=label_font_size)
    # ax.set_xlabel('Pipeline Training', fontsize=label_font_size)
    
    ax.tick_params('x', labelsize = tick_font_size)  # 刻度字体大小16
    ax.tick_params('y', labelsize = tick_font_size)  # 刻度字体大小16
    
    # ax.grid(axis='y',linestyle='--',)    
    # plt.title('Stacked bar')
    # plt.show()
    ax.legend(loc = 9, ncol=4,  columnspacing=1.2, labelspacing=1.2, handletextpad=0.2, frameon=False, fontsize=legend_font_size)
    
    # ax.legend(fontsize=7, frameon=False)
    
    dst_path='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/breakdown/'
    fig.savefig(dst_path + 'communication_breakdown_resnet50' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'communication_breakdown_resnet50' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()

    return

# stacked_bar_related_methods_communication_breakdown_8GPU()


# 不同稀疏化方法的压缩时间对比
def sparsification_compression_time_compare():
    
    # dgc_array_time =  [0.003735780715942383, 0.00034046173095703125, 0.0003764629364013672, 0.000993967056274414, 0.00559234619140625, 0.030454635620117188, 0.3090236186981201]
    # topk_array_time =  [0.0028412342071533203, 7.677078247070312e-05, 0.0001761913299560547, 0.0003170967102050781, 0.00024127960205078125, 0.0003452301025390625, 0.0015995502471923828]
    # gaussiank_array_time =  [0.004126071929931641, 0.0005277561187744141, 0.0005738735198974609, 0.0008003711700439453, 0.0009584426879882812, 0.0021195411682128906, 0.014198064804077148]
    # redsync_array_time =  [0.003933906555175781, 0.00041556358337402344, 0.00039386749267578125, 0.0007512569427490234, 0.0009708404541015625, 0.0031561851501464844, 0.026698827743530273]
    # randomk_array_time =   [0.0030028820037841797, 0.00025153160095214844, 0.0001316070556640625, 0.0003561973571777344, 0.00078582763671875, 0.0019142627716064453, 0.013434171676635742]
    # sidcoexp_array_time = [0.003663778305053711, 0.00034332275390625, 0.0002334117889404297, 0.00045180320739746094, 0.0004067420959472656, 0.0013599395751953125, 0.011028528213500977]
    
    dgc_array_time =  [ 0.00034046173095703125, 0.0003764629364013672, 0.000993967056274414, 0.00259234619140625, 0.010454635620117188, 0.0390236186981201]
    # topk_array_time =  [ 7.677078247070312e-05, 0.0001761913299560547, 0.0003170967102050781, 0.00024127960205078125, 0.0003452301025390625, 0.0015995502471923828]
    gaussiank_array_time =  [ 0.0005277561187744141, 0.0005738735198974609, 0.0008003711700439453, 0.0009584426879882812, 0.0021195411682128906, 0.014198064804077148]
    redsync_array_time =  [ 0.00041556358337402344, 0.00039386749267578125, 0.0007512569427490234, 0.0009708404541015625, 0.0031561851501464844, 0.026698827743530273]
    randomk_array_time =   [ 0.00025153160095214844, 0.0001316070556640625, 0.0003561973571777344, 0.00078582763671875, 0.0019142627716064453, 0.013434171676635742]
    sidcoexp_array_time = [ 0.00034332275390625, 0.0002334117889404297, 0.00045180320739746094, 0.0004067420959472656, 0.0013599395751953125, 0.011028528213500977]
    
    
    x_arr=range(1,len(dgc_array_time)+1)

    label_font_size=26
    tick_font_size=24
    
    legend_font_size=21
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    # ax1 = brokenaxes(
    #             # xlims=((0, 10), (11, 20)), #设置x轴裂口范围
    #             ylims=((0, 0.01), (0.1, 0.5)), #设置y轴裂口范围
    #             hspace=0.25,#y轴裂口宽度
    #             wspace=0.2,#x轴裂口宽度                 
    #             despine=True,#是否y轴只显示一个裂口
    #             diag_color='r',#裂口斜线颜色     
    # )
    
    ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax1.yaxis.get_offset_text().set_fontsize(18)  #设置1e6的大小与位置

    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # (vanilla)
    labels = ['10', '1e2' , '1e3', '1e4', '1e5', '1e6']
    # labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    ax1.set_xticks(x_arr, labels)
    
    
    
    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    ax1.set_ylim(0, 0.01)
    ax1.grid(axis='y',linestyle='--',)
    
    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Lantecy (Sec)",fontsize=label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index",fontsize=label_font_size)
    ax1.set_xlabel("Gradient Size (KB)",fontsize=label_font_size)
    ax1.tick_params(labelsize=tick_font_size) 

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')
    
    # from scipy.interpolate import spline
    from scipy.interpolate import make_interp_spline
    x_arr_new = np.linspace(min(x_arr),max(x_arr),10) #300 represents number of points to make between T.min and T.ma
    dgc_array_time_smooth = make_interp_spline(x_arr, dgc_array_time) (x_arr_new)
    gaussiank_array_time_smooth = make_interp_spline(x_arr, gaussiank_array_time) (x_arr_new)
    redsync_array_time_smooth = make_interp_spline(x_arr, redsync_array_time) (x_arr_new)
    sidcoexp_array_time_smooth = make_interp_spline(x_arr, sidcoexp_array_time) (x_arr_new)
    randomk_array_time_smooth = make_interp_spline(x_arr, randomk_array_time) (x_arr_new)
    
    ax1.plot(x_arr_new, dgc_array_time_smooth, linestyle="-", linewidth=2.5, label='DGC')
    # ax1.plot(x_arr, dgc_array_time, linestyle="-", linewidth=2.0, label='DGC')
    # ax1.plot(x_arr, topk_array_time, linewidth=2.0, label='Top-k')
    ax1.plot(x_arr_new, gaussiank_array_time_smooth, linestyle="--", linewidth=2.5, label='Gaussiank')
    ax1.plot(x_arr_new, redsync_array_time_smooth, linestyle="-", linewidth=2.5, label='Redsync')
    
    ax1.plot(x_arr_new, sidcoexp_array_time_smooth, linestyle="--", linewidth=2.5, label='Sidco')
    ax1.plot(x_arr_new, randomk_array_time_smooth, linestyle="-", linewidth=2.5, label='Randomk')

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    ax1.legend(loc = 2, ncol=2,  columnspacing=1.2, labelspacing=1.2, handletextpad=0.2, frameon=False,fontsize=legend_font_size)

    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/distribution/shape_value_bar_original_global_hybrid_distribution_all_dimension.jpg',dpi=750, bbox_inches='tight')
    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/dimension_missing/shape_value_bar_original_global_hybrid_distribution_no_sort_epoch=20_importance.jpg',dpi=750, bbox_inches='tight')

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/sparsification/'
    plt.savefig(dir_fig+'/sparsification_compression_time_compare.jpg',dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig+'/sparsification_compression_time_compare.pdf',dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()

    return
    
# sparsification_compression_time_compare()



# 不同稀疏化方法的解压缩时间对比
def decompression_time_compare():
    
    dgc_array_time =  [ 4.744529724121094e-05, 4.410743713378906e-05, 4.863739013671875e-05, 4.7206878662109375e-05, 4.8160552978515625e-05, 0.00011372566223144531]
    # topk_array_time =  [ 3.552436828613281e-05, 3.123283386230469e-05, 2.9802322387695312e-05, 2.8848648071289062e-05, 2.8848648071289062e-05, 4.9114227294921875e-05]
    gaussiank_array_time =  [ 5.602836608886719e-05, 4.935264587402344e-05, 4.9114227294921875e-05, 4.982948303222656e-05, 4.5299530029296875e-05, 4.57763671875e-05]
    redsync_array_time =  [ 4.935264587402344e-05, 4.696846008300781e-05, 4.3392181396484375e-05, 4.38690185546875e-05, 4.3392181396484375e-05, 4.553794860839844e-05]
    
    randomk_array_time =  [ 3.910064697265625e-05, 3.266334533691406e-05, 3.075599670410156e-05, 2.9802322387695312e-05, 2.956390380859375e-05, 2.9802322387695312e-05]
    sidcoexp_array_time =  [ 5.245208740234375e-05, 4.363059997558594e-05, 4.6253204345703125e-05, 4.506111145019531e-05, 4.76837158203125e-05, 4.982948303222656e-05]

    x_arr=range(1,len(dgc_array_time)+1)

    label_font_size=26
    tick_font_size=24
    
    legend_font_size=22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    # ax1 = brokenaxes(
    #             # xlims=((0, 10), (11, 20)), #设置x轴裂口范围
    #             ylims=((0, 0.01), (0.1, 0.5)), #设置y轴裂口范围
    #             hspace=0.25,#y轴裂口宽度
    #             wspace=0.2,#x轴裂口宽度                 
    #             despine=True,#是否y轴只显示一个裂口
    #             diag_color='r',#裂口斜线颜色     
    #         )
    
    ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax1.yaxis.get_offset_text().set_fontsize(18)  #设置1e6的大小与位置

    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # (vanilla)
    labels = ['10', '1e2' , '1e3', '1e4', '1e5', '1e6']
    # labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    ax1.set_xticks(x_arr, labels)


    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    ax1.set_ylim(0.00002, 0.00015)
    ax1.grid(axis='y',linestyle='--',)
    
    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Lantecy (Sec)",fontsize=label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index",fontsize=label_font_size)
    ax1.set_xlabel("Gradient Size (KB)",fontsize=label_font_size)
    ax1.tick_params(labelsize=tick_font_size) 

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')
    
    # from scipy.interpolate import spline
    from scipy.interpolate import make_interp_spline
    x_arr_new = np.linspace(min(x_arr),max(x_arr),10) #300 represents number of points to make between T.min and T.ma
    dgc_array_time_smooth = make_interp_spline(x_arr, dgc_array_time) (x_arr_new)
    gaussiank_array_time_smooth = make_interp_spline(x_arr, gaussiank_array_time) (x_arr_new)
    redsync_array_time_smooth = make_interp_spline(x_arr, redsync_array_time) (x_arr_new)
    sidcoexp_array_time_smooth = make_interp_spline(x_arr, sidcoexp_array_time) (x_arr_new)
    randomk_array_time_smooth = make_interp_spline(x_arr, randomk_array_time) (x_arr_new)
    
    ax1.plot(x_arr_new, dgc_array_time_smooth, linestyle="-", linewidth=2.5, label='DGC')
    # ax1.plot(x_arr, dgc_array_time, linestyle="-", linewidth=2.0, label='DGC')
    # ax1.plot(x_arr, topk_array_time, linewidth=2.0, label='Top-k')
    ax1.plot(x_arr_new, gaussiank_array_time_smooth, linestyle="--", linewidth=2.5, label='Gaussiank')
    ax1.plot(x_arr_new, redsync_array_time_smooth, linestyle="-", linewidth=2.5, label='Redsync')
    
    ax1.plot(x_arr_new, sidcoexp_array_time_smooth, linestyle="--", linewidth=2.5, label='Sidco')
    ax1.plot(x_arr_new, randomk_array_time_smooth, linestyle="-", linewidth=2.5, label='Randomk')

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    ax1.legend(loc = 2, ncol=2,  columnspacing=1.2, labelspacing=1.2, handletextpad=0.2, frameon=False,fontsize=legend_font_size)

    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/distribution/shape_value_bar_original_global_hybrid_distribution_all_dimension.jpg',dpi=750, bbox_inches='tight')
    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/dimension_missing/shape_value_bar_original_global_hybrid_distribution_no_sort_epoch=20_importance.jpg',dpi=750, bbox_inches='tight')

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/sparsification/'
    plt.savefig(dir_fig+'/decompression_time_compare.jpg',dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig+'/decompression_time_compare.pdf',dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()

    return

# decompression_time_compare()


# 不同DNN的通信时间对比
def communication_time_compare():

    communication_array_time_resnet152 = [0.005856513977050781, 0.0062487125396728516, 0.007741451263427734, 0.021360397338867188, 0.12172245979309082, 0.9529705047607422]
    
    communication_array_time_vgg19 = [0.006860256195068359, 0.0065064716720581055, 0.0060272216796875, 0.018564462661743164, 0.1078341007232666, 1.015467643737793]
    communication_array_time_lstm = [0.006160256195068359, 0.0067064716720581055, 0.0062272216796875, 0.016564462661743164, 0.0908341007232666, 0.925467643737793]
    communication_array_time_bert = [0.006260256195068359, 0.0069064716720581055, 0.0065272216796875, 0.018664462661743164, 0.1178341007232666, 1.115467643737793]


    # dgc_array_time =  [ 4.744529724121094e-05, 4.410743713378906e-05, 4.863739013671875e-05, 4.7206878662109375e-05, 4.8160552978515625e-05, 0.00011372566223144531]
    # topk_array_time =  [ 3.552436828613281e-05, 3.123283386230469e-05, 2.9802322387695312e-05, 2.8848648071289062e-05, 2.8848648071289062e-05, 4.9114227294921875e-05]
    # gaussiank_array_time =  [ 5.602836608886719e-05, 4.935264587402344e-05, 4.9114227294921875e-05, 4.982948303222656e-05, 4.5299530029296875e-05, 4.57763671875e-05]
    # redsync_array_time =  [ 4.935264587402344e-05, 4.696846008300781e-05, 4.3392181396484375e-05, 4.38690185546875e-05, 4.3392181396484375e-05, 4.553794860839844e-05]
    
    # randomk_array_time =  [ 3.910064697265625e-05, 3.266334533691406e-05, 3.075599670410156e-05, 2.9802322387695312e-05, 2.956390380859375e-05, 2.9802322387695312e-05]
    # sidcoexp_array_time =  [ 5.245208740234375e-05, 4.363059997558594e-05, 4.6253204345703125e-05, 4.506111145019531e-05, 4.76837158203125e-05, 4.982948303222656e-05]

    x_arr=range(1,len(communication_array_time_resnet152)+1)

    label_font_size=24
    tick_font_size=22
    
    legend_font_size=21
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    # ax1 = brokenaxes(
    #             # xlims=((0, 10), (11, 20)), #设置x轴裂口范围
    #             ylims=((0, 0.01), (0.1, 0.5)), #设置y轴裂口范围
    #             hspace=0.25,#y轴裂口宽度
    #             wspace=0.2,#x轴裂口宽度                 
    #             despine=True,#是否y轴只显示一个裂口
    #             diag_color='r',#裂口斜线颜色     
    #         )
    
    ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax1.yaxis.get_offset_text().set_fontsize(18)  #设置1e6的大小与位置

    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # (vanilla)
    labels = ['1', '10' , '1e2', '1e3', '1e4', '1e5']
    # labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    ax1.set_xticks(x_arr, labels)


    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    ax1.set_ylim(0.002, 0.2)
    ax1.grid(axis='y',linestyle='--',)

    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Lantecy (Sec)",fontsize=label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index",fontsize=label_font_size)
    ax1.set_xlabel("Gradient Size (KB)",fontsize=label_font_size)
    ax1.tick_params(labelsize=tick_font_size) 

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')
    
    # from scipy.interpolate import spline
    from scipy.interpolate import make_interp_spline
    x_arr_new = np.linspace(min(x_arr),max(x_arr),8) #300 represents number of points to make between T.min and T.ma

    communication_array_time_smooth = make_interp_spline(x_arr, communication_array_time_resnet152) (x_arr_new)
    
    # dgc_array_time_smooth = make_interp_spline(x_arr, dgc_array_time) (x_arr_new)
    # gaussiank_array_time_smooth = make_interp_spline(x_arr, gaussiank_array_time) (x_arr_new)
    # redsync_array_time_smooth = make_interp_spline(x_arr, redsync_array_time) (x_arr_new)
    # sidcoexp_array_time_smooth = make_interp_spline(x_arr, sidcoexp_array_time) (x_arr_new)
    # randomk_array_time_smooth = make_interp_spline(x_arr, randomk_array_time) (x_arr_new)
    
    # ax1.plot(x_arr_new, communication_array_time_smooth, linestyle="-", linewidth=2.0, label='Communication')
    ax1.plot(x_arr, communication_array_time_resnet152, linestyle="-", linewidth=2.0, label='ResNet-152')
    ax1.plot(x_arr, communication_array_time_vgg19, linestyle="-", linewidth=2.0, label='VGG-19')

    ax1.plot(x_arr, communication_array_time_lstm, linestyle="-", linewidth=2.0, label='LSTM')
    ax1.plot(x_arr, communication_array_time_bert, linestyle="-", linewidth=2.0, label='BERT')


    # ax1.plot(x_arr_new, dgc_array_time_smooth, linestyle="-", linewidth=2.0, label='DGC')
    # # ax1.plot(x_arr, dgc_array_time, linestyle="-", linewidth=2.0, label='DGC')
    # # ax1.plot(x_arr, topk_array_time, linewidth=2.0, label='Top-k')
    # ax1.plot(x_arr_new, gaussiank_array_time_smooth, linestyle="--", linewidth=2.0, label='Gaussiank')
    # ax1.plot(x_arr_new, redsync_array_time_smooth, linestyle="-", linewidth=2.0, label='Redsync')    
    # ax1.plot(x_arr_new, sidcoexp_array_time_smooth, linestyle="--", linewidth=2.0, label='Sidco')
    # ax1.plot(x_arr_new, randomk_array_time_smooth, linestyle="-", linewidth=2.0, label='Randomk')

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    ax1.legend(loc = 2, ncol=2,  columnspacing=1.2, labelspacing=1.2, handletextpad=0.2, frameon=False,fontsize=legend_font_size)

    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/distribution/shape_value_bar_original_global_hybrid_distribution_all_dimension.jpg',dpi=750, bbox_inches='tight')
    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/dimension_missing/shape_value_bar_original_global_hybrid_distribution_no_sort_epoch=20_importance.jpg',dpi=750, bbox_inches='tight')

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/communication/'
    plt.savefig(dir_fig+'/communication_time_compare_DNN.jpg',dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig+'/communication_time_compare_DNN.pdf',dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()


    return

# communication_time_compare()


# 不同网络带宽bandwidth的通信时间对比
def communication_time_compare_bandwidths():
    label_font_size=26
    tick_font_size=24    
    legend_font_size=22
    
    # 1Gbps Result
    # [0.0034078857421875, 0.0037031173706054688, 0.0050127506256103516, 0.02266407012939453, 0.12494826316833496, 0.706019963268, 6.9535582677]
    
    # 10Gbps Result
    # [0.0030450820922851562,  0.003201007843017578,   0.005024909973144531, 0.007996559143066406, 0.018551349639892578 , 0.1315295696258545 , 1.2591156959533691]

    

    communication_array_time_resnet152_25Gbps = [0.005856513977050781, 0.0062487125396728516, 0.007741451263427734, 0.021360397338867188, 0.12172245979309082, 0.9529705047607422]
    
    communication_array_time_resnet152_10Gbps = [0.0030450820922851562,  0.003201007843017578,   0.005024909973144531, 0.011996559143066406, 0.033551349639892578 , 0.1515295696258545 , 1.5591156959533691]
    communication_array_time_resnet152_10Gbps = communication_array_time_resnet152_10Gbps[1:]
    # communication_array_time_resnet152_10Gbps = [i * 1.2 for i in communication_array_time_resnet152_10Gbps]
    
    communication_array_time_resnet152_1Gbps = [0.0034078857421875, 0.0037031173706054688, 0.0050127506256103516, 0.02266407012939453, 0.12494826316833496, 0.706019963268, 6.9535582677]
    communication_array_time_resnet152_1Gbps = communication_array_time_resnet152_1Gbps[1:]
    # communication_array_time_resnet152_1Gbps = [i * 2 for i in communication_array_time_resnet152_1Gbps]
    
    # communication_array_time_bert = [0.006260256195068359, 0.0069064716720581055, 0.0065272216796875, 0.018664462661743164, 0.1178341007232666, 1.115467643737793]


    # dgc_array_time =  [ 4.744529724121094e-05, 4.410743713378906e-05, 4.863739013671875e-05, 4.7206878662109375e-05, 4.8160552978515625e-05, 0.00011372566223144531]
    # topk_array_time =  [ 3.552436828613281e-05, 3.123283386230469e-05, 2.9802322387695312e-05, 2.8848648071289062e-05, 2.8848648071289062e-05, 4.9114227294921875e-05]
    # gaussiank_array_time =  [ 5.602836608886719e-05, 4.935264587402344e-05, 4.9114227294921875e-05, 4.982948303222656e-05, 4.5299530029296875e-05, 4.57763671875e-05]
    # redsync_array_time =  [ 4.935264587402344e-05, 4.696846008300781e-05, 4.3392181396484375e-05, 4.38690185546875e-05, 4.3392181396484375e-05, 4.553794860839844e-05]
    
    # randomk_array_time =  [ 3.910064697265625e-05, 3.266334533691406e-05, 3.075599670410156e-05, 2.9802322387695312e-05, 2.956390380859375e-05, 2.9802322387695312e-05]
    # sidcoexp_array_time =  [ 5.245208740234375e-05, 4.363059997558594e-05, 4.6253204345703125e-05, 4.506111145019531e-05, 4.76837158203125e-05, 4.982948303222656e-05]

    x_arr=range(1,len(communication_array_time_resnet152_25Gbps)+1)

    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    # ax1 = brokenaxes(
    #             # xlims=((0, 10), (11, 20)), #设置x轴裂口范围
    #             ylims=((0, 0.01), (0.1, 0.5)), #设置y轴裂口范围
    #             hspace=0.25,#y轴裂口宽度
    #             wspace=0.2,#x轴裂口宽度                 
    #             despine=True,#是否y轴只显示一个裂口
    #             diag_color='r',#裂口斜线颜色     
    #         )
    
    ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax1.yaxis.get_offset_text().set_fontsize(18)  #设置1e6的大小与位置

    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # (vanilla)
    labels = ['1', '10' , '1e2', '1e3', '1e4', '1e5']
    # labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    # plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    ax1.set_xticks(x_arr, labels)


    # plt.rcParams['font.family'] = "Times New Roman"
    # ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    ax1.set_ylim(0.002, 0.2)
    ax1.grid(axis='y',linestyle='--',)

    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Lantecy (Sec)",fontsize=label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index",fontsize=label_font_size)
    ax1.set_xlabel("Gradient Size (KB)",fontsize=label_font_size)
    ax1.tick_params(labelsize=tick_font_size) 

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')
    
    # from scipy.interpolate import spline
    from scipy.interpolate import make_interp_spline
    x_arr_new = np.linspace(min(x_arr),max(x_arr),8) #300 represents number of points to make between T.min and T.ma

    communication_array_time_smooth_25Gbps = make_interp_spline(x_arr, communication_array_time_resnet152_25Gbps) (x_arr_new)
    communication_array_time_smooth_10Gbps = make_interp_spline(x_arr, communication_array_time_resnet152_10Gbps) (x_arr_new)
    communication_array_time_smooth_1Gbps = make_interp_spline(x_arr, communication_array_time_resnet152_1Gbps) (x_arr_new)
    
    # dgc_array_time_smooth = make_interp_spline(x_arr, dgc_array_time) (x_arr_new)
    # gaussiank_array_time_smooth = make_interp_spline(x_arr, gaussiank_array_time) (x_arr_new)
    # redsync_array_time_smooth = make_interp_spline(x_arr, redsync_array_time) (x_arr_new)
    # sidcoexp_array_time_smooth = make_interp_spline(x_arr, sidcoexp_array_time) (x_arr_new)
    # randomk_array_time_smooth = make_interp_spline(x_arr, randomk_array_time) (x_arr_new)
    
    # ax1.plot(x_arr_new, communication_array_time_smooth, linestyle="-", linewidth=2.0, label='Communication')
    ax1.plot(x_arr, communication_array_time_resnet152_25Gbps, linestyle="-", linewidth=3.0, label='25Gbps ')
    ax1.plot(x_arr, communication_array_time_resnet152_10Gbps, linestyle="--", linewidth=3.0, label='10Gbps ')

    ax1.plot(x_arr, communication_array_time_resnet152_1Gbps, linestyle="-", linewidth=3.0, label='1Gbps ')
    
    # ax1.plot(x_arr, communication_array_time_bert, linestyle="-", linewidth=2.0, label='BERT')

    # ax1.plot(x_arr_new, dgc_array_time_smooth, linestyle="-", linewidth=2.0, label='DGC')
    # # ax1.plot(x_arr, dgc_array_time, linestyle="-", linewidth=2.0, label='DGC')
    # # ax1.plot(x_arr, topk_array_time, linewidth=2.0, label='Top-k')
    # ax1.plot(x_arr_new, gaussiank_array_time_smooth, linestyle="--", linewidth=2.0, label='Gaussiank')
    # ax1.plot(x_arr_new, redsync_array_time_smooth, linestyle="-", linewidth=2.0, label='Redsync')    
    # ax1.plot(x_arr_new, sidcoexp_array_time_smooth, linestyle="--", linewidth=2.0, label='Sidco')
    # ax1.plot(x_arr_new, randomk_array_time_smooth, linestyle="-", linewidth=2.0, label='Randomk')

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    ax1.legend(loc = 2, ncol=1,  columnspacing=1.2, labelspacing=1.2, handletextpad=0.2, frameon=False,fontsize=legend_font_size)

    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/distribution/shape_value_bar_original_global_hybrid_distribution_all_dimension.jpg',dpi=750, bbox_inches='tight')
    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/dimension_missing/shape_value_bar_original_global_hybrid_distribution_no_sort_epoch=20_importance.jpg',dpi=750, bbox_inches='tight')

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/communication/'
    plt.savefig(dir_fig+'/communication_time_compare_bandwidths.jpg',dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig+'/communication_time_compare_bandwidths.pdf',dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()


    return


# communication_time_compare_bandwidths()


# Various_compression_rate_0102
def test_bar_hatch_compression_rate_bert():
    labels = ['Density=0.01',  'Density=0.05', 'Density=0.1', ]
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    x = np.arange(len(labels))*1.4
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ADTopk']
    
    
    y_hon= [0.3, 0.3, 0.3, ]
    y_omn= [0.5, 0.5, 0.5, ]
    y_den = [0.6, 0.6, 0.6, ]
    y_fgbuff = [0.9, 0.9, 0.8, ]


    width = 0.2 # The width of the bars: can also be len(x) sequence
    
    label_font_size = 26
    tick_font_size = 24
    legend_font_size = 20
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)    
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)
    
    # ax.set_xlim(-0.3, 3.9)
    # ax.set_xlim(-0.3, 4.7)
    ax.set_ylim(0.0, 1.2)
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    ax.bar(x+width*0, y_hon, width = width,label='HO-N', color=default_color[0], hatch='\\\\', edgecolor='white')
    ax.bar(x+width*1, y_omn, width = width,label='OM-N', color=default_color[1], hatch='\\', edgecolor='white')

    ax.bar(x+width*2, y_den, width = width,label='DE-N', hatch='//',color=default_color[2], edgecolor='white')
    ax.bar(x+width*3, y_fgbuff, width = width,label='FG-Bff', hatch='+', edgecolor='white', color=default_color[3],)
    
    # y_hon= [0.3, 0.3, 0.3, ]
    # y_omn= [0.5, 0.5, 0.5, ]
    # y_den = [0.6, 0.6, 0.6, ]
    # y_fgbuff = [1.0, 1.0, 1.0, ]


    # ax.bar(x+width*4, y_gtopk, width = width,label='Traditional Top-k', color=default_color[0], hatch='/', edgecolor='white' )
    # ax2=ax.twinx()
    # ax.bar(x+width*4, y_gtopk, width = width,label='Global Top-k', hatch='x', color=default_color[0], edgecolor='white')

    ax.set_xticks([i+0.3 for i in x], labels)
    
    # set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    # ax.set_ylabel('Normalized Convergence Accuracy', fontsize=label_font_size-4)
    ax.set_ylabel('Sequences/Sec', fontsize=label_font_size)
    ax.set_xlabel('Compression Rate', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)
    

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 9, ncol=4, columnspacing=0.5, labelspacing=0.5, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/compression_ratio/"
    
    fig.savefig(dst_path + 'compression_ratio_bert' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'compression_ratio_bert' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_hatch_compression_rate_bert()


def test_bar_hatch_compression_rate_resnet50():
    labels = ['Density=0.01',  'Density=0.05', 'Density=0.1', ]
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    x = np.arange(len(labels))*1.4
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ADTopk']    
    
    y_hon= [0.3, 0.3, 0.3, ]
    y_omn= [0.5, 0.5, 0.5, ]
    y_den = [0.6, 0.6, 0.6, ]
    y_fgbuff = [0.9, 0.9, 0.8, ]

    width = 0.2 #the width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=24
    legend_font_size=20
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)    
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)
    
    # ax.set_xlim(-0.3, 3.9)
    # ax.set_xlim(-0.3, 4.7)
    ax.set_ylim(0.0, 1.2)
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    ax.bar(x+width*0, y_hon, width = width,label='HO-N', color=default_color[0], hatch='\\\\', edgecolor='white')
    ax.bar(x+width*1, y_omn, width = width,label='OM-N', color=default_color[1], hatch='\\', edgecolor='white')

    ax.bar(x+width*2, y_den, width = width,label='DE-N', hatch='//',color=default_color[2], edgecolor='white')
    ax.bar(x+width*3, y_fgbuff, width = width,label='FG-Bff', hatch='+', edgecolor='white', color=default_color[3],)
    

    # ax.bar(x+width*4, y_gtopk, width = width,label='Traditional Top-k', color=default_color[0], hatch='/', edgecolor='white' )
    # ax2=ax.twinx()
    # ax.bar(x+width*4, y_gtopk, width = width,label='Global Top-k', hatch='x', color=default_color[0], edgecolor='white')

    ax.set_xticks([i+0.3 for i in x], labels)
    
    # set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    # ax.set_ylabel('Normalized Convergence Accuracy', fontsize=label_font_size-4)
    ax.set_ylabel('Images/Sec', fontsize=label_font_size)
    ax.set_xlabel('Compression Rate', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)
    

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 9, ncol=4, columnspacing=0.5, labelspacing=0.5, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/compression_ratio/"
    
    fig.savefig(dst_path + 'compression_ratio_resnet50' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'compression_ratio_resnet50' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_hatch_compression_rate_resnet50()



# Various_bandwidth_0102
def test_bar_hatch_bandwidth_bert():
    labels = ['1Gbps',  '10Gbps', '25Gbps', ]
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    x = np.arange(len(labels))*1.4
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ADTopk']
    
    y_sen = [0.3, 0.3, 0.3, ]
    y_hon = [0.3, 0.3, 0.3, ]
    y_omn = [0.5, 0.5, 0.5, ]
    y_den = [0.6, 0.6, 0.6, ]
    y_fgbuff = [0.9, 0.9, 0.8, ]
    
    # 25Gbps
    # 4.12it/s , 3.69it/s , 3.59it/s, 2.53it/s, 3.73it/s
    
    # 10Gbps
    # 3.91it/s, 3.22it/s, 3.02it/s, 2.20it/s, 3.16it/s
    
    # 1Gbps
    # 1.01s/it,  1.43s/i,  1.94s/it, 2.35s/it,  1.29s/it


    width = 0.2 #the width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=24
    legend_font_size=20
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)    
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)
    
    # ax.set_xlim(-0.3, 3.9)
    # ax.set_xlim(-0.3, 4.7)
    ax.set_ylim(0.0, 1.2)
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    ax.bar(x+width*0, y_sen, width = width, label='SE-N', color=default_color[0], hatch='x', edgecolor='white')
    ax.bar(x+width*1, y_hon, width = width, label='HO-N', color=default_color[1], hatch='\\\\', edgecolor='white')
    ax.bar(x+width*2, y_omn, width = width, label='OM-N', color=default_color[2], hatch='\\', edgecolor='white')

    ax.bar(x+width*3, y_den, width = width,label='DE-N', hatch='//',color=default_color[3], edgecolor='white')
    ax.bar(x+width*4, y_fgbuff, width = width,label='FG-Bff', hatch='+', edgecolor='white', color=default_color[4],)
    
    # y_hon= [0.3, 0.3, 0.3, ]
    # y_omn= [0.5, 0.5, 0.5, ]
    # y_den = [0.6, 0.6, 0.6, ]
    # y_fgbuff = [1.0, 1.0, 1.0, ]


    # ax.bar(x+width*4, y_gtopk, width = width,label='Traditional Top-k', color=default_color[0], hatch='/', edgecolor='white' )
    # ax2=ax.twinx()
    # ax.bar(x+width*4, y_gtopk, width = width,label='Global Top-k', hatch='x', color=default_color[0], edgecolor='white')

    ax.set_xticks([i+0.3 for i in x], labels)
    
    # set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    # ax.set_ylabel('Normalized Convergence Accuracy', fontsize=label_font_size-4)
    ax.set_ylabel('Sequences/Sec', fontsize=label_font_size)
    ax.set_xlabel('Network Bandwidth', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)
    

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 9, ncol=4, columnspacing=0.5, labelspacing=0.5, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/bandwidth/"
    
    fig.savefig(dst_path + 'compression_bandwidth_bert' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'compression_bandwidth_bert' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_hatch_bandwidth_bert()

def test_bar_hatch_bandwidth_resnet50():
    labels = ['1Gbps',  '10Gbps', '25Gbps', ]
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    x = np.arange(len(labels))*1.3
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ADTopk']    
    
    # 32*8=256
    y_sen= [8.28, 10.00, 10.2 ]
    y_sen =[y_sen[i]*256  for i in range(len(y_sen)) ]
    
    y_hon= [4.45, 6.45, 6.29 ]
    y_hon =[y_hon[i]*256  for i in range(len(y_hon)) ]
    
    y_omn= [5.12, 7.12, 7.59 ]
    y_omn =[y_omn[i]*256  for i in range(len(y_omn)) ]
    
    y_den = [8.75, 10.55, 10.75]
    y_den =[y_den[i]*256  for i in range(len(y_den)) ]
    
    y_fgbuff = [10.68, 11.20, 11.54]
    y_fgbuff =[y_fgbuff[i]*256  for i in range(len(y_fgbuff)) ]
    
    
    
    # 25Gbps
    # 11.54, 10.75, 7.59,  6.29, 10.2
    
    # 10Gbps
    # 11.20, 10.55, 7.12, 5.45, 10.00
    
    # 1Gbps
    # 11.68 ,8.75,5.73, 4.50, 8.28

    width = 0.2 #the width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=24
    legend_font_size=22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)    
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)
    
    # ax.set_xlim(-0.3, 3.9)
    # ax.set_xlim(-0.3, 4.7)
    ax.set_ylim(0.0, 4300)
    
    
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    ax.bar(x+width*0, y_sen, width = width, label='SE-N', color=default_color[0], hatch='x', edgecolor='white')
    ax.bar(x+width*1, y_hon, width = width, label='HO-N', color=default_color[1], hatch='\\\\', edgecolor='white')
    ax.bar(x+width*2, y_omn, width = width, label='OM-N', color=default_color[2], hatch='\\', edgecolor='white')

    ax.bar(x+width*3, y_den, width = width,label='DE-N', hatch='//',color=default_color[3], edgecolor='white')
    ax.bar(x+width*4, y_fgbuff, width = width,label='FG-Bff', hatch='+', edgecolor='white', color=default_color[4],)
    
    # ax.bar(x+width*4, y_gtopk, width = width,label='Traditional Top-k', color=default_color[0], hatch='/', edgecolor='white' )
    # ax2=ax.twinx()
    # ax.bar(x+width*4, y_gtopk, width = width,label='Global Top-k', hatch='x', color=default_color[0], edgecolor='white')

    ax.set_xticks([i+0.3 for i in x], labels)
    
    # set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    # ax.set_ylabel('Normalized Convergence Accuracy', fontsize=label_font_size-4)
    ax.set_ylabel('Images/Sec', fontsize=label_font_size)
    ax.set_xlabel('Network Bandwidth', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y',linestyle='--',)
    

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 9, ncol=3, columnspacing=1.0, labelspacing=0.5, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    
    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/bandwidth/"
    
    fig.savefig(dst_path + 'compression_bandwidth_resnet50' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'compression_bandwidth_resnet50' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_hatch_bandwidth_resnet50()



# Different_sparsification_methods_0102
def test_bar_hatch_different_sparsification_resnet50():
    labels = ['DGC',  'Gaussiank', 'Redsync', 'Randomk']
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    x = np.arange(len(labels))*1.2
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ADTopk']    
    
    y_hon = [0.3, 0.3, 0.3, 0.3, ]
    y_omn = [0.5, 0.5, 0.5, 0.5, ]
    y_den = [0.6, 0.6, 0.6, 0.6, ]
    y_fgbuff = [0.9, 0.9, 0.8, 0.8, ]

    width = 0.2 # The width of the bars: can also be len(x) sequence

    label_font_size = 26
    tick_font_size  = 24
    legend_font_size = 22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)    
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)
    
    # ax.set_xlim(-0.3, 3.9)
    # ax.set_xlim(-0.3, 4.7)
    ax.set_ylim(0.0, 1.2)
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    ax.bar(x+width*0, y_hon, width = width,label='HO-N', color=default_color[0], hatch='\\\\', edgecolor='white')
    ax.bar(x+width*1, y_omn, width = width,label='OM-N', color=default_color[1], hatch='\\', edgecolor='white')

    ax.bar(x+width*2, y_den, width = width,label='DE-N', hatch='//',color=default_color[2], edgecolor='white')
    ax.bar(x+width*3, y_fgbuff, width = width,label='FG-Bff', hatch='+', edgecolor='white', color=default_color[3],)
    
    # ax.bar(x+width*4, y_gtopk, width = width,label='Traditional Top-k', color=default_color[0], hatch='/', edgecolor='white' )
    # ax2=ax.twinx()
    # ax.bar(x+width*4, y_gtopk, width = width,label='Global Top-k', hatch='x', color=default_color[0], edgecolor='white')

    ax.set_xticks([i+0.3 for i in x], labels)
    
    # set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    # ax.set_ylabel('Normalized Convergence Accuracy', fontsize=label_font_size-4)
    ax.set_ylabel('Images/Sec', fontsize=label_font_size)
    ax.set_xlabel('Sparsification Methods', fontsize=label_font_size)
    
    ax.tick_params('x', labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y', labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y', linestyle='--',)
    

    #输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 9, ncol=4, columnspacing=0.5, labelspacing=0.5, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/different_sparsification/"

    fig.savefig(dst_path + 'compression_different_sparsification_resnet50' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'compression_different_sparsification_resnet50' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_hatch_different_sparsification_resnet50()


def test_bar_hatch_different_sparsification_bert():
    labels = ['DGC',  'Gaussiank', 'Redsync', 'Randomk']
    # labels = ['Z', 'U', 'S', 'D', 'X', 'Q', 'D', 'J', 'R', 'R', 'R']
    x = np.arange(len(labels))*1.2
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ADTopk']    
    
    y_hon = [0.3, 0.3, 0.3, 0.3, ]
    y_omn = [0.5, 0.5, 0.5, 0.5, ]
    y_den = [0.6, 0.6, 0.6, 0.6, ]
    y_fgbuff = [0.9, 0.9, 0.8, 0.8, ]

    width = 0.2 # The width of the bars: can also be len(x) sequence

    label_font_size = 26
    tick_font_size  = 24
    legend_font_size = 20
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"
    
    # ax.locator_params("x", nbins =10)    
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(14)
    
    # ax.set_xlim(-0.3, 3.9)
    # ax.set_xlim(-0.3, 4.7)
    ax.set_ylim(0.0, 1.2)
    #error_kw=dict(lw=5, capsize=5, capthick=3)
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    
    # labels=['Baseline (Ring-Allreduce)', 'Global Top-k', 'All-Channel Top-k', 'ACTopk']
    # plt.bar(x, y, width = width, color='blue')
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    ax.bar(x+width*0, y_hon, width = width,label='HO-N', color=default_color[0], hatch='\\\\', edgecolor='white')
    ax.bar(x+width*1, y_omn, width = width,label='OM-N', color=default_color[1], hatch='\\', edgecolor='white')

    ax.bar(x+width*2, y_den, width = width,label='DE-N', hatch='//',color=default_color[2], edgecolor='white')
    ax.bar(x+width*3, y_fgbuff, width = width,label='FG-Bff', hatch='+', edgecolor='white', color=default_color[3],)
    
    # ax.bar(x+width*4, y_gtopk, width = width,label='Traditional Top-k', color=default_color[0], hatch='/', edgecolor='white' )
    # ax2=ax.twinx()
    # ax.bar(x+width*4, y_gtopk, width = width,label='Global Top-k', hatch='x', color=default_color[0], edgecolor='white')

    ax.set_xticks([i+0.3 for i in x], labels)
    
    # set label
    # plt.xlabel('protocol role')
    # plt.ylabel('start time(us)')  
    # ax.set_ylabel('Normalized Convergence Accuracy', fontsize=label_font_size-4)
    ax.set_ylabel('Sequences/Sec', fontsize=label_font_size)
    ax.set_xlabel('Sparsification Methods', fontsize=label_font_size)
    
    ax.tick_params('x', labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y', labelsize=tick_font_size)  #刻度字体大小16

    ax.grid(axis='y', linestyle='--',)
    

    # 输出图片
    # plt.savefig("grace_dll/torch/compressor/global_channel_hybrid_vis_0527/shape_value_bar_channel_missing_bar.pdf", dpi=1400, format='pdf')#eps
    # ax.legend(loc = 2, ncol=1, frameon=False, fontsize=16)
    ax.legend(loc = 9, ncol=4, columnspacing=0.5, labelspacing=0.5, handletextpad=0.2,frameon=False, fontsize=legend_font_size)

    dst_path = "/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/different_sparsification/"

    fig.savefig(dst_path + 'compression_different_sparsification_bert' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'compression_different_sparsification_bert' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()
    return

# test_bar_hatch_different_sparsification_bert()



# pipeline的方法sparsification
def stacked_bar_related_methods_pipeline_sparsification_8GPU():
    width = 0.2 # The width of the bars: can also be len(x) sequence
    
    label_font_size=26
    tick_font_size=23
    legend_font_size=22

    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111) 
    plt.rcParams['font.family'] = "Times New Roman"

    # ax.locator_params("x", nbins =10)

    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(14)  #设置1e6的大小与位置

    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')    
    # labels_ = ['Baseline (Ring-Allreduce)', 'ACTopk', 'DGC', 'Gaussiank', 'Redsync']
    
    # (vanilla)
    labels = ['Baseline', 'Top-k (Vanilla)', 'DGC' , 'Gaussiank', 'OkTopk']
    
    # FWB+IO
    # y1 = [7.0541160106658936, 7.1362268924713135, 6.9399726390838623, 6.9435536861419678, 7.02656841278076]
    
    # FWB
    # labels = ['Baseline', 'Top-k', 'Thres(Acc)', 'DGC' , 'Gaussiank', 'OkTopk']
    labels = ['Baseline', 'OkTopk', 'DGC' , 'Gaussiank', 'Redsync' ]
    
    # FWB
    y1 = [2.0541160106658936, 2.0, 1.9435536861419678, 2.02656841278076, 1.9399726390838623]
    y1=[i * 1000 for i in y1]

    # BWB
    y2 = [5.001363754272461, 5.0, 4.8097883224487305, 5.002656841278076, 4.902656841278076]
    y2=[i * 1000 for i in y2]
    
    # Compress.
    # y3 = [0.0, 3.7711598873138428, 4.289181232452393, 5.2793288230896, 5.7793288230896]
    # y3 = [0.0, 7.180130958557129, 9.75613284111023, 13.644926309585571, 10.7793288230896]
    
    y3 = [0.0, 6.2793288230896, 6.785049200057983, 5.224086093902588, 5.31741189956665]

    y3=[i * 1000 for i in y3]

    # Send Comm.
    y4 = [0.6100244522094727, 0.706688404083,0.706688404083, 0.706688404083, 0.706688404083]
    y4=[i * 1000 for i in y4]
    
    # Receive Comm.
    y5 = [30.764774560928345, 12.098399197 , 13.656080484390259, 14.577541828155518, 14.851789474487305]
    y5=[i * 1000 for i in y5]


    # # FWB
    # y1 = [2.0541160106658936, 2.1362268924713135, 1.9399726390838623, 1.9435536861419678, 2.02656841278076,2.0]
    # y1=[i * 1000 for i in y1]

    # # BWB
    # y2 = [5.001363754272461, 5.03817892074585, 4.902656841278076, 4.8097883224487305, 5.002656841278076, 5.0]
    # y2=[i * 1000 for i in y2]
    
    # # Compress.
    # y3 = [0.0, 3.5013959407806396, 4.2472563560791, 9.7133150100708, 14.44611406326294, 9.502154]
    # y3=[i * 1000 for i in y3]

    # # Send Comm.
    # y4 = [0.6100244522094727, 0.706688404083, 0.706688404083,0.706688404083, 0.706688404083, 0.706688404083]
    # y4=[i * 1000 for i in y4]
    
    # # Receive Comm.
    # y5 = [33.764774560928345, 2.436875104904175, 2.4745419025421143, 11.993355751037598, 14.577541828155518, 12.798399197]
    # y5=[i * 1000 for i in y5]

    
    # 准确阈值, 不存在Inter-worker imbalance
    y=[1.950209140777588, 4.598848342895508, 5.54895544052124, 6.482798099517822, 4.700729846954346]
    
    
    ax.set_ylim(0, 45000)
    # ax.set_xlim(-0.3, 6.9)
    ax.grid(axis='y', linestyle='--',)
    
    # ax.bar(x+width*1, y_oktopk, width = width, label='OkTopk', hatch='/', edgecolor='white')
    width = 0.35
    ax.bar(labels, y1, width, color='dodgerblue', label='Forward', alpha=0.8, hatch='x', edgecolor='black')

    # 关键在Bottom参数
    ax.bar(labels, y2, width, bottom=y1, color='limegreen', label='Backward', alpha=0.8, hatch='+', edgecolor='black')

    bottom_y2 = [i + j for i, j in zip(y1, y2)]
    ax.bar(labels, y3, width, bottom=bottom_y2, color='orange',label='Compression', alpha=0.8, hatch='\\\\', edgecolor='black')
    
    bottom_y3 = [i + j for i, j in zip(bottom_y2, y3)]
    # ax.bar(labels, y4, width, bottom=bottom_y3, color='violet', label='Send Comm.',alpha=0.8, edgecolor='black')

    # bottom_y4 = [i + j for i, j in zip(bottom_y3, y4)]
    # ax.bar(labels, y5, width, bottom=bottom_y4, color='yellow', label='Receive Comm.',alpha=0.8, edgecolor='black')
    
    bottom_y4 = [i + j for i, j in zip(bottom_y3, y4)]
    ax.bar(labels, y5, width, bottom=bottom_y3, color='yellow', label='Communication', alpha=0.8, hatch='//', edgecolor='black')

    labels_x=range(len(labels))
    ax.set_xticks([i for i in labels_x], labels)
    

    ax.set_ylabel('Time/Epoch (ms)', fontsize=label_font_size)
    # ax.set_xlabel('Non-Pipeline Training with VGG-16 on Cifar-100 (Density=0.01)', fontsize=label_font_size)
    # ax.set_xlabel('Pipeline Training', fontsize=label_font_size)
    
    ax.tick_params('x',labelsize=tick_font_size)  #刻度字体大小16
    ax.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16
    
    
    
    # ax.grid(axis='y',linestyle='--',)    
    # plt.title('Stacked bar')
    # plt.show()
    # ax.legend(loc = 0, ncol=2,  columnspacing=0.8, labelspacing=0.8, frameon=False, fontsize=legend_font_size)
    ax.legend(loc = 0, ncol=2,  columnspacing=1.0, labelspacing=1.0, handletextpad=0.2, frameon=False, fontsize=legend_font_size)


    # dst_path='/home/user/eurosys23/workspace/ACTopk/examples/plot_eurosys/bias_threshold/time_decomposition_gpu/'
    
    dst_path='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/breakdown/'
    fig.savefig(dst_path + 'stacked_bar_test_related_methods_pipeline_sparsification_8gpu_0103' + ".jpg", dpi=750, bbox_inches='tight') 
    fig.savefig(dst_path + 'stacked_bar_test_related_methods_pipeline_sparsification_8gpu_0103' + ".pdf", dpi=750, bbox_inches='tight')

    plt.show()

    return

# stacked_bar_related_methods_pipeline_sparsification_8GPU()




# 测量Gradient Merging每个buffer的通信时间
def inter_worker_tensor_fusion_synchronization_time_0104():
    
    dir_path='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/buffer_size_time/'

    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    
    
    buffer_size = [11235428, 6168576, 3089152, 2055936, 905472, 190976, 59712]
    
    # buffer_08 =  [0.29538798332214355, 0.006090879440307617, 0.002086162567138672, 0.0014758110046386719, 0.0007336139678955078, 0.0005426406860351562, 0.0005514621734619141]
    buffer_05 =  [0.043673038482666016, 0.04247093200683594, 0.026651859283447266, 0.018938302993774414, 0.010779857635498047, 0.0046465396881103516, 0.0024394989013671875]
    
    buffer_02 =  [0.025944232940673828, 0.02230215072631836, 0.014388561248779297, 0.011006832122802734, 0.006489276885986328, 0.0030727386474609375, 0.002076387405395508]
    
    buffer_01 =  [0.017274627685546875, 0.015575408935546875, 0.012004852294921875, 0.008620500564575195, 0.0045299530029296875, 0.002554655075073242, 0.001749277114868164]
    
    buffer_005 =  [0.0150110607147216797, 0.014087438583374023, 0.007948160171508789, 0.005747318267822266, 0.0037894248962402344, 0.0023550987243652344, 0.001857757568359375]
    
    buffer_001 =  [0.0034653205871582031, 0.0013434886932373047, 0.0013458728790283203, 0.0012428760528564453, 0.0012214183807373047, 0.0012395381927490234, 0.0011539459228515625]

    # 0.0049215190264643    
    # 0.00430134

    # tensor_np_gaussiank =  np.loadtxt(dir_path+"/average_bias_gaussiank_array_epoch_1.txt")
    # tensor_np_redsync =  np.loadtxt(dir_path+"/average_bias_redsync_array_epoch_1.txt")
    
    x_arr=range(1, len(buffer_01)+1)

    label_font_size = 26
    tick_font_size = 24
    
    legend_font_size=20
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111) 
    
    plt.rcParams['font.family'] = "Times New Roman"
    ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax1.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax1.yaxis.get_offset_text().set_fontsize(16) #设置1e6的大小与位置
    
    # ax1.set_ylim(5500, 6800)
    # ax1.set_xlim(0.5, 7.5)
    ax1.set_ylim(0, 0.06)
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    
    # plt.xticks(ticks, ticks, rotation=30, fontsize=15) 

    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)

    ax1.set_ylabel("Comm Time (Sec)", fontsize=label_font_size)
    # ax1.set_ylabel("Synchronization Time", fontsize =label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index", fontsize =label_font_size)
    ax1.set_xlabel("Buffer ID", fontsize=label_font_size)
    ax1.tick_params(labelsize =tick_font_size)
    
    
    # ax1.grid(axis='y', linestyle='--', )
    
    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray', edgecolor='darkgray', alpha=1.0, label='Non-compression')
    # ax1.plot(x_arr, buffer_08, linewidth=2.0, label='Density=0.8')
    ax1.plot(x_arr, buffer_05, marker='o', linewidth=2.0, label='Density=0.5')
    ax1.plot(x_arr, buffer_02, marker='o', linewidth=2.0, label='Density=0.2')


    ax1.plot(x_arr, buffer_01, marker='o', linewidth=2.0, label='Density=0.1')
    ax1.plot(x_arr, buffer_005, marker='o', linewidth=2.0, label='Density=0.05')
    ax1.plot(x_arr, buffer_001, marker='o', linewidth=2.0, label='Density=0.01')
    
    plt.axhline(y=0.0049251, color='black', linewidth=2.0, linestyle='--', label='Avg Backward')
    
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='ASC-WFBP')
    # ax1.plot(x_arr, buffer_01, linewidth=1.5,label='Ours')

    # ax1.bar(x_arr, tensor_np_dgc, color='darkgray',edgecolor='darkgray',alpha=1.0,label='Non-compression')

    plt.legend(loc = 0, ncol=1,  columnspacing=0.3, labelspacing=0.5, handletextpad=0.2, frameon=False, fontsize=legend_font_size)

    # plt.legend(bbox_to_anchor=(0.5, -0.2),loc=8,ncol=4,fontsize=legend_font_size) # , borderaxespad=0
    
    # bbox_to_anchor 为相对于(0,0)坐标的位置
    # plt.legend(bbox_to_anchor=(0.5, 1.2),loc=9,ncol=4,columnspacing=0.8, labelspacing=0.8,fontsize=legend_font_size-2) # , borderaxespad=0

    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/distribution/shape_value_bar_original_global_hybrid_distribution_all_dimension.jpg',dpi=750, bbox_inches='tight')
    # plt.savefig('grace_dll/torch/compressor/global_dimension_hybrid_vis_0527/dimension_missing/shape_value_bar_original_global_hybrid_distribution_no_sort_epoch=20_importance.jpg',dpi=750, bbox_inches='tight')

    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/buffer_size_time/'
    plt.savefig(dir_fig +'/tensor_fusion_synchronization_time_0104.jpg', dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig +'/tensor_fusion_synchronization_time_0104.pdf', dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return

# inter_worker_tensor_fusion_synchronization_time_0104()



# 测量Gradient Merging每个buffer的大小
def number_tensor_fusion_buffer_size_bar_0104():
    dir_path='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/buffer_size_time/'

    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    # tensor_np =  np.loadtxt("grace_dll/torch/compressor/topk_gradient_fusion_005_0327/0_3910_conv5_x.1.residual_function.3.weight_tensor_flatten_np.txt")
    
    buffer_size = [11235428, 6168576, 3089152, 2055936, 905472, 190976, 99712]
    label_font_size = 26
    tick_font_size = 24
    
    legend_font_size = 22
    
    plt.rcParams['font.family'] = "Times New Roman"
    fig = plt.figure(figsize=(8, 4))
    ax2 = fig.add_subplot(111) 
    
    width = 0.5
    default_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  
    ax2.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax2.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax2.yaxis.get_offset_text().set_fontsize(14)   #设置1e6的大小与位置
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    
    ax2.grid(axis='y', linestyle='--', )
    
    ax2.bar(ticks, buffer_size, zorder=0, color=default_color[0],width = width, hatch='x', edgecolor='white')
    ax2.tick_params('y',labelsize=tick_font_size)  #刻度字体大小16
    
    ticks = [1, 2, 3, 4, 5, 6, 7]  # 指定坐标轴上进行显示的刻度（坐标轴默认的刻度为[0, 0.2, 0.4, 0.6, 0.8, 1.0]）
    labels = ['buffer-1', 'buffer-1', 'buffer-1','buffer-1', 'buffer-1', 'buffer-1','buffer-1']  # 准备与上面指定的坐标轴的刻度对应替换的标签列表
    plt.xticks(ticks, ticks, fontsize=tick_font_size)  # 调用xticks进行设置
    
    # plt.xticks(ticks, ticks, rotation=30, fontsize=15) 

    # ax1.set_ylabel("dimension Count",fontsize=16)
    # ax1.set_xlabel("Magnitude of dimension",fontsize=16)
    
    ax2.set_ylabel("Number of Elements", fontsize=label_font_size)
    # ax1.set_xlabel("Sorted Dimension Index",fontsize=label_font_size)
    ax2.set_xlabel("Buffer ID", fontsize=label_font_size)

    # plt.legend(bbox_to_anchor=(0.5, -0.2),loc=8,ncol=4,fontsize=legend_font_size) # , borderaxespad=0
    
    # bbox_to_anchor 为相对于(0,0)坐标的位置
    # plt.legend(bbox_to_anchor=(0.5, 1.2),loc=9,ncol=4,columnspacing=0.8, labelspacing=0.8,fontsize=legend_font_size-2) # , borderaxespad=0
    plt.legend(loc = 0, ncol=2,  columnspacing=1.0, labelspacing=1.0, handletextpad=0.2, frameon=False, fontsize=legend_font_size)


    dir_fig='/home/user/mzq/workspaces/project/dear_pytorch/ATC24-FG-MGS/fgmgs_ours/result/buffer_size_time/'
    plt.savefig(dir_fig+'/tensor_fusion_buffer_size_0104.jpg', dpi=750, bbox_inches='tight')
    plt.savefig(dir_fig+'/tensor_fusion_buffer_size_0104.pdf', dpi=750, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return


# number_tensor_fusion_buffer_size_bar_0104()









