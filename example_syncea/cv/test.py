import time
import torch
import numpy as np
import sys
sys.path.append("../..") 
import utils
from profiling import CommunicationProfiler




sub_buffer = utils.optimal_gradient_merging_0101(11111, 'resnet50', density=0.1)
