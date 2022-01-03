import numpy as np
import torch

# _min = -23.1728
# _max = 20.3891

_scale = 23.4838
_min = -23.1728

def quant_fix(features):
    for name, pyramid in features.items():
        pyramid_q = (pyramid-_min) * _scale
        features[name] = pyramid_q
    return features
    
def dequant_fix(x):
    return x.type(torch.float32)/_scale + _min

        