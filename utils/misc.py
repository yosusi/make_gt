import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn
import torch_pruning as tp
from scipy.ndimage import distance_transform_edt as distance
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
import json
from glob import glob


def get_catheter_pred(folder_name):
    output = {}
    json_file_list = glob(folder_name + '/*/preds_legacy.json')
    for json_file in json_file_list:
        with open(json_file) as file:
            data = json.load(file)
            output.update(data)

    return output


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    # print(t.size())
    _sum = t.sum(axis).type(torch.float32)  # problem
    # print("start_ones")
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    # print(torch.allclose(_sum, _ones))
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def dir_check(path: str) -> bool:
    # Check dir is present
    isdir = os.path.isdir(path)
    assert isdir, f'Not exists [{path}]'


def file_check(path: str):
    # check file present
    isexist = os.path.isfile(path)
    assert isexist, f'Not exists [{path}]'


# calculate distance
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


# convert rt->xy
def rt2xy(img: np.ndarray) -> np.ndarray:
    shape = img.shape
    xyimage = cv2.linearPolar(img, (shape[1] / 2, shape[0] / 2), shape[0] / 2, cv2.WARP_INVERSE_MAP)
    circle_mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    circle_mask = cv2.circle(circle_mask, center=(shape[1] // 2, shape[0] // 2), radius=shape[0] // 2 - 1, color=255,
                             thickness=-1)

    if shape[2] == 4:
        xyimage[circle_mask == 0] = [0, 0, 0, 255]
    elif shape[2] == 3:
        xyimage[circle_mask == 0] = [0, 0, 0]

    return xyimage


def rt2xy_point(r, t):
    """
    r: radius
    t: angle(arb.)
    In function, I used 2*pi*t as angle.
    """
    return np.array([r * np.cos(2 * np.pi * t), r * np.sin(2 * np.pi * t)])


def check_dir(path):
    """
    path : file path
    check path
    if not dir -> create path
    """
    dir_path = path.split('/')[:-1]
    dir_path = '/'.join(dir_path)
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path, exist_ok=True)


# Filters

def transfer_calc(img, transfer_max=0.95, transfer_min=0.05, transfer_linear=2, transfer_gain=0.4):
    """stephen's filter"""
    value = ((pow(img * (transfer_max + transfer_min) - transfer_min, 3.0) + transfer_linear * (img - 0.5)) / (
            1 + transfer_linear)) + transfer_gain
    value[value < 0] = 0
    value[value > 1] = 1
    return value


def min_max(img):
    shape = img.shape
    a = img.reshape((shape[0], -1))
    min = torch.min(a, 1)[0]
    max = torch.max(a, 1)[0]

    min = min.reshape(shape[0], 1)
    max = max.reshape(shape[0], 1)
    dif = max - min

    a = (a - min) / dif
    a = a.reshape(shape)

    return a

# model pruning
def prune_model(model, rate=0.3):
    print(rate)
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 256, 300) )
    def prune_conv(conv, pruned_prob):
        weight = conv.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        num_pruned = int(out_channels * pruned_prob)
        prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        plan = DG.get_pruning_plan(conv, tp.prune_conv, prune_index)
        #rint(plan)
        plan.exec()
    #print(model.modules)
    
    for m in model.input_block:
        if isinstance(m, nn.Conv2d):
            prune_conv(m, rate)
    for m in model.down_blocks:
        for n in m:
            #print(n)
            #prune_conv(n.conv1, rate)
            prune_conv(n.conv2, rate)
            #prune_conv(n.conv3, 0.3)
            #print(n)
            if n.downsample in locals(): 
                print('OK')
                prune_conv(n.downsample[0], rate)
    
    for m in model.bridge.bridge:
        #print(m)
        prune_conv(m.conv, rate)
    for m in model.up_blocks:
        prune_conv(m.conv_block_1.conv, rate)
        prune_conv(m.conv_block_2.conv, rate)
        #if isinstance( m, nn.Sequential):
        #    prune_conv( m[0].conv1, 0.3 )
        #    prune_conv( m[1].conv1, 0.3 )

    return model    


if __name__ == '__main__':
    output = get_catheter_pred('/media/t02/DataCase/project/1_zfix_tenting/Source/catheter_prediction')
    print(output)