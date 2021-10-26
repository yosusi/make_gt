import numpy as np
import torch


def mask_to_GT(mask, mode="RBYG"):
    """
    mask image to gt layer
    0,0,0 -> black
    255,0,0 -> Red
    0,0,255 -> Blue
    255,255,0 -> Yellow
    0,255,0 -> Green
    ------------------
    mask : torch.Tensor in cuda()
    ground truth picture

    mode:"RBYG" or "RB"
    default is RBYG mode
    """
    mask_shape = mask.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_num = 5
    color_list = torch.Tensor([
        [255, 0, 0],  # Red:Main vessel
        [0, 0, 255],  # Blue:Catheter
        [255, 255, 0],  # Yellow:SubVessel
        [0, 255, 0],  # Green:outer
        [0, 0, 0]  # Black:Tissue
    ]).to(device)

    if mode == "RB":#this GT layer is not same as RBYG black is mainvessel and something
        layer_num = 3
        color_list = torch.Tensor([
            [0, 0, 0],  # Black:NOT Tissue or Catheter
            [0, 0, 255],  # Blue: Catheter
            [255, 0, 0]  # Red: Tissue
            ]).to(device)

    if mode == "leftred":#mode for left red
        layer_num = 3
        color_list = torch.Tensor([
            [255, 0, 0],  # Red: Main vessel
            [0, 0, 255],  # Blue: Catheter
            [0, 0, 0]  # Black:NOT Tissue or main vessel
            ]).to(device)

    target = torch.zeros((layer_num, mask_shape[0], mask_shape[1])).to(device)
    for i in range(layer_num):
        #print(mask[:, :, 0:3])
        target[i][(mask[:, :, 0:3] == color_list[i]).all(dim=2)] = 1

    return target


def prediction_to_pic(prediction, mode="RBYG"):
    """
    prediction to image
    0,0,0 -> black
    255,0,0 -> Red
    0,0,255 -> Blue
    255,255,0 -> Yellow
    0,255,0 -> Green
    ------------------
    prediction : torch.Tensor
    ground truth picture

    mode:"RBYG" or "RB"
    default is RBYG mode
    TODO: replace for Torch

    """
    prediction_shape = prediction.shape
    layer_num = 5
    color_list = np.array([
        [255, 0, 0],  # Red:Main vessel
        [0, 0, 255],  # Blue:Catheter
        [255, 255, 0],  # Yellow:SubVessel
        [0, 255, 0],  # Green:outer
        [0, 0, 0]  # Black:Tissue
    ])

    if mode == "RB":
        layer_num = 3
        color_list = np.array([
            [255, 0, 0],  # Red: main
            [0, 0, 255],  # Blue: Catheter
            [0, 0, 0]  # Black:NOT Tissue and not main
        ])
    if mode == "old":
        layer_num = 5
        color_list = np.array([
            [0, 0, 0],  # Black:Tissue
            [0, 0, 255],  # Blue: Catheter
            [255, 0, 0],  # Red:Main vessel
            [0, 255, 0],  # Green:outer
            [255, 255, 0]  # Yellow:SubVessel
        ])
    _, max_color = prediction.cpu().max(1)
    max_color.numpy()
    #print(max_color)
    image = np.zeros((prediction_shape[2], prediction_shape[3], 3), dtype=np.uint8)
    #print(image.shape)
    for i in range(layer_num):
        image[max_color[0] == i] = color_list[i]
    return image
