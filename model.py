# model class
import cv2
import numpy as np
import torch
import sys
from pathlib import Path                                 # 追加
sys.path.append(str(Path(__file__).parent.parent)) 
import glob
import json
import yaml
import os
import json
from pathlib import Path
from utils import mask_gt_conv, misc,halfUnet, pruning_model
from PIL import Image
import time


class SegmentationModel():
    '''
    Segmentation part of tenting detection only for making GT
    without catheter hint
    '''

    def __init__(self):
        self.dir_path = os.getcwd()
        print(self.dir_path)
        #model_path = /home/t02/IVUS/resource/model_report/toasty-sweep-1511.pth # halfunet
        model_path = "/home/t02/IVUS/resource/model_report/report5_pruning/devoted-sweep-5923pruning70_time_1nnc.pth" # pruning07
        self.model = halfUnet.UNetWithPolarResnet50Encoder_half(5)
        self.model = pruning_model.manual_pruning_07(self.model)
        self.model.cuda()
        self.model.eval()

        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)))


    def transfer_calc(self, img, transfer_max=0.95, transfer_min=0.05, transfer_linear=2, transfer_gain=0.4):
        """stephen's filter"""
        value = ((pow(img * (transfer_max + transfer_min) - transfer_min, 3.0) + transfer_linear * (img - 0.5)) / (
                1 + transfer_linear)) + transfer_gain
        value[value < 0] = 0
        value[value > 1] = 1
        return value

    def min_max(self, img):
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

    def input_preparation(self, file_path: str, cath=None) -> torch.Tensor:
        img = torch.Tensor(np.array(Image.open(file_path).convert("L").resize((300, 256))))
        img = img / 255
        img.cuda()
        norm = self.min_max(img).cuda()
        transfer = self.transfer_calc(img).cuda()
        cath_hint = torch.zeros(img.shape).cuda()
        if cath != None:
            for c in cath:
                cath_hint[int(c[1]),int(c[0])] = 1

        
        input_layer = torch.stack([norm, norm, cath_hint], dim=0)
        input_layer = torch.unsqueeze(input_layer, 0)

        return input_layer

    def picture_check(self, file_path, out_path, cath=None):
        input_layer = self.input_preparation(file_path, cath)
        output = self.model(input_layer)
        image = mask_gt_conv.prediction_to_pic(output)
        #out_path = file_path.replace("rt", "rt_pred")
        pil_img = Image.fromarray(image, mode="RGB")
        misc.check_dir(out_path)
        pil_img.save(out_path)

    def __call__(self, file_path) -> np.array:
        #start = time.time()
        input_layer = self.input_preparation(file_path)
        output = self.model(input_layer)
        torch.cuda.synchronize()
        #mid = time.time()
        _, max_color = output.max(1)
        #max_color = torch.squeeze(max_color, 1)

        return max_color

    def segmentation_with_pic(self, file_path) -> np.array:
        input_layer = self.input_preparation(file_path)
        output = self.model(input_layer)
        _, max_color = output.max(1)
        pred_image = mask_gt_conv.prediction_to_pic(max_color)
        #max_color = torch.squeeze(max_color, 1)

        return max_color, pred_image


    def save_pic(self, file_path, is_resize = False):
        save_path = file_path.replace('rt', 'RBY')
        pdir = str(Path(save_path).parents[0])
        pdir = pdir.replace('rt', 'RBY_prune_devoted')
        os.makedirs(pdir, exist_ok=True)
        input_layer = self.input_preparation(file_path)
        output = self.model(input_layer)
        #_, max_color = output.max(1)
        image = mask_gt_conv.prediction_to_pic(output)
        if is_resize:
            im = Image.fromarray(image).resize((600,512), resample=Image.NEAREST)
        else:
            im = Image.fromarray(image)
        im.save(save_path)


if __name__ == "__main__":
    #Test code
    m = SegmentationModel()
    cath_pred = misc.get_catheter_pred('/media/t02/DataCase/project/1_zfix_tenting/Source/catheter_prediction')
    folder_list = glob.glob('/media/t02/DataCase/project/1_zfix_tenting/DataSet/*/*')
    for folder in folder_list:
        seq = Path(folder).stem
        catheter = cath_pred[seq]
        #print(len(catheter))
        for i in range(len(catheter)):
            file_path = f'{folder}/rt/{i}.png'
            save_path = file_path.replace('rt', 'RBY_cath')
            pdir = str(Path(save_path).parents[0])
            os.makedirs(pdir, exist_ok=True)
            cath = catheter[i]
            input_layer = m.input_preparation(file_path,cath=cath)
            output = m.model(input_layer)
            image = mask_gt_conv.prediction_to_pic(output)
            im = Image.fromarray(image)
            im.save(save_path)
            #print(save_path)
            #print(cath)

    







