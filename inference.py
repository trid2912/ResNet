import os

import numpy as np
from PIL import Image

import os
import torch
import torch.nn as nn
import argparse
import random
import time

from configs import cfg
from datasets import *
from models import DeeplabV3plus
from utils import setup_logger, IoU, OverallAcc
import segmentation_models_pytorch as smp

def multi_scale_inference(self, model, image, scales=[1], flip=False, deq=True):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            # new_img = self.multi_scale_aug(image=image,
            #                                rand_scale=scale,
            #                                rand_crop=False)
            new_img = cv2.resize(image, (int(scale * ori_width), int(scale * ori_height)), cv2.INTER_LINEAR)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip, deq)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip, deq)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear')
            final_pred += preds
        return final_pred

model = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
saved = torch.load("/home/kc/tritd/models/train_deeplabv3+/best_model(ResNet50).pkl")
model.load_state_dict(saved['model_state_dict'])
img = Image.open(multi_scale_inference(model=model, image= "/home/kc/luantt/kaggle_data/dataset-medium/image-chips/1d4fbe33f3_F1BE1D4184INSPIRE-000000.png", deq=False))
img.show()