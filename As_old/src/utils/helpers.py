import cv2
import torch
import numpy as np
from pytorch_lightning.metrics import functional


def preds_to_integer(preds):

    preds = functional.to_categorical(preds)
    out=[]
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != preds[i - 1]:
            out.append(preds[i])
    return torch.tensor(out)

class Resize(object):

    def __init__(self, height=32, dynamic_height=0):
        self.height = height
        if dynamic_height > 0:
            print('dynamic height feature is disabled')
        self.dynamic_height = 0

    def __call__(self, img):
        # increase the height randomly by a number between 0 and dynamic height
        height_increase = np.random.randint(0, self.dynamic_height+1)
        current_height = self.height + height_increase
        img = cv2.resize(img, (int(current_height * img.shape[1] / img.shape[0]), current_height))
        return img