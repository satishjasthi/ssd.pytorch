import os
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from data import VOC_CLASSES as labels

ProjectRoot = Path(__file__).resolve().parent

class SSD_Api:

  def __init__(self):
    self.net = build_ssd('test', 300,21)
    self.net.load_weights(str(ProjectRoot/'weights/ssd300_mAP_77.43_v2.pth'))

  def predict(self, cv2_image:cv2.imread):
    """
    Function to predict the bbox of a person
    in given image
    """
    top_k=10
    temp_img = "temp.jpg"
    cv2.imwrite(temp_img, cv2_image)
    cv2_image = cv2.imread(temp_img, cv2.IMREAD_COLOR)
    # resize
    x = cv2.resize(cv2_image, (300, 300)).astype(np.float32)
    # standardize
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # convert into tensor
    x = torch.from_numpy(x).permute(2, 0, 1)
    # forward pass through network
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = self.net(xx)
    # parse the detections
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(cv2_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
      j = 0
      while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        if label_name == 'person':
          pt = (detections[0,i,j,1:]*scale).cpu().numpy()
          coords = int(pt[0]), int(pt[1]), int(pt[2]-pt[0]+1), int(pt[3]-pt[1]+1)
          break
        else:
          coords = None
    os.remove(temp_img)
    return coords


  def crop_person(self, cv2_image:cv2.imread, side=False):
    """
    Function to crop a person from image
    """
    coords = self.predict(cv2_image)
    x1, y1, x2, y2 = coords
    # adjust the coords
    try:
      # adjust the coordinates so that entire person can be captured
      if side:
        cv2_image = cv2_image[y1 - 150 : y2 + 100, x1 - 100 : x2 + 1000]
      else:
        cv2_image = cv2_image[y1 - 150 : y2 + 1000, x1 - 120 : x2 + 450]
    except Exception as e:
        print(e)
    Image.fromarray(cv2_image.astype(np.uint8)).show()
    return cv2_image


if __name__ == "__main__":
  m = SSD_Api()
  image = cv2.imread('/Volumes/SatishJ 1/ML/Detalytics/ssd.pytorch/images/amit-front-darkbg.jpeg')
  m.crop_person(image,side=False)