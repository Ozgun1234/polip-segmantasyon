import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from ML_Pipeline.network import UNetPP
from argparse import ArgumentParser
import albumentations as A


val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
])


def image_loader(image_name):
    img = imread(image_name)
    img = val_transform(image=img)["image"]
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)

    return img

