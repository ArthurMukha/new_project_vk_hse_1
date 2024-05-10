import cv2
from PIL import Image, ImageStat
import math
import numpy as np
from numpy import asarray
import csv
import os
import itertools


def brightness(im_file):
    im = Image.open(im_file).convert("RGB")
    stat = ImageStat.Stat(im)
    print(stat)
    gs = (math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
          for r, g, b in im.getdata())
    return sum(gs) / stat.count[0]


def BlurValue(img):
    canny = cv2.Canny(img, 50, 250)
    return np.mean(canny)


def contrastValue(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    kernel = np.ones((5, 5), np.uint8)
    min = cv2.erode(L, kernel, iterations=1)
    max = cv2.dilate(L, kernel, iterations=1)
    min = min.astype(np.float64)
    max = max.astype(np.float64)
    contrast = (max - min) / (max + min)
    average_contrast = 100 * np.mean(contrast)
    return average_contrast


def SatValue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    return saturation


def heurestic(img_path):
    bright = round(brightness(img_path) * 100 / 255 / 10, 2)
    original_image = Image.open(img_path).convert("RGB")
    original_image = np.array(original_image)
    img = original_image

    blur = round(BlurValue(img) / 10, 2)
    contrast = round(contrastValue(img) / 10, 2)
    saturation = round(SatValue(img) * 100 / 255 / 10, 2)
    if not np.isnan(contrast):
        score = round((0.8 * bright + 0.1 * blur + 0.05 * contrast + 0.05 * saturation), 2)
    else:
        score = round((0.8 * bright + 0.1 * blur + 0.05 * saturation), 2)
    return score