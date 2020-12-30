import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import math
import imgaug.augmenters as iaa
import torch
from utils.bbox import quad_2_rbox, rbox_2_quad, mask_valid_boxes


class HSV(object):
    def __init__(self , saturation=0, brightness=0, p=0.):
        self.saturation = saturation 
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(-1, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels


class HSV_pos(object):
    def __init__(self , saturation=0, brightness=0, p=0.):
        self.saturation = saturation 
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(0, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels    

class Blur(object):
    def __init__(self, sigma=0 ,p=0.):
        self.sigma = sigma 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0,self.sigma))
            img = blur_aug.augment_image(img)
        return img, labels


class Grayscale(object):
    def __init__(self, grayscale=0. ,p=0.):
        self.alpha = random.uniform(grayscale,1.0)
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, labels


class Gamma(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            gm = random.uniform(1-self.intensity,1+self.intensity)
            img = np.uint8(np.power(img/float(np.max(img)), gm)*np.max(img))
        return img, labels


class Noise(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, labels



class Sharpen(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity,1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, labels


class Contrast(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            contrast_aug = aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img=contrast_aug.augment_image(img)
        return img, labels


####################################
class HorizontalFlip(object):
    def __init__(self, p=0.):
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            img = np.fliplr(img)
            if mode == 'cxywha':    
                labels[:, 1] = img.shape[1] - labels[:, 1]
                labels[:, 5] = -labels[:, 5]
            if mode == 'xyxyxyxy':
                labels[:, [0,2,4,6]] = img.shape[1] - labels[:, [0,2,4,6]]
            if mode == 'xywha':
                labels[:, 0] = img.shape[1] - labels[:, 0]
                labels[:, -1] = -labels[:, -1]                
        return img, labels        


class VerticalFlip(object):
    def __init__(self ,p=0.):
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            img = np.flipud(img)
            if mode == 'cxywha': 
                labels[:, 2] = img.shape[0] - labels[:, 2]
                labels[:, 5] = -labels[:, 5]
            if mode == 'xyxyxyxy':
                labels[:, [1,3,5,7]] = img.shape[0] - labels[:, [1,3,5,7]]
            if mode == 'xywha':
                labels[:, 1] = img.shape[0] - labels[:, 1]
                labels[:, -1] = -labels[:, -1]   
        return img, labels 


class Affine(object):
    def __init__(self, degree = 0., translate = 0., scale = 0., shear = 0., p=0.):
        self.degree = degree 
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            if mode == 'xywha':
                labels = rbox_2_quad(labels, mode = 'xywha')
                img, labels = random_affine(img, labels, 
                            degree=self.degree,translate=self.translate,
                            scale=self.scale,shear=self.shear ) 
                labels = quad_2_rbox(labels, mode = 'xywha')

            else:
                img, labels = random_affine(img, labels, 
                                degree=self.degree,translate=self.translate,
                                scale=self.scale,shear=self.shear ) 
        return img, labels 


class Augment(object):
    def __init__(self, augmentations, probs=1, box_mode=None):
        self.augmentations = augmentations
        self.probs = probs
        self.mode = box_mode
        
    def __call__(self, img, labels):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                img, labels = augmentation(img, labels, self.mode)

        return img, labels







def random_affine(img,  targets=(), degree=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degree=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    
    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    # # # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)


    M =  T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))  # BGR order borderValue

    # Return warped points also
    t = targets.copy()
    targets[:, [0,2,4,6]] = t[:, [0,2,4,6]] * M[0,0] + t[:, [1,3,5,7]] * M[0,1] + M[0,2]
    targets[:, [1,3,5,7]] = t[:, [0,2,4,6]] * M[1,0] + t[:, [1,3,5,7]] * M[1,1] + M[1,2]
    for x in range(0,8,2):
        targets[:,x] = targets[:,x].clip(0, width)
    for y in range(1,8,2):
        targets[:,y] = targets[:,y].clip(0, height)
    return imw, targets



def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2, x1y1x2y2=True):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # random mask_size up to 50% image size
    mask_h = random.randint(1, int(h * 0.5))
    mask_w = random.randint(1, int(w * 0.5))

    # box center
    cx = random.randint(0, h)
    cy = random.randint(0, w)

    xmin = max(0, cx - mask_w // 2)
    ymin = max(0, cy - mask_h // 2)
    xmax = min(w, xmin + mask_w)
    ymax = min(h, ymin + mask_h)

    # apply random color mask
    mask_color = [random.randint(0, 255) for _ in range(3)]
    image[ymin:ymax, xmin:xmax] = mask_color

    # return unobscured labels
    if len(labels):
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        labels = labels[ioa < 0.90]  # remove >90% obscured labels
    return labels

