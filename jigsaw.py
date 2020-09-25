import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import re

def create_cropping(img, block=8):
    """create cropping block \times block.
    """
    h, w, c = img.shape
    h_size, w_size = h // block, w // block
    img_crops = np.zeros((block, block, h_size, w_size, c), dtype=np.uint8)
    for row in range(block):
        for col in range(block):
            img_crops[row, col] = img[row * h_size: (row + 1) * h_size, col * w_size : (col + 1) * w_size]
    return img_crops

def permute(img_crops):
    size = img_crops.shape[0] * img_crops.shape[1]
    h = img_crops.shape[0] * img_crops.shape[2]
    w = img_crops.shape[1] * img_crops.shape[3]
    new_order = np.random.permutation(np.arange(size))
    new_img_crops = np.zeros(img_crops.shape)
    for index, i in enumerate(new_order):
        new_img_crops[index // img_crops.shape[1], index % img_crops.shape[1]] = img_crops[i // img_crops.shape[1], i % img_crops.shape[1]]
    return new_img_crops.transpose(0, 2, 1, 3, 4).reshape((h, w, img_crops.shape[-1])).astype(np.uint8)

def read_img(files):
    return [cv2.imread(file) for file in files]

if __name__ == '__main__':
    files = ['/home/wuyiming/temp/duke/' + name for name in os.listdir('/home/wuyiming/temp/duke/')]
    imgs = read_img(files)
    new_imgs = []
    for img in imgs:
        img = img.copy()
        img_crop = create_cropping(img)
        new_img = permute(img_crop)
        plt.figure()
        plt.subplot(121)
        plt.imshow(img[:,:,::-1])
        plt.subplot(122)
        plt.imshow(new_img[:,:,::-1])
        plt.show()
        new_imgs.append(new_img)
    for file, new_img in zip(files, new_imgs):
        cv2.imwrite(file.strip('.jpg') + '_jigsaw.jpg', new_img)
