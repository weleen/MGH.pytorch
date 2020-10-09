import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch


def create_cropping(img, block=4):
    """create cropping block \times block.
    """
    c, h, w = img.shape
    h_size, w_size = h // block, w // block
    img_crops = torch.zeros((c, block, block, h_size, w_size), dtype=img.dtype)
    for row in range(block):
        for col in range(block):
            img_crops[:, row, col] = img[:, row * h_size: (row + 1) * h_size, col * w_size: (col + 1) * w_size]
    return img_crops


def shuffle_cropping(img_crops):
    """
    :param img_crops(torch.Tensor): size (C,H,W) tensor.
    :return:
    """
    c, block, _, h_size, w_size = img_crops.shape
    h, w = block * h_size, block * w_size
    for _ in range(20):
        order = np.arange(block * block)
        new_order = np.random.permutation(order)
        if np.abs(new_order - order).mean() < block:
            continue
        new_img = torch.zeros_like(img_crops)
        for index, i in enumerate(new_order):
            new_img[:, index // block, index % block] = img_crops[:, i // block, i % block]
        return new_img.transpose(2, 3).contiguous().view(c, h, w)

def create_jigsaw(img, block=4):
    """
    convert img to jigsaw img.
    :param img: (C, H, W) tensor.
    :return:
    """
    img_crops = create_cropping(img, block=block)
    new_img = shuffle_cropping(img_crops)
    return new_img

def read_img(files):
    return [cv2.imread(file) for file in files]


if __name__ == '__main__':
    files = ['/home/wuyiming/temp/duke/' + name for name in os.listdir('/home/wuyiming/temp/duke/')]
    imgs = read_img(files)
    imgs = torch.Tensor([img.transpose(1, 2, 0) for img in imgs])
    new_imgs = []
    for img in imgs:
        img = img.clone()
        img_crop = create_cropping(img)
        new_img = permute(img_crop)
        plt.figure()
        plt.subplot(121)
        plt.imshow(img[:, :, ::-1])
        plt.subplot(122)
        plt.imshow(new_img[:, :, ::-1])
        plt.show()
        new_imgs.append(new_img)
    for file, new_img in zip(files, new_imgs):
        cv2.imwrite(file.strip('.jpg') + '_jigsaw.jpg', new_img)
