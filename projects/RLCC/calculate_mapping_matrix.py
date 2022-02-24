import argparse
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import torch


def iou(i, m2):
    for j in range(m2):
        shared_array[i][j] = np.minimum(I1[i], I2[j]).sum() / np.maximum(I1[i], I2[j]).sum()


def iou_bin(m1, j, I_1, I_2):
    for i in range(m1):
        inter = int(I_1[i], 2) & int(I_2[j], 2)
        union = int(I_1[i], 2) | int(I_2[j], 2)
        shared_arr[i][j] = 1.0 * inter / union


def gt(m1, m2, last_labels, labels):
    print('gt')
    st = time.time()
    C = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            set1 = set(np.where(last_labels == i)[0])
            set2 = set(np.where(labels == j)[0])
            C[i, j] = len(set1.intersection(set2)) * 1.0 / len(set1.union(set2))
    et = time.time()
    print('Task done in {}s'.format(et - st))
    C /= C.sum(-1, keepdims=True)
    return C


def method_1(m1, m2, last_labels, labels):
    a = time.time()
    print('method_1')
    p = Pool()
    for i in range(m1):
        p.apply_async(func=iou, args=(i, m2))
    p.close()
    p.join()
    b = time.time()
    print('Task done in {}s.'.format(b - a))
    # Task done in 575.8600080013275s (about 10min)


def method_2(m1, m2, num_memory, last_labels, labels):
    print('method_2')
    st = time.time()
    I1 = [0] * m1
    I2 = [0] * m2
    for i in range(num_memory):
        tmp = int('1' + i * '0', 2)
        I1[last_labels[i].item()] = I1[last_labels[i].item()] | tmp
        I2[labels[i].item()] = I2[labels[i].item()] | tmp

    def func2(num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        return count

    # C = torch.zeros((m1, m2))
    C = np.zeros((m1, m2))
    for i, j in itertools.product(range(m1), range(m2)):
        inter = func2(I1[i] & I2[j])
        union = func2(I1[i] | I2[j])
        C[i, j] = 1.0 * inter / union
    et = time.time()
    print('Task done in {}s'.format(et - st))
    C /= C.sum(-1, keepdims=True)
    return C


def method_3(m1, m2, num_memory, last_labels, labels):
    print('method_3')
    st = time.time()
    I1 = [0] * m1
    I2 = [0] * m2
    for i in range(num_memory):
        tmp = int('1' + i * '0', 2)
        I1[last_labels[i].item()] = I1[last_labels[i].item()] | tmp
        I2[labels[i].item()] = I2[labels[i].item()] | tmp

    def func2(num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        return count

    p = Pool()
    for j in range(m2):
        p.apply_async(func=iou_bin, args=(m1, j, I1, I2))
    p.close()
    p.join()
    et = time.time()
    print('Task done in {}s.'.format(et - st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate mapping matrix between two generation.')
    parser.add_argument('--file', '-f', required=True)
    args = parser.parse_args()

    file_name = args.file
    res = torch.load(file_name)
    last_labels, labels = res['last_label'].cpu(), res['label'].cpu()
    num_memory = len(labels)
    labels = labels.numpy()
    last_labels = last_labels.numpy()

    assert len(last_labels) == num_memory
    m1, m2 = last_labels.max() + 1, labels.max() + 1

    # gt
    # C_gt = gt(m1, m2, last_labels, labels)
    # C_gt = np.fromfile('C_gt.bin').reshape(m1, m2)

    # # 方法一
    # t1 = np.ctypeslib.as_ctypes(np.zeros((m1, num_memory)))
    # I1 = sharedctypes.RawArray(t1._type_, t1)
    # t2 = np.ctypeslib.as_ctypes(np.zeros((m2, num_memory)))
    # I2 = sharedctypes.RawArray(t2._type_, t2)
    # for i in range(num_memory):
    #     I1[last_labels[i]][i] = 1
    #     I2[labels[i]][i] = 1

    # C = np.ctypeslib.as_ctypes(np.zeros((m1, m2)))
    # shared_array = sharedctypes.RawArray(C._type_, C)

    # method_1(m1, m2, last_labels, labels)
    # C = np.ctypeslib.as_array(shared_array)
    # C /= C.sum(-1, keepdims=True)
    # assert np.allclose(C_gt, C)

    # 方法二
    C_2 = method_2(m1, m2, num_memory, last_labels, labels)
    # assert np.allclose(C_gt, C_2)

    # # 方法三
    # C_3 = np.ctypeslib.as_ctypes(np.zeros((m1, m2)))
    # shared_arr = sharedctypes.RawArray(C_3._type_, C_3)
    # C_3 = method_3(m1, m2, num_memory, last_labels, labels)
    # C_3 = np.ctypeslib.as_array(shared_array)
    # C_3 /= C_3.sum(-1, keepdims=True)
    # assert np.allclose(C_gt, C_3)

    C = torch.from_numpy(C_2)
    torch.save(C, os.path.join(os.path.dirname(file_name), 'C.pt'))
