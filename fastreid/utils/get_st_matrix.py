import glob
import math
import os

import numpy as np


def joint_similarity(q_cam, q_frame, g_cam, g_frame, distribution):
    gamma = 5
    interval = 100
    score_st = np.zeros((len(q_cam), len(g_cam)))
    for i in range(len(q_cam)):
        for j in range(len(g_cam)):
            if q_frame[i] > g_frame[j]:
                diff = q_frame[i] - g_frame[j]
                hist_ = int(diff / interval)
                pr = distribution[q_cam[i] - 1][g_cam[j] - 1][hist_]
            else:
                diff = g_frame[j] - q_frame[i]
                hist_ = int(diff / interval)
                pr = distribution[g_cam[j] - 1][q_cam[i] - 1][hist_]
            score_st[i][j] = pr

    dist = 1 - 1 / (1 + np.exp(-gamma * score_st))
    return dist

def gaussian_func(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)

def gauss_smooth(arr, o):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i

    approximate_delta = 3*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
            
    xxx = np.dot(matrix,arr)
    return xxx

def get_id(img_path):
    camera_id = []
    labels = []
    frames = []
    for path in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        # frame = filename[9:16]
        frame = filename.split('_')[2][1:]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        frames.append(int(frame))
    return camera_id, labels, frames

def spatial_temporal_distribution(camera_id, labels, frames):
    class_num = len(set(labels))
    cam_num = len(set(camera_id))
    max_hist = 3000
    spatial_temporal_sum = np.zeros((class_num, cam_num))                       
    spatial_temporal_count = np.zeros((class_num, cam_num))
    eps = 0.0000001
    interval = 100.0
    
    for i in range(len(camera_id)):
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i] - 1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        spatial_temporal_sum[label_k][cam_k] += frame_k
        spatial_temporal_count[label_k][cam_k] += 1

    spatial_temporal_avg = spatial_temporal_sum / (spatial_temporal_count + eps)          # spatial_temporal_avg: 751 ids, 6 cameras, center point
    
    distribution = np.zeros((cam_num, cam_num, max_hist))
    for i in range(class_num):
        for j in range(cam_num - 1):
            for k in range(j + 1, cam_num):
                if spatial_temporal_count[i][j] == 0 or spatial_temporal_count[i][k] == 0:
                    continue
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij > st_ik:
                    diff = st_ij - st_ik
                    hist_ = int(diff / interval)
                    distribution[j][k][hist_] += 1     # [big][small]
                else:
                    diff = st_ik - st_ij
                    hist_ = int(diff / interval)
                    distribution[k][j][hist_] += 1
    smooth = 50
    for i in range(cam_num):
        for j in range(cam_num):
            #print("gauss "+str(i)+"->"+str(j))
            distribution[i][j][:] = gauss_smooth(distribution[i][j][:],smooth)

    sum_ = np.sum(distribution, axis=2)

    for i in range(cam_num):
        for j in range(cam_num):
            distribution[i][j] /= (sum_[i][j] + eps)
    
    import pdb;pdb.set_trace()

    return distribution  # [to][from], to xxx camera, from xxx camera

def get_st_matrix(imgs_path, pseudo_labels=None):
    train_cam, train_label, train_frames = get_id(imgs_path)

    if pseudo_labels is not None:
        labels = pseudo_labels
    else:
        labels = train_label

    reordered_label = []
    dic = {}
    i = 0
    for label in labels:
        if dic.get(label) is None:
            dic[label] = i
            i += 1
        reordered_label.append(dic[label])

    distribution = spatial_temporal_distribution(train_cam, reordered_label, train_frames)
    matrix = joint_similarity(train_cam, train_frames, train_cam, train_frames, distribution)

    return matrix

if __name__ == "__main__":
    data_dir = "datasets/market1501/bounding_box_train"
    imgs_path = glob.glob(os.path.join(data_dir, '*.jpg'))
    matrix = get_st_matrix(imgs_path, pseudo_labels=None)
    import pdb;pdb.set_trace()
    print(123)
