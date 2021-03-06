import glob
import math
import os

import numpy as np


def joint_similarity(q_cam, q_frame, g_cam, g_frame, distribution, score=None):
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
    if score is None:
        dist = 1 - 1 / (1 + 2 * np.exp(-gamma * score_st))
        return dist
    else:
        assert score.shape == score_st.shape
        dist = 1 - 1 / (1 + np.exp(-gamma * score)) * 1 / (1 + 2 * np.exp(-gamma * score_st)) # https://github.com/ljn114514/JVTC/blob/master/utils/st_distribution.py#L5
        return dist

def single_process(q_frame, g_frame, q_cam, g_cam, i, distribution):
    # print(i, 'running')
    interval = 100
    score_st_i = np.zeros(len(g_cam))
    for j in range(len(g_cam)):
        if q_frame > g_frame[j]:
            diff = q_frame - g_frame[j]
            hist_ = int(diff / interval)
            pr = distribution[q_cam - 1][g_cam[j] - 1][hist_]
        else:
            diff = g_frame[j] - q_frame
            hist_ = int(diff / interval)
            pr = distribution[g_cam[j] - 1][q_cam - 1][hist_]
        score_st_i[j] = pr
    return score_st_i

def joint_similarity_parallel(q_cam, q_frame, g_cam, g_frame, distribution, score=None):
    gamma = 5

    import multiprocessing
    print('multiple processes start')
    print(multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(maxtasksperchild=10)
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    res_list = []
    for i in range(len(q_cam)):
        res = pool.apply_async(func=single_process, args=(q_frame[i], g_frame, q_cam[i], g_cam, i, distribution))
        res_list.append(res)
            
    pool.close()
    pool.join()
    print('multiple processes finished')

    res_list = [res.get() for res in res_list]
    score_st = np.asarray(res_list)

    if score is None:
        dist = 1 - 1 / (1 + 2 * np.exp(-gamma * score_st))
        return dist
    else:
        assert score.shape == score_st.shape
        dist = 1 - 1 / (1 + np.exp(-gamma * score)) * 1 / (1 + 2 * np.exp(-gamma * score_st)) # https://github.com/ljn114514/JVTC/blob/master/utils/st_distribution.py#L5
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
        if "MSMT17" in path:
            camera, period, frame = filename.split('_')[2:5]
            camera_id.append(int(camera))
        else:
            camera = filename.split('c')[1]
            if "Market" in path:
                frame = filename.split('_')[2][1:]
            elif "DukeMTMC" in path:
                frame = filename[9:16]
            camera_id.append(int(camera[0]))
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        frames.append(int(frame))
    return camera_id, labels, frames

def spatial_temporal_distribution(camera_id, labels, frames):  # code from https://github.com/Wanggcong/Spatial-Temporal-Re-identification/blob/master/gen_st_model_market.py#L48
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

    return distribution  # [to][from], to xxx camera, from xxx camera

def get_st_matrix(imgs_path, pseudo_labels=None, score=None):
    train_cam, train_label, train_frames = get_id(imgs_path)
    if pseudo_labels is not None:
        labels = pseudo_labels
    else:
        labels = train_label
    
    inlier_ind = np.where(np.array(labels, dtype=np.int64) != -1)[0]
    # select inlier
    t_cam, t_label, t_frames = [], [], []
    for ind in inlier_ind:
        t_cam.append(train_cam[ind])
        t_label.append(labels[ind])
        t_frames.append(train_frames[ind])

    reordered_label = []
    dic = {}
    i = 0
    for label in t_label:
        if dic.get(label) is None:
            dic[label] = i
            i += 1
        reordered_label.append(dic[label])

    import time
    a = time.time()
    # file_path = 'distribution.npy'
    # if not os.path.exists(file_path):
    #     # distribution = spatial_temporal_distribution(t_cam, reordered_label, t_frames)
    distribution = get_st_distribution(t_cam, reordered_label, t_frames)
    #     np.save(file_path, distribution)
    # else:
    #     distribution = np.load(file_path)
    b = time.time()
    print('spatial temporal matrix computation get_st_distribution cost: {}'.format(b - a))
    print('Use multiprocessing')
    matrix = joint_similarity_parallel(train_cam, train_frames, train_cam, train_frames, distribution, score)
    c = time.time()
    print(f'cost {c - b}s')
    # np.save('matrix2.npy', matrix2)
    # print('Use single process')
    # matrix1 = joint_similarity(train_cam, train_frames, train_cam, train_frames, distribution, score) # if score is not None, calculate distance matrix based on joint similarity
    # np.save('matrix1.npy', matrix1)
    # print(f'cost {time.time() - c}s')
    # assert np.allclose(matrix1, matrix)
    # raise ValueError
    # np.save('spatial_temporal_matrix_market.npy', matrix)
    return matrix

def get_st_distribution(camera_id, labels, frames):  # code from https://github.com/ljn114514/JVTC/blob/master/utils/st_distribution.py#L72:5
    id_num = len(set(labels))
    cam_num = len(set(camera_id))
    spatial_temporal_sum = np.zeros((id_num, cam_num))
    spatial_temporal_count = np.zeros((id_num, cam_num))
    eps = 0.0000001
    interval = 100.0

    for i in range(len(camera_id)):
        label_k = int(labels[i])  #### not in order, done
        cam_k = int(
            camera_id[i] - 1
        )  ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]

        spatial_temporal_sum[label_k][
            cam_k] = spatial_temporal_sum[label_k][cam_k] + frame_k
        spatial_temporal_count[label_k][
            cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum / (
        spatial_temporal_count + eps
    )  # spatial_temporal_avg: 702 ids, 8cameras, center point

    distribution = np.zeros((cam_num, cam_num, 3000))
    for i in range(id_num):
        for j in range(cam_num - 1):
            for k in range(j + 1, cam_num):
                if spatial_temporal_count[i][j] == 0 or spatial_temporal_count[
                        i][k] == 0:
                    continue
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij > st_ik:
                    diff = st_ij - st_ik
                    hist_ = int(diff / interval)
                    distribution[j][k][
                        hist_] = distribution[j][k][hist_] + 1  # [big][small]
                else:
                    diff = st_ik - st_ij
                    hist_ = int(diff / interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_] + 1

    for i in range(id_num):
        for j in range(cam_num):
            if spatial_temporal_count[i][j] > 1:

                frames_same_cam = []
                for k in range(len(camera_id)):
                    if labels[k] == i and camera_id[k] - 1 == j:
                        frames_same_cam.append(frames[k])
                frame_id_min = min(frames_same_cam)

                #print 'id, cam, len',i, j, len(frames_same_cam)
                for item in frames_same_cam:
                    #if item != frame_id_min:
                    diff = item - frame_id_min
                    hist_ = int(diff / interval)
                    #print item, frame_id_min, diff, hist_
                    distribution[j][j][hist_] = distribution[j][j][
                        hist_] + spatial_temporal_count[i][j]

    smooth = 50
    for i in range(cam_num):
        for j in range(cam_num):
            #print("gauss "+str(i)+"->"+str(j))
            distribution[i][j][:] = gauss_smooth(distribution[i][j][:], smooth)

    sum_ = np.sum(distribution, axis=2)
    for i in range(cam_num):
        for j in range(cam_num):
            distribution[i][j][:] = distribution[i][j][:] / (sum_[i][j] + eps)

    return distribution  # [to][from], to xxx camera, from xxx camera


if __name__ == "__main__":
    data_dir = "/data/wuyiming/reid/msmt/MSMT17_V1/train/"
    imgs_path = glob.glob(os.path.join(data_dir, '*', '*.jpg'))
    matrix = get_st_matrix(imgs_path, pseudo_labels=None)

