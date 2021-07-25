import matplotlib.pyplot as plt
import numpy as np

def draw(cam_i, cam_j, y):
    x = [i for i in range(len(y))]
    ax = plt.subplot(111)
    ax.plot(x, y,)# label=f'cam {cam_i + 1} to cam {cam_j + 1}')
    return ax

def main():
    file_path = "/home/wuyiming/tianjian/fast-reid/train_distribution_gt.npy"
    distribution = np.load(file_path) # (cam, cam, 3000)
    cam_num, _, h = distribution.shape

    plt.figure(facecolor='grey',edgecolor='white')
    # draw(0, 1, distribution[0][1])
    for i in range(cam_num):
        for j in range(cam_num):
    # for i in [2]:
    #     for j in [3]:
            if i == j: continue
            print(f'cam {i + 1} to cam {j + 1} starts')
            ax = draw(i, j, distribution[i][j])

            plt.xlabel('Time Interval')
            plt.ylabel('Frequceny')
            ax.grid(c='w')
            # plt.xticks([1000 * i for i in range(4)])
            plt.xlim(0, 1000)
            plt.ylim(0, 0.01)
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            # ax.yaxis.set_ticks_position('left')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # ax.xaxis.set_ticks_position('bottom')
            # plt.legend()
            plt.savefig('pdf/cam_{}2cam_{}.pdf'.format(i + 1, j + 1))
            plt.cla()
            

if __name__ == "__main__":
    main()