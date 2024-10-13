import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def plot_frame(sample, mod, save_dir=None, name=None, show=True):
    if mod in ['rgb', 'depth']:
        sample = sample * 1000
        image = sample.astype(np.uint8)
        plot_image(image, save_dir, name, show)
    elif mod in ['seg']:
        sample = sample.cpu().numpy()
        image_uint8 = (sample * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_uint8)
        image_pil.save(os.path.join(save_dir, name+'.png'))
    elif mod in ['lidar']:
        point_cloud = sample
        plot_point_cloud(point_cloud, save_dir, name, show)
    else:
        raise ValueError('Unsupport for this modality')
    pass


def plot_image(im, save_dir=None, name=None, show=True):
    if len(im.shape) == 3:
        # convert BGR to RGB
        im = im[:, :, ::-1]
    plt.imshow(im)

    # save the plot
    if save_dir is not None:
        if name is None:
            name = 'image'
        else:
            name = '{}_image'.format(name)
        plt.savefig(os.path.join(save_dir, name+'.png'))
    if show:
        plt.show()
    plt.close()


def plot_point_cloud(pc, save_dir=None, name=None, show=True):
    '''
    pc: [N, 3]
    '''
    # 创建3D坐标轴
    ax = plt.axes(projection='3d')
    pc = pc[:, [2, 0, 1]]
    # 绘制点云
    sizes = np.ones((pc.shape[0])) * 2
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], s=sizes, cmap='Greens')

    # 添加标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    # 设置视角
    # 设置方位角为135度，俯仰角为30度
    # ax.view_init(azim=135, elev=30)
    ax.view_init(azim=45, elev=30)

    # 设置轴范围
    # ax.set_xlim([0, 6.4])  # 设置 x 轴的范围
    # ax.set_ylim([-3.2, 3.2])  # 设置 y 轴的范围
    # ax.set_zlim([-2, 2])  # 设置 z 轴的范围

    ax.set_xlim([-3, 3])  # 设置 x 轴的范围
    ax.set_ylim([-3, 3])  # 设置 y 轴的范围
    ax.set_zlim([-3, 3])  # 设置 z 轴的范围

    # save the plot
    if save_dir is not None:
        if name is None:
            name = 'point_cloud'
        else:
            name = '{}_point_cloud'.format(name)
        plt.savefig(os.path.join(save_dir, name+'.png'))

    # 显示图形
    if show:
        plt.show()
    plt.close()

def plot_voxel(vl, save_dir=None, name=None):
    # 创建一个 Dx*Dy*Dz 的体素网格
    Dx, Dy, Dz = vl.shape
    voxels = np.zeros((Dx, Dy, Dz), dtype=bool)

    # 将有点云的体素进行可视化
    voxels[vl > 0] = True

    # 绘制体素数据
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, edgecolor='k')

    # 添加标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel')

    # 设置坐标轴范围
    ax.set_xlim(0, Dx)
    ax.set_ylim(0, Dy)
    ax.set_zlim(0, Dz)

    # 设置视角
    # 设置方位角为135度，俯仰角为30度
    ax.view_init(azim=135, elev=30)

    # save the plot
    if save_dir is not None:
        if name is None:
            name = 'voxel'
        else:
            name = '{}_voxel'.format(name)
        plt.savefig(os.path.join(save_dir, name+'.png'))

    # 显示图形
    plt.show()
    plt.close()

def plot_bev(bev, save_dir=None, name=None):
    plt.imshow(bev, cmap='jet')

    # save the plot
    if save_dir is not None:
        if name is None:
            name = 'nev'
        else:
            name = '{}_bev'.format(name)
        plt.savefig(os.path.join(save_dir, name+'.png'))

    plt.show()
    plt.close()


# if __name__ == '__main__':
#     plot_voxel()
