import numpy as np
import torch
from scipy.spatial import distance
import ot


def array2samples_distance(arr1, arr2):
    '''
    CD distance of single point cloud
    arr1: tensor, (N, 3)
    arr2: tensor, (N, 3)
    '''
    N1, C1 = arr1.shape
    N2, C2 = arr2.shape
    device = arr1.device

    exp_arr1 = arr1.repeat(N2, 1)   # (N*N, 1)
    exp_arr1 = exp_arr1.to(device)
    exp_arr2 = torch.reshape(torch.unsqueeze(arr2, 1).repeat(1, N1, 1), (-1, C2))
    exp_arr2 = exp_arr2.to(device)

    distances = (exp_arr1 - exp_arr2) * (exp_arr1 - exp_arr2)
    distances = torch.sum(distances, dim=1)
    distances = torch.reshape(distances, (N2, N1))
    distances = torch.min(distances, dim=1)[0]
    avg_distances = torch.mean(distances)
    return distances, avg_distances


def metric_chamfer_distance(arr1, arr2):
    '''
    arr1: tensor, (B, N, 3)
    arr2: tensor, (B, N, 3)
    '''
    B, N, C = arr1.shape
    dist = 0
    for b in range(B):
        _, avg_dist1 = array2samples_distance(arr1[b], arr2[b])
        _, avg_dist2 = array2samples_distance(arr2[b], arr1[b])
        dist = dist + (0.5 * avg_dist1 + 0.5 * avg_dist2) / B
    return float(dist)


def metric_earth_mover_distance(arr1, arr2):
    '''
    arr1: tensor, (B, N, 3)
    arr2: tensor, (B, N, 3)
    '''
    B, N, C = arr1.shape
    arr1 = arr1.detach().cpu().numpy()
    arr2 = arr2.detach().cpu().numpy()
    dist = 0
    for b in range(B):
        point_cloud1, point_cloud2 = arr1[b], arr2[b]
        # 计算点云之间的距离矩阵
        distance_matrix = distance.cdist(point_cloud1, point_cloud2, metric='euclidean')

        # 将点云转换为概率分布
        point_cloud1_distribution = np.ones(len(point_cloud1)) / len(point_cloud1)
        point_cloud2_distribution = np.ones(len(point_cloud2)) / len(point_cloud2)

        # 使用 POT 库计算点云之间的 EMD 距离
        emd_dist = ot.emd2(point_cloud1_distribution, point_cloud2_distribution, distance_matrix)

        dist = dist + emd_dist / B
    return float(dist)


def metric_f_score(arr1, arr2, threshold=0.01):
    '''
    arr1: tensor, (B, N, 3)
    arr2: tensor, (B, N, 3)
    '''
    B, N, C = arr1.shape
    dist1_list = []
    dist2_list = []
    for b in range(B):
        dist1, _ = array2samples_distance(arr1[b], arr2[b])
        dist2, _ = array2samples_distance(arr2[b], arr1[b])
        dist1_list.append(dist1.unsqueeze(0))
        dist2_list.append(dist2.unsqueeze(0))
    dist1_batch = torch.cat(dist1_list, dim=1)
    dist2_batch = torch.cat(dist2_list, dim=1)
    precision_1 = torch.mean((dist1_batch < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2_batch < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return float(fscore), float(precision_1), float(precision_2)


def calulate_metrics(pred, gt):
    '''
    pred: tensor, (B, N, 3)
    gt: tensor, (B, M, 3)
    N和M可以不相同
    '''
    metric_CD = metric_chamfer_distance(pred, gt)
    metric_EMD = metric_earth_mover_distance(pred, gt)
    metric_F_Score, _, _ = metric_f_score(pred, gt)

    metrics = {
        'metric_CD': metric_CD,
        'metric_EMD': metric_EMD,
        'metric_F_Score': metric_F_Score,
    }
    return metrics

