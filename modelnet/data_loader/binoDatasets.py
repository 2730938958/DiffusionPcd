from torch.utils import data
import numpy as np
import open3d as o3d



class binoDataset(data.Dataset):
    def __init__(self):
        self.file_path = '../datasets/bunny/data'
        f=open("bino.txt","r")
        self.src_list = []
        self.ref_list = []
        for line in f:
            src_ele, ref_ele = line.split()
            self.src_list.append(src_ele)
            self.ref_list.append(ref_ele)
        # self.label_dict=eval(f.read())
        f.close()

    def __getitem__(self,index):
        # ref_id = list(self.label_dict.values())[index-1]
        # src_id = list(self.label_dict.keys())[index-1]
        src_id = self.src_list[index]
        ref_id = self.ref_list[index]

        src_path = self.file_path+'/'+str(src_id)
        ref_path = self.file_path+'/'+str(ref_id)
        src_pcd = o3d.io.read_point_cloud(src_path)
        ref_pcd = o3d.io.read_point_cloud(ref_path)
        src_arr = np.asarray(src_pcd.points)
        ref_arr = np.asarray(ref_pcd.points)
        return src_arr,ref_arr

    def __len__(self):
        return len(self.src_list)