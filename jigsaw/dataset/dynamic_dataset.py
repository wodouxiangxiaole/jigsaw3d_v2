import os
import random

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm

class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        category='',
        num_points=1000,
        min_num_part=2,
        max_num_part=20,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
    ):
        # store parameters
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree


        self.min_part_point = 60 # ensure that each piece has at least # points
        # list of fracture folder path
        self.data_list = self._read_data(data_fn)
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

        self.data_dicts = []        

        print("start processed data")

        for index in tqdm(range(len(self.data_list))):
            pcs, piece_id, nps, areas = self._get_pcs(self.data_list[index])
            num_parts = len(pcs)
            cur_pts, cur_quat, cur_trans, cur_pts_gt = [], [], [], []
            for i, (pc, n_p) in enumerate(zip(pcs, nps)):
                pc_gt = pc.copy()
                pc, gt_trans = self._recenter_pc(pc)
                pc, gt_quat = self._rotate_pc(pc)
                # pc_shuffle, pc_gt_shuffle = self._shuffle_pc(pc, pc_gt)

                cur_pts.append(pc)
                cur_quat.append(gt_quat)
                cur_trans.append(gt_trans)
                cur_pts_gt.append(pc_gt)

            cur_pts = np.concatenate(cur_pts).astype(np.float32)  # [N_sum, 3]
            cur_pts_gt = np.concatenate(cur_pts_gt).astype(np.float32)  # [N_sum, 3]
            cur_quat = self._pad_data(np.stack(cur_quat, axis=0), self.max_num_part).astype(np.float32)  # [P, 4]
            cur_trans = self._pad_data(np.stack(cur_trans, axis=0), self.max_num_part).astype(np.float32)  # [P, 3]
            n_pcs = self._pad_data(np.array(nps), self.max_num_part).astype(np.int64)  # [P]
            valids = np.zeros(self.max_num_part, dtype=np.float32)
            valids[:num_parts] = 1.0


            data_dict = {
                'part_pcs': cur_pts,
                'part_quat': cur_quat,
                'part_trans': cur_trans,
                "n_pcs": n_pcs,
                'part_pcs_gt': cur_pts_gt,
            }
            # valid part masks
            data_dict['part_valids'] = valids
            data_dict['mesh_file_path'] = self.data_list[index]
            data_dict['num_parts'] = num_parts

            # data_id
            data_dict['data_id'] = index

            self.data_dicts.append(data_dict)



    def _read_data(self, data_fn):
        """Filter out invalid number of parts."""
        with open(os.path.join(self.data_dir, data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]
        data_list = []
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.data_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            for frac in os.listdir(mesh_dir):
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                num_parts = len(os.listdir(os.path.join(self.data_dir, frac)))
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)
        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):
        """pc: [N, 3]"""
        if self.rot_range > 0.:
            rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        return pc

    def _pad_data(self, data, pad_size=None):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        if len(data.shape) > 1:
            pad_shape = (pad_size,) + tuple(data.shape[1:])
        else:
            pad_shape = (pad_size,)
        pad_data = np.zeros(pad_shape, dtype=data.dtype)
        pad_data[: data.shape[0]] = data
        return pad_data
    

    @staticmethod
    def sample_points_by_areas(areas, num_points):
        """areas: [P], num_points: N"""
        total_area = np.sum(areas)
        nps = np.ceil(areas * num_points / total_area).astype(np.int32)
        nps[np.argmax(nps)] -= np.sum(nps) - num_points
        return np.array(nps, dtype=np.int64)

    def sample_reweighted_points_by_areas(self, areas):
        """ Sample points by areas, but ensures that each part has at least # points.
        areas: [P]
        """
        nps = self.sample_points_by_areas(areas, self.num_points)
        if self.min_part_point <= 1:
            return nps
        delta = 0
        for i in range(len(nps)):
            if nps[i] < self.min_part_point:
                delta += self.min_part_point - nps[i]
                nps[i] = self.min_part_point
        while delta > 0:
            k = np.argmax(nps)
            if nps[k] - delta >= self.min_part_point:
                nps[k] -= delta
                delta = 0
            else:
                delta -= nps[k] - self.min_part_point
                nps[k] = self.min_part_point
        # simply take points from the largest parts
        # This implementation is not very elegant, could improve by resample by areas.
        return np.array(nps, dtype=np.int64)
    
    
    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `piece`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0/piece_0.obj
        data_folder = os.path.join(self.data_dir, data_folder)
        mesh_files = os.listdir(data_folder)
        mesh_files.sort()
        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
            raise ValueError

        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file), force="mesh")
            for mesh_file in mesh_files
        ]
        areas = [mesh.area for mesh in meshes]
        areas = np.array(areas)
        pcs, piece_id, nps = [], [], []
        # if self.sample_by == "area":
        nps = self.sample_reweighted_points_by_areas(areas)
     

        for i, (mesh) in enumerate(meshes):
            num_points = nps[i]
            samples, fid = mesh.sample(num_points, return_index=True)
            pcs.append(samples)
            piece_id.append([i] * num_points)

        piece_id = np.concatenate(piece_id).astype(np.int64).reshape((-1, 1))
        return pcs, piece_id, nps, areas

    def __getitem__(self, index):
        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_trans': MAX_NUM x 3
                Translation vector

            'part_quat': MAX_NUM x 4
                Rotation as quaternion.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'instance_label': MAX_NUM x 0, useless

            'part_label': MAX_NUM x 0, useless

            'part_ids': MAX_NUM, useless

            'data_id': int
                ID of the data.

        }
        """

        return self.data_dicts[index]

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):
    data_dict = dict(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,
        category=cfg.data.category,
        num_points=cfg.data.num_pc_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
    )
    train_set = GeometryPartDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = GeometryPartDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return train_loader, val_loader
