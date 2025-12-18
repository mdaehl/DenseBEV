# ------------------------------------------------------------------------
# Copyright (c) DenseBEV 2025. All rights reserved.
# ------------------------------------------------------------------------

from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
import random
from mmdet3d.core import LiDARInstance3DBoxes
import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
import copy
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
import subprocess
import math
from ..core.prediction_to_waymo import Prediction2Waymo


@DATASETS.register_module()
class CustomWaymoDataset(Custom3DDataset):
    def __init__(
        self,
        ann_file,
        use_memory,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        split="training",
        valid_classes=["Car", "Pedestrian", "Cyclist"],
        queue_length=4,
        test_mode=False,
        seq_split_num=2
    ):
        self.load_interval = load_interval
        self.queue_length = queue_length
        self.split = split
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.data_infos = self.data_infos["data_list"]  # account for new format
        self.data_infos = self.data_infos[:: self.load_interval]
        if self.load_interval > 1:
            print_log(
                f"Sample size will be reduced to 1/{self.load_interval} of"
                " the original data sample",
                logger="current",
            )

        # rerun group flag after fixing data infos
        if not self.test_mode:
            self._set_group_flag()

        self.data_prefix = dict(
            pts="velodyne",
            CAM_FRONT="image_0",
            CAM_FRONT_LEFT="image_1",
            CAM_FRONT_RIGHT="image_2",
            CAM_SIDE_LEFT="image_3",
            CAM_SIDE_RIGHT="image_4",
        )

        self.label_mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        self.valid_labels = [
            self.label_mapping[class_name] for class_name in valid_classes
        ]

        if use_memory:
            self.seq_split_num = seq_split_num
            self._set_sequence_group_flag()

    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)
        while True:
            data = self.prepare_train_data(index)
            if data is None:
                index = self._rand_another(index)
                continue
            return data

    def prepare_train_data(self, index):
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            queue.append(example)
        return self.union2one(queue)

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego_pose = np.array(info["ego2global"])
        ego2global_translation = ego_pose[3, :3]
        ego2global_rotation = Quaternion(matrix=ego_pose[:3, :3])
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)

        input_dict = dict(
            sample_idx=info["sample_idx"],
            scene_token=info["context_name"],
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            timestamp=info["timestamp"] / 1e6,
        )

        # can bus stuff
        can_bus = np.zeros(18)
        can_bus[:3] = ego2global_translation
        can_bus[3:7] = ego2global_rotation.q
        patch_angle = quaternion_yaw(ego2global_rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict["can_bus"] = can_bus

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["images"].items():
                image_paths.append(
                    f"{self.data_root}/{self.split}/{self.data_prefix[cam_type]}/{cam_info['img_path']}"
                )

                intrinsic = np.array(cam_info["cam2img"])
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

                lidar2cam = np.array(cam_info["lidar2cam"])
                lidar2img_rt = viewpad @ lidar2cam
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]

        ann_info = info["cam_sync_instances"]
        if len(ann_info):
            bboxes_3d = np.array([ann["bbox_3d"] for ann in ann_info])
            bbox_labels_3d = np.array([ann["bbox_label_3d"] for ann in ann_info])

            # filter w.r.t. valid classes
            mask = np.isin(bbox_labels_3d, self.valid_labels)
            bboxes_3d = bboxes_3d[mask]
            bbox_labels_3d = bbox_labels_3d[mask]
        else:
            bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            bbox_labels_3d = np.zeros(0, dtype=np.int64)

        bboxes_3d = LiDARInstance3DBoxes(bboxes_3d)

        anns_results = dict(gt_bboxes_3d=bboxes_3d, gt_labels_3d=bbox_labels_3d)
        return anns_results

    def union2one(self, queue):
        imgs_list = [each["img"].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        # todo get general scene identifier
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"].data
            if metas_map[i]["scene_token"] != prev_scene_token:
                metas_map[i]["prev_bev_exists"] = False
                prev_scene_token = metas_map[i]["scene_token"]
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                # overwrite position & angle with change - first step change is zero
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
            else:
                metas_map[i]["prev_bev_exists"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                # overwrite position & angle with change since previous step
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def evaluate(
        self,
        results,
        metric=["LET_mAP"],
        logger=None,
        jsonfile_prefix=None,
        submission_prefix=None,
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        pkl_prefix = "results"
        waymo_results_final_path = f"{pkl_prefix}.bin"
        converter = Prediction2Waymo(results, self.data_infos, waymo_results_final_path)
        converter.convert()

        metric = ["LET_mAP"]

        if 'mAP' in metric:
            eval_str = '/deps/mmdetection3d/mmdet3d/core/evaluation/waymo_utils/' + \
                f'compute_detection_metrics_main {pkl_prefix}.bin ' + \
                f'{self.data_root}/cam_gt.bin'
            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts, logger=logger)

            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall/L1 mAP'] = \
                (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                    ap_dict['Cyclist/L1 mAP']) / 3
            ap_dict['Overall/L1 mAPH'] = \
                (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                    ap_dict['Cyclist/L1 mAPH']) / 3
            ap_dict['Overall/L2 mAP'] = \
                (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                    ap_dict['Cyclist/L2 mAP']) / 3
            ap_dict['Overall/L2 mAPH'] = \
                (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                    ap_dict['Cyclist/L2 mAPH']) / 3
            
        if "LET_mAP" in metric: 
            eval_str = '/deps/mmdetection3d/mmdet3d/core/evaluation/waymo_utils/' + \
                    f'compute_detection_let_metrics_main {pkl_prefix}.bin ' + \
                    f'{self.data_root}/cam_gt.bin',

            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')

            print_log(ret_texts, logger=logger)
            ap_dict = {
                'Vehicle mAPL': 0,
                'Vehicle mAP': 0,
                'Vehicle mAPH': 0,
                'Pedestrian mAPL': 0,
                'Pedestrian mAP': 0,
                'Pedestrian mAPH': 0,
                'Sign mAPL': 0,
                'Sign mAP': 0,
                'Sign mAPH': 0,
                'Cyclist mAPL': 0,
                'Cyclist mAP': 0,
                'Cyclist mAPH': 0,
                'Overall mAPL': 0,
                'Overall mAP': 0,
                'Overall mAPH': 0
            }
            mAPL_splits = ret_texts.split('mAPL ')
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 3) + 1
                if idx % 3 == 0:  # mAPL
                    ap_dict[key] = float(mAPL_splits[split_idx].split(']')[0])
                elif idx % 3 == 1:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall mAPL'] = \
                (ap_dict['Vehicle mAPL'] + ap_dict['Pedestrian mAPL'] +
                    ap_dict['Cyclist mAPL']) / 3
            ap_dict['Overall mAP'] = \
                (ap_dict['Vehicle mAP'] + ap_dict['Pedestrian mAP'] +
                    ap_dict['Cyclist mAP']) / 3
            ap_dict['Overall mAPH'] = \
                (ap_dict['Vehicle mAPH'] + ap_dict['Pedestrian mAPH'] +
                    ap_dict['Cyclist mAPH']) / 3
        return ap_dict

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        prev_scene_token = self.data_infos[0]["context_name"]
        for idx in range(len(self.data_infos)):
            if self.data_infos[idx]['context_name'] != prev_scene_token:
                prev_scene_token = self.data_infos[idx]['context_name']
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix