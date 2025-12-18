# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

from mmcv.utils import print_log
import mmcv
from typing import List
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2
import numpy as np


class Prediction2Waymo:
    def __init__(
        self,
        results: List[dict],
        data_infos: List[dict],
        waymo_results_final_path: str,
        parallel: bool = True,
        num_workers: int = 12,
    ):
        self.results = results
        self.waymo_results_final_path = waymo_results_final_path
        self.parallel = parallel
        self.num_workers = num_workers

        self.classes = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

        self.k2w_cls_map = {
            "Car": label_pb2.Label.TYPE_VEHICLE,
            "Pedestrian": label_pb2.Label.TYPE_PEDESTRIAN,
            "Sign": label_pb2.Label.TYPE_SIGN,
            "Cyclist": label_pb2.Label.TYPE_CYCLIST,
        }

        self.context_names = [item["context_name"] for item in data_infos]
        self.timestamps = [item["timestamp"] for item in data_infos]

    def convert(self):
        print_log("Start converting ...", logger="current")

        # if self.parallel:
        #    objects_list = mmcv.track_parallel_progress(self.convert_one, range(len(self)), nproc=self.num_workers)
        # else:
        objects_list = mmcv.track_progress(self.convert_one, range(len(self)))

        combined = metrics_pb2.Objects()
        for objects in objects_list:
            for o in objects.objects:
                combined.objects.append(o)

        with open(self.waymo_results_final_path, "wb") as f:
            f.write(combined.SerializeToString())

    def convert_one(self, res_idx: int):
        if len(self.results[res_idx]["pts_bbox"]["labels_3d"]):
            objects = self.parse_objects_from_origin(
                self.results[res_idx],
                self.context_names[res_idx],
                self.timestamps[res_idx],
            )
        else:
            print(res_idx, "not found.")
            objects = metrics_pb2.Objects()

        return objects

    def parse_objects_from_origin(self, result: dict, contextname: str, timestamp: str):
        lidar_boxes = result["pts_bbox"]["boxes_3d"]
        scores = result["pts_bbox"]["scores_3d"]
        labels = result["pts_bbox"]["labels_3d"]

        objects = metrics_pb2.Objects()
        for lidar_box, score, label in zip(lidar_boxes, scores, labels):
            lidar_box = np.array(lidar_box)
            score = float(score)
            label = int(label)
            # Parse one object
            box = label_pb2.Label.Box()
            height = lidar_box[5]
            heading = lidar_box[6]

            box.center_x = lidar_box[0]
            box.center_y = lidar_box[1]
            box.center_z = (
                lidar_box[2] + height / 2
            )  # converts from KITTI format to box center
            box.length = lidar_box[3]
            box.width = lidar_box[4]
            box.height = height
            box.heading = heading

            object = metrics_pb2.Object()
            object.object.box.CopyFrom(box)

            class_name = self.classes[label]
            object.object.type = self.k2w_cls_map[class_name]
            object.score = score
            object.context_name = contextname
            object.frame_timestamp_micros = timestamp
            objects.objects.append(object)

        return objects

    def __len__(self):
        return len(self.results)
