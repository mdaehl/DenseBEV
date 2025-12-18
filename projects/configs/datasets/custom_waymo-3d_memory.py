# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = "CustomWaymoDataset"
data_root = "data/waymo/kitti_format/"
file_client_args = dict(backend="disk")


img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = ["Car", "Pedestrian", "Cyclist"]
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=False, use_camera=True)


train_pipeline = [
    dict(type="LoadDifferentShapeMultiViewImageFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CustomCollect3D", keys=["gt_bboxes_3d", "gt_labels_3d", "img"]),
]


test_pipeline = [
    dict(type="LoadDifferentShapeMultiViewImageFromFiles", to_float32=True),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1920, 1280),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(type="CustomCollect3D", keys=["img"]),
        ],
    ),
]


# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
num_gpus = 8
batch_size = 2
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "waymo_infos_train.pkl",
            split="training",
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            filter_empty_gt=False,
            use_memory=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
            # load one frame every five frames
            queue_length=4,
            load_interval=5,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "waymo_infos_val.pkl",
        split="training",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        use_memory=False,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "waymo_infos_val.pkl",
        split="training",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        use_memory=False,
        box_type_3d="LiDAR",
    ),
    shuffler_sampler=dict(type="InfiniteGroupEachSampleInBatchSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

num_iters_per_epoch = 31617 // (num_gpus * batch_size)  # 158081 / 5 = 31617 (load interval)

evaluation = dict(interval=24 * num_iters_per_epoch, pipeline=test_pipeline)
