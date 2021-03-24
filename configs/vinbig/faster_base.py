_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './dataset_base.py',
    './scheduler_base.py',
    '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1, 2, 4, 8, 16],
            ratios=[0.33, 0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=14
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms=dict(type='nms', iou_threshold=0.7),
        ),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100
        )
    )
)