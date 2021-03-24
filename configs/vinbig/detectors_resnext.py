_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    './dataset_base.py',
    './scheduler_base.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=14
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=14
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=14
            )
        ]
    ),
    test_cfg=dict(
        rpn=dict(
            nms_thr=0.7
        ),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.0)
        )
    )
)
