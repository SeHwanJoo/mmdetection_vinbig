model = dict(
    type='FasterRCNN',
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        groups=32,
        base_width=4),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
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
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=14,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100)))
classes = ('Aortic_enlargement', 'Atelectasis', 'Calcification',
           'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
           'Lung_Opacity', 'Nodule/Mass', 'Other_lesion', 'Pleural_effusion',
           'Pleural_thickening', 'Pneumothorax', 'Pulmonary_fibrosis')
albu_train_transforms = [
    dict(type='ShiftScaleRotate', scale_limit=0.15, rotate_limit=10, p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5),
    dict(
        type='Cutout',
        num_holes=8,
        max_h_size=16,
        max_w_size=16,
        fill_value=0,
        p=0.7)
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        classes=('Aortic_enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
                 'Lung_Opacity', 'Nodule/Mass', 'Other_lesion',
                 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax',
                 'Pulmonary_fibrosis'),
        ann_file='data/vinbigdata/annotations/train_annotations_0.json',
        img_prefix='data/vinbigdata/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(512, 512), (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        scale_limit=0.15,
                        rotate_limit=10,
                        p=0.5),
                    dict(type='RandomBrightnessContrast', p=0.5),
                    dict(
                        type='Cutout',
                        num_holes=8,
                        max_h_size=16,
                        max_w_size=16,
                        fill_value=0,
                        p=0.7)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('Aortic_enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
                 'Lung_Opacity', 'Nodule/Mass', 'Other_lesion',
                 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax',
                 'Pulmonary_fibrosis'),
        ann_file='data/vinbigdata/annotations/val_annotations_0.json',
        img_prefix='data/vinbigdata/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512), (1024, 1024)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('Aortic_enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
                 'Lung_Opacity', 'Nodule/Mass', 'Other_lesion',
                 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax',
                 'Pulmonary_fibrosis'),
        ann_file='data/vinbigdata/annotations/test_annotations.json',
        img_prefix='data/vinbigdata/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512), (1024, 1024)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.1),
    cyclic_times=4,
    step_ratio_up=0.05)
optimizer = dict(type='SGD', lr=0.0004, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/fasterRCNN_x101_32d'
gpu_ids = range(0, 2)
