_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]
classes = ("Aortic_enlargement", "Atelectasis",
           "Calcification", "Cardiomegaly",
           "Consolidation", "ILD", "Infiltration",
           "Lung_Opacity", "Nodule/Mass", "Other_lesion",
           "Pleural_effusion", "Pleural_thickening",
           "Pneumothorax", "Pulmonary_fibrosis")

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=14
        )
    )
)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0,
        scale_limit=0.15,
        rotate_limit=10,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
]

data = dict(
    # batch size
    samples_per_gpu=2,
    # num worker
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='../competition/input/vinbigdata-coco-dataset/annotations/train_annotations_0.json',
        img_prefix='../competition/input/vinbigdata-coco-dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(512, 512)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Albu',
                 transforms=albu_train_transforms,
                 bbox_params=dict(
                     type='BboxParams',
                     format='pascal_voc',
                     label_fields=['gt_labels'],
                     min_visibility=0.0,
                     filter_lost_elements=True),
                 keymap={
                     'img': 'image',
                     'gt_masks': 'masks',
                     'gt_bboxes': 'bboxes'
                 },
                 update_pad_shape=False,
                 skip_img_without_anno=True),
            dict(
                type='Normalize',
                # xray??? ????????????
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='../competition/input/vinbigdata-coco-dataset/annotations/val_annotations_0.json',
        img_prefix='../competition/input/vinbigdata-coco-dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
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
        classes=classes,
        ann_file='../competition/input/vinbigdata-coco-dataset/test_annotations.json',
        img_prefix='../competition/input/vinbigdata-coco-dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
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
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(type='EpochBasedRunner', max_epochs=25)
