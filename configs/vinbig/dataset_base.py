classes = ("Aortic_enlargement", "Atelectasis",
           "Calcification", "Cardiomegaly",
           "Consolidation", "ILD", "Infiltration",
           "Lung_Opacity", "Nodule/Mass", "Other_lesion",
           "Pleural_effusion", "Pleural_thickening",
           "Pneumothorax", "Pulmonary_fibrosis")

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        scale_limit=0.15,
        rotate_limit=10,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        p=0.5),
    dict(type="Cutout", num_holes=8,
         max_h_size=16, max_w_size=16, fill_value=0, p=0.7)
]

data = dict(
    # batch size
    samples_per_gpu=4,
    # num worker
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='../competition/input/vinbigdata-coco-dataset/annotations/train_annotations_nofind.json',
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
                # xray로 바꾸면됨
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
        ]
    )
)
