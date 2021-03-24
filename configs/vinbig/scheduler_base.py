lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-1),
    cyclic_times=4,
    step_ratio_up=0.05
)

optimizer = dict(type='SGD', lr=4e-4, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(type='EpochBasedRunner', max_epochs=40)