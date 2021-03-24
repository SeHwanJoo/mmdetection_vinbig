_base_ = [
    './faster_base.py'
]
model = dict(
    pretrained='open-mmlab://regnetx_12gf',
    backbone=dict(
        _delete_=True,
        type='RegNet',
        arch='regnetx_12gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        in_channels=[224, 448, 896, 2240]
    )
)
