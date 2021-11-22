_base_ = ['./config2.py']

loss_bbox=dict(
    _delete_=True,
    type='IoULoss',
    eps=1e-6,
    loss_weight=1.0,
    reduction='none')