_base_ = './retinanet_r50_fpn_1x_cell_voc.py'
model = dict(
    bbox_head=dict(num_classes=40)
)
# learning policy
# lr_config = dict(step=[16, 22])
lr_config = dict(policy='step', step=[2])
runner = dict(type='EpochBasedRunner', max_epochs=24)
