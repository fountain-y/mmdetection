_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_cell_voc.py'

model = dict(
    rpn_head=dict(
        type='RankBasedRPNHead',
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='GIoULoss', reduction='none'),
        head_weight=0.20),
    roi_head=dict(
        type='RankBasedStandardRoIHead',
        bbox_head=dict(
            # num_classes=40,
            num_classes=21,
            type='RankBasedShared2FCBBoxHead',
            reg_decoded_bbox= True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_bbox=dict(type='GIoULoss', reduction='none'),
            loss_cls=dict(use_sigmoid=True))),
    
    # Model training and testing settings
    train_cfg = dict(
        rpn=dict(sampler=dict(type='PseudoSampler')),
        # R-CNN has a bug for PseudoSampler, so we unlimit the number of examples
        # to be sampler as a workaround
        rcnn=dict(sampler=dict(num=1e10))),

    # We increase the score_threshold to 0.40
    # for more efficient inference
    # This can decrease the performance by 0.01 AP.
    # test_cfg = dict(rcnn=dict(score_thr=0.40))
    test_cfg = dict(
                rcnn=dict(score_thr=0.4),
                nms=dict(class_agnostic='True'))
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
total_epochs = 30