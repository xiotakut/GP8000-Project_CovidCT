checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'
conv_kernel_size = 1
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'Covid/'
dataset_type = 'XinyuDataset_Covid'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2500,
        max_keep_ckpts=2,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=768,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        kernel_generate_head=dict(
            align_corners=False,
            channels=512,
            dropout_ratio=0.1,
            in_channels=[
                192,
                384,
                768,
                1536,
            ],
            in_index=[
                0,
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            pool_scales=(
                1,
                2,
                3,
                6,
            ),
            type='UPerHead'),
        kernel_update_head=[
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=150,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=150,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=150,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
        ],
        num_stages=3,
        type='IterativeDecodeHead'),
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
num_stages = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=80000,
        milestones=[
            60000,
            72000,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='Covid/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='XinyuDataset_Covid'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='Covid/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='XinyuDataset_Covid'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='Covid/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='XinyuDataset_Covid'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='WandbVisBackend'),
    ])
work_dir = './work_dirs/XinyuDataset-KNet_Covid'
