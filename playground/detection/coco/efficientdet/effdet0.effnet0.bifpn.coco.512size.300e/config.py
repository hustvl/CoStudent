import os.path as osp

from cvpods.configs.efficientdet_config import EfficientDetConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="cvpods/ImageNetPretrained/GOOGLE/EfficientNet/efficientnet-b0.pth",
        EFFICIENTNET=dict(
            MODEL_NAME="efficientnet-b0",
            NORM="BN",
            DROP_CONNECT_RATE=0.0,  # survival_prob = 1.0
        ),
        BIFPN=dict(
            INPUT_SIZE=512,
            NUM_LAYERS=3,
            OUT_CHANNELS=64,
            NORM="BN",
        ),
        EFFICIENTDET=dict(
            HEAD=dict(
                NUM_CONV=3,
                NORM="BN",
            )
        )
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_ITER=int(300 * 120000 / 64),
            WARMUP_FACTOR=0.008 / 0.16,
            WARMUP_ITERS=int(120000 / 64),
            WARMUP_METHOD="linear",
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.08,
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=4e-5,
            WEIGHT_DECAY_NORM=0,
            WEIGHT_DECAY_BIAS=4e-5,
            MOMENTUM=0.9,
        ),
        CHECKPOINT_PERIOD=20000,
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=8,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RandomFlip", dict(prob=0.5)),
                ("RandomScale", dict(output_size=(512, 512), ratio_range=(0.1, 2.0))),
                ("RandomCrop", dict(crop_type="absolute", crop_size=(512, 512), strict_mode=False)),
                ("Pad", dict(
                    top=0, left=0, target_h=512, target_w=512,
                    pad_value=[v * 255. for v in [0.485, 0.456, 0.406]])),
            ],
            TEST_PIPELINES=[
                ("ResizeLongestEdge", dict(
                    long_edge_length=(512,), sample_style="choice")),
            ],
        ),
        FORMAT="RGB",
    ),
    TEST=dict(
        EVAL_PERIOD=20000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomEfficientDetConfig(EfficientDetConfig):
    def __init__(self):
        super(CustomEfficientDetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomEfficientDetConfig()
