import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig
from coco import COCOMutiBranch  # noqa

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/path/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            PSEUDO_SCORE_THRES=0.5,
            MATCHING_IOU_THRES=0.4,
            WITH_TEACHER=True
        ),
    ),
    DATASETS=dict(
        CO_MINING=True,
        TRAIN=("coco_missing_50p",),
        TEST=("coco_2017_val",),
    ),
   
    TRAINER=dict(
        NAME="MultiBranchRunner",
        FP16=dict(
            ENABLED=False,
            # options: [APEX, PyTorch]
            TYPE="APEX",
            # OPTS: kwargs for each option
            OPTS=dict(
                OPT_LEVEL="O1",
            ),
        ),
        WITH_TEACHER=True
    ),
    
    SOLVER=dict(
            CHECKPOINT_PERIOD=120000,
            LR_SCHEDULER=dict(
                MAX_ITER=1440000,
                STEPS=(960000, 1280000),
            ),
            OPTIMIZER=dict(
                BASE_LR=0.01, # learning rate in original config is used for 8 GPUs 16 total batch;
            ),
            IMS_PER_DEVICE=8,
    ),
    TEST=dict(
        EVAL_PERIOD=120000,
        DETECTIONS_PER_IMAGE=100,
    ),
    DATALOADER=dict(
        # Number of data loading threads
        NUM_WORKERS=8),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                
                [ dict(type = "OrderList", 
                            transforms = [dict(type ="RandResizeShortestEdge",
                                        short_edge_length=(800,), max_size=1333, sample_style="choice"),
                                        ],
                            record = True)
                ],

                [ dict(type = "OrderList", transforms = 
                                                [dict(type ="RandResizeShortestEdge",
                                                     short_edge_length=(800,), max_size=1333, sample_style="choice"),
                                                 dict(type ="RandFlip",prob=0.5),
                                                 dict(type="ColorJiter")],
                                                 record = True
                                                 )
                ],


                [ dict(type = "OrderList", transforms = 
                                                [dict(type ="RandResizeShortestEdge",
                                                     short_edge_length=(800,), max_size=1333, sample_style="choice"),
                                                 dict(type ="RandFlip",prob=0.5),
                                                 dict(
                                                    type="ShuffleList",
                                                    transforms=[
                                                        dict(
                                                            type="OneOf",
                                                            transforms=[
                                                                dict(type = "RandAutoAugment",name=k,magnitude=10,)
                                                                        for k in [
                                                                            "Identity",
                                                                            "AutoContrast",
                                                                            "Equalize",
                                                                            "Solarize",
                                                                            "Color",
                                                                            "Contrast",
                                                                            "Brightness",
                                                                            "Sharpness",
                                                                            "PosterizeResearch",
                                                                        ]
                                                            ],
                                                        ),
                                                        dict(
                                                            type="OneOf",
                                                            transforms=[
                                                                dict(type="RandTranslate", x=(-0.1, 0.1)),
                                                                dict(type="RandShear", x=(-30, 30)),
                                                                dict(type="RandShear", y=(-30, 30)),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                    dict(
                                                        type="RandErase",
                                                        n_iterations=(1, 5),
                                                        size=[0, 0.2],
                                                        squared=True,
                                                    ),
                                                 ],
                                                 record = True
                                )
                ],
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
