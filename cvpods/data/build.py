#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
"""
This file contains the default logic to build a dataloader for training or testing.
"""
from loguru import logger

import numpy as np

import torch.utils.data

from cvpods.utils import comm, seed_all_rng

from .detection_utils import check_sample_valid
from .registry import DATASETS, PATH_ROUTES, SAMPLERS, TRANSFORMS
from .samplers import Infinite

__all__ = [
    "build_dataset",
    "build_train_loader",
    "build_train_loader_mutiBranch",
    "build_test_loader",
    "build_transform_gens",
    "build_mutiBranch_transform_gens",
    "build_test_loader_mutiBranch"
]

def build_transform_gens(pipelines):
    """
    Create a list of :class:`TransformGen` from config.

    Transform list is a list of tuple which includes Transform name and parameters.
    Args:
        pipelines: cfg.INPUT.TRAIN_PIPELINES and cfg.INPUT.TEST_PIPELINES are used here

    Returns:
        list[TransformGen]: a list of several TransformGen.
    """

    def build(pipeline):
        tfm_gens = []

        for (aug, args) in pipeline:
            if "List" in aug:
                assert "transforms" in args, "List Transforms must contain a `transforms` key"
                sub_pipelines = args["transforms"]
                args["transforms"] = build_transform_gens(sub_pipelines)
                tfm = TRANSFORMS.get(aug)(**args)
            else:
                if aug == "ResizeShortestEdge":
                    check_sample_valid(args)
                if aug.startswith("Torch_"):
                    tfm = TRANSFORMS.get("TorchTransformGen")(args)
                else:
                    tfm = TRANSFORMS.get(aug)(**args)
            tfm_gens.append(tfm)

        return tfm_gens

    if isinstance(pipelines, dict):
        tfm_gens_dict = {}
        for key, tfms in pipelines.items():
            tfm_gens_dict[key] = build(tfms)
        return tfm_gens_dict
    else:
        return build(pipelines)

def build_mutiBranch_transform_gens(pipelines):
    """
    Create a list of :class:`TransformGen` from config.

    Transform list is a list of tuple which includes Transform name and parameters.
    Args:
        pipelines: cfg.INPUT.TRAIN_PIPELINES and cfg.INPUT.TEST_PIPELINES are used here

    Returns:
        list[TransformGen]: a list of several TransformGen.
    """

    def build(pipeline):
        tfm_gens = []
        if isinstance(pipeline,list):
            for i in range(len(pipeline)):
                tfm_gens.append(build(pipeline[i]))

        elif isinstance(pipeline,dict):
            aug = pipeline.pop('type')
            if "List" in aug or aug == "OneOf":
                assert "transforms" in pipeline, "List Transforms must contain a `transforms` key"
                sub_pipelines = pipeline["transforms"]
                pipeline["transforms"] = build(sub_pipelines)
                tfm = TRANSFORMS.get(aug)(**pipeline)
                tfm_gens = tfm
            else:
                if aug == "ResizeShortestEdge" or aug == "RandResizeShortestEdge":
                    check_sample_valid(pipeline)
                if aug.startswith("Torch_"):
                    tfm = TRANSFORMS.get("TorchTransformGen")(pipeline)
                else:
                    tfm = TRANSFORMS.get(aug)(**pipeline)
                tfm_gens = tfm
        return tfm_gens

    if isinstance(pipelines, list): #list[list[TransformGen]]
        # assert len(pipelines) == 3
        return [build(pipeline[0]) for pipeline in pipelines]
    elif isinstance(pipelines, dict):
        return build(pipelines)


def build_dataset(config, dataset_names, transforms=[], is_train=True):
    """
    dataset_names: List[str], in which elemements must be in format of "dataset_task_version"
    """

    def _build_single_dataset(config, dataset_name, transforms=[], is_train=True):
        """
        Build a single dataset according to dataset_name.

        Args:
            config (BaseConfig): config.
            dataset_name (str): dataset_name should be of 'dataset_xxx_xxx' format,
                so that corresponding dataset can be acquired from the first token in this argument.
            transforms (List[TransformGen]): list of transforms configured in config file.
            is_train (bool): whether is in training mode or not.
        """
        dataset_type = dataset_name.split("_")[0].upper()
        assert dataset_type in PATH_ROUTES, "{} not found in PATH_ROUTES".format(dataset_type)
        name = PATH_ROUTES.get(dataset_type)["dataset_type"]
        dataset = DATASETS.get(name)(config, dataset_name, transforms=transforms, is_train=is_train)
        return dataset

    datasets = [
        _build_single_dataset(config, dataset_name, transforms=transforms, is_train=is_train)
        for dataset_name in dataset_names
    ]
    custom_type, args = config.DATASETS.CUSTOM_TYPE
    # wrap all datasets, Dataset concat is the default behaviour
    dataset = DATASETS.get(custom_type)(datasets, **args)
    return dataset


def build_train_loader(cfg):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    # For simulate large batch training
    num_devices = comm.get_world_size()
    rank = comm.get_rank()

    # use subdivision batchsize
    assert cfg.SOLVER.BATCH_SUBDIVISIONS == 1
    images_per_minibatch = cfg.SOLVER.IMS_PER_DEVICE // cfg.SOLVER.BATCH_SUBDIVISIONS

    transform_gens = build_transform_gens(cfg.INPUT.AUG.TRAIN_PIPELINES)

    logger.info(f"TransformGens used: {transform_gens} in training")
    dataset = build_dataset(
        cfg, cfg.DATASETS.TRAIN, transforms=transform_gens, is_train=True
    )

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using training sampler {}".format(sampler_name))

    assert sampler_name in SAMPLERS, "{} not found in SAMPLERS".format(sampler_name)
    if sampler_name == "TrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, cfg.DATALOADER.REPEAT_THRESHOLD)
    elif sampler_name == "DistributedGroupSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, images_per_minibatch, num_devices, rank)

    if cfg.DATALOADER.ENABLE_INF_SAMPLER:
        sampler = Infinite(sampler)
        logger.info("Wrap sampler with infinite warpper...")
    # from pdb import set_trace
    # set_trace()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_minibatch,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        persistent_workers=True
    )

    return data_loader

def build_train_loader_mutiBranch(cfg):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    # For simulate large batch training
    num_devices = comm.get_world_size()
    rank = comm.get_rank()

    # use subdivision batchsize
    images_per_minibatch = cfg.SOLVER.IMS_PER_DEVICE // cfg.SOLVER.BATCH_SUBDIVISIONS

    transform_gens = build_mutiBranch_transform_gens(cfg.INPUT.AUG.TRAIN_PIPELINES)
    if isinstance(cfg.INPUT.AUG.TRAIN_PIPELINES,list):
        if len(cfg.INPUT.AUG.TRAIN_PIPELINES) == 3:
            branchs = ['RAW',"NORMAL_AUG","STRONG_AUG"]
        elif len(cfg.INPUT.AUG.TRAIN_PIPELINES) == 2:
            branchs = ['RAW_TEACHER',"NORMAL_STUDENT"]
        elif len(cfg.INPUT.AUG.TRAIN_PIPELINES) == 1:
            branchs = ['RAW']

        for i in range(len(cfg.INPUT.AUG.TRAIN_PIPELINES)):
            branch = branchs[i]
            trg = transform_gens[i]
            logger.info(f"{branch}TransformGens used: {trg} in training")
    else:
        logger.info(f"TransformGens used: {transform_gens} in training")

    dataset = build_dataset(
        cfg, cfg.DATASETS.TRAIN, transforms=transform_gens, is_train=True
    )

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using training sampler {}".format(sampler_name))

    assert sampler_name in SAMPLERS, "{} not found in SAMPLERS".format(sampler_name)
    if sampler_name == "TrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, cfg.DATALOADER.REPEAT_THRESHOLD)
    elif sampler_name == "DistributedGroupSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, images_per_minibatch, num_devices, rank)

    if cfg.DATALOADER.ENABLE_INF_SAMPLER:
        sampler = Infinite(sampler)
        logger.info("Wrap sampler with infinite warpper...")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_minibatch,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        persistent_workers=True
    )

    return data_loader

def build_test_loader_mutiBranch(cfg):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:

       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """

    transform_gens = build_mutiBranch_transform_gens(cfg.INPUT.AUG.TEST_PIPELINES)
    if isinstance(cfg.INPUT.AUG.TEST_PIPELINES,list):
        if len(cfg.INPUT.AUG.TEST_PIPELINES) == 3:
            branchs = ['RAW',"NORMAL_AUG","STRONG_AUG"]
        elif len(cfg.INPUT.AUG.TEST_PIPELINES) == 2:
            branchs = ['RAW_TEACHER',"NORMAL_STUDENT"]
        elif len(cfg.INPUT.AUG.TEST_PIPELINES) == 1:
            branchs = ['RAW']

        for i in range(len(cfg.INPUT.AUG.TEST_PIPELINES)):
            branch = branchs[i]
            trg = transform_gens[i]
            logger.info(f"{branch}TransformGens used: {trg} in training")
    else:
        logger.info(f"TransformGens used: {transform_gens} in training")

    
    dataset = build_dataset(cfg,
                            cfg.DATASETS.TEST,
                            transforms=transform_gens,
                            is_train=False)

    sampler = SAMPLERS.get("InferenceSampler")(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=True,
    )
    return data_loader



def build_test_loader(cfg):
    """
    Similar to `build_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a cvpods CfgNode

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    transform_gens = build_transform_gens(cfg.INPUT.AUG.TEST_PIPELINES)
    logger.info(f"TransformGens used: {transform_gens} in testing")
    dataset = build_dataset(cfg,
                            cfg.DATASETS.TEST,
                            transforms=transform_gens,
                            is_train=False)
    sampler = SAMPLERS.get("InferenceSampler")(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=True,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)
