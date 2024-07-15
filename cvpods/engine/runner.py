#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import math
import os
from collections import OrderedDict
from cvpods.structures import instances
from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel
import time
from cvpods.checkpoint import DefaultCheckpointer
from cvpods.data import build_test_loader, build_train_loader,build_train_loader_mutiBranch,build_test_loader_mutiBranch
from cvpods.data.samplers.infinite import Infinite
from cvpods.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    inference_on_files,
    print_csv_format,
    verify_results
)
from cvpods.modeling.nn_utils.module_converter import maybe_convert_module
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import comm
from cvpods.utils.compat_wrapper import deprecated
from cvpods.utils.dump.events import CommonMetricPrinter, JSONWriter

from . import hooks
from .base_runner import RUNNERS, SimpleRunner

from tqdm import tqdm
import numpy as np
costom_num = dict()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

past = dict()
def grad_check(model,params_container,print_flag=True):
    """print which layer's weight changed;

    Args:
        model:model
        params_container:(global list/dict) [] or last model's params
    Returns:
        params_container:(global list/dict) new model's params
        count: changed layers number
    """
    i = 0
    count = 0
    if isinstance(params_container,list):
        if len(params_container) !=0:
            for name, param in model.named_parameters():
                if torch.where(params_container[i]-param.clone().detach())[0].numel()!=0:
                    if print_flag:
                        print(name,':changed!')
                else:
                    if print_flag:
                        print(name,':no_changed!')
                    count += 1
                i += 1
    elif isinstance(params_container,dict):
        if len(params_container) !=0:
            for name, param in model.named_parameters():
                if torch.where(params_container[name]-param.clone().detach())[0].numel()!=0:
                    if print_flag:
                        print(name,':changed!')
                else:
                    # if print_flag:
                        # print(name,':no_changed!')
                    count += 1
    # if len(params_container) == 0:
    #     print_requires_grad_layers = True
    # else:
    #     print_requires_grad_layers = False
    # params_container = []
    # for name, param in model.named_parameters(): #查看可优化的参数有哪些
    #     if print_requires_grad_layers:
    #         if param.requires_grad:
    #             if print_flag:
    #                 print('requires grad:',name)
    #         else:
    #             if print_flag:
    #                 print('not requires grad:',name)
    #     params_container.append(param.clone().detach())
    if print_flag:
        print('*' *30)
        print('*' *30)
    return params_container,count


@RUNNERS.register()
class DefaultRunner(SimpleRunner):
    """
    A runner with default training logic. It does the following:

    1. Create a :class:`DefaultRunner` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`DefaultRunner` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`DefaultRunner`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in cvpods.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        runner = DefaultRunner(cfg)
        runner.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        runner.train()

    Attributes:
        scheduler:
        checkpointer (DefaultCheckpointer):
        cfg (config dict):
    """

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """

        self.data_loader = self.build_train_loader(cfg)
        # Assume these objects must be constructed in this order.
        model = build_model(cfg)
        self.model = maybe_convert_module(model)
        logger.info(f"Model: \n{self.model}")

        # Assume these objects must be constructed in this order.
        self.optimizer = self.build_optimizer(cfg, self.model)

        if cfg.TRAINER.FP16.ENABLED:
            self.mixed_precision = True
            if cfg.TRAINER.FP16.TYPE == "APEX":
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                )
        else:
            self.mixed_precision = False

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            if cfg.MODEL.DDP_BACKEND == "torch":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=False,
                    find_unused_parameters=False
                )
            elif cfg.MODEL.DDP_BACKEND == "apex":
                from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                self.model = ApexDistributedDataParallel(self.model)
            else:
                raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))

        super().__init__(
            self.model,
            self.data_loader,
            self.optimizer,
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,

            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.OptimizationHook(
                accumulate_grad_steps=cfg.SOLVER.BATCH_SUBDIVISIONS,
                grad_clipper=None,
                mixed_precision=cfg.TRAINER.FP16.ENABLED
            ),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))
            # Put `PeriodicDumpLog` after writers so that can dump all the files,
            # including the files generated by writers

        return ret

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`JSONWriter`.

        Args:
            output_dir: directory to store JSON metrics and tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(
                self.max_iter,
                window_size=self.window_size,
                epoch=self.max_epoch,
            ),
            JSONWriter(
                os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"),
                window_size=self.window_size
            )
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        if self.max_epoch is None:
            logger.info("Starting training from iteration {}".format(self.start_iter))
        else:
            logger.info("Starting training from epoch {}".format(self.start_epoch))

        super().train(self.start_iter, self.start_epoch, self.max_iter)

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`cvpods.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, **kwargs):
        """
        It now calls :func:`cvpods.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, **kwargs)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_test_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            # TODO: add this tutorial
            """
If you want DefaultRunner to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
            """
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None, output_folder=None):
        """
        Args:
            cfg (config dict):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            # assert evaluators is not None
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    # from pdb import set_trace
                    # set_trace()
                    from cvpods.evaluation import build_evaluator
                    evaluator = build_evaluator(
                        cfg, dataset_name, data_loader.dataset, output_folder=output_folder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultRunner.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue

            results_i = inference_on_dataset(model, data_loader, evaluator)
            # if cfg.TEST.ON_FILES:
            #     results_i = inference_on_files(evaluator)
            # else:
            #     results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def auto_scale_config(cfg, dataloader):
    max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
    max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER

    subdivision = cfg.SOLVER.BATCH_SUBDIVISIONS
    # adjust lr by batch_subdivisions
    cfg.SOLVER.OPTIMIZER.BASE_LR *= subdivision

    """
    Here we use batch size * subdivision to simulator large batch training
    """
    if max_epoch:
        epoch_iter = math.ceil(
            len(dataloader.dataset) / (cfg.SOLVER.IMS_PER_BATCH * subdivision))

        if max_iter is not None:
            logger.warning(
                f"Training in EPOCH mode, automatically convert {max_epoch} epochs "
                f"into {max_epoch*epoch_iter} iters...")

        cfg.SOLVER.LR_SCHEDULER.MAX_ITER = max_epoch * epoch_iter
        cfg.SOLVER.LR_SCHEDULER.STEPS = [
            x * epoch_iter for x in cfg.SOLVER.LR_SCHEDULER.STEPS
        ]
        cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS = int(
            cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS * epoch_iter)
        cfg.SOLVER.CHECKPOINT_PERIOD = epoch_iter * cfg.SOLVER.CHECKPOINT_PERIOD
        cfg.TEST.EVAL_PERIOD = epoch_iter * cfg.TEST.EVAL_PERIOD
    else:
        epoch_iter = -1

    cfg.SOLVER.LR_SCHEDULER.EPOCH_ITERS = epoch_iter


@RUNNERS.register()
@deprecated("Use DefaultRunner instead.")
class DefaultTrainer(DefaultRunner):
    pass


@RUNNERS.register()
class MultiBranchRunner(SimpleRunner):

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """
        self.only_forward = cfg.TRAINER.get('ONLY_FORWARD',False)
        self.data_loader = build_train_loader_mutiBranch(cfg)
        if cfg.MODEL.FCOS.get('WITH_TEACHER',None):
            model,self.teacher_model = build_model(cfg)
        else:
            model= build_model(cfg)
            self.teacher_model = None
        self.model = maybe_convert_module(model)
        if self.teacher_model is not None:
            self.teacher_model = maybe_convert_module(self.teacher_model)
        logger.info(f"Model: \n{self.model}")
        # from pdb import set_trace
        # set_trace()
        # setattr(self.model,"teacher_model",self.teacher_model)

        # if not self.only_forward:
        self.optimizer = build_optimizer(cfg, self.model)

        if cfg.TRAINER.FP16.ENABLED:
            self.mixed_precision = True
            if cfg.TRAINER.FP16.TYPE == "APEX":
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                )

        else:
            self.mixed_precision = False
        

        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            if cfg.MODEL.DDP_BACKEND == "torch":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=False,
                    find_unused_parameters=False
                )

            elif cfg.MODEL.DDP_BACKEND == "apex":
                from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                self.model = ApexDistributedDataParallel(self.model)
            else:
                raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))


        super().__init__(
            self.model,
            self.data_loader,
            self.optimizer,
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)
        self.scheduler = build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            "model_iter_",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        if self.teacher_model is not None:
            self.teacher_checkpointer = DefaultCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.teacher_model,
                cfg.OUTPUT_DIR,
                "teacher_model_iter_",
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        else:
            self.teacher_checkpointer = None
        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg

        self.splg_start_iter = cfg.TRAINER.get('START_SPLG_ITER')
        if self.only_forward:
            self.model.freeze_all()
            self.model.eval()
        self.register_hooks(self.build_hooks())        


    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """

        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        # _ = (self.teacher_checkpointer.resume_or_load(
        #     self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)

        if self.teacher_checkpointer:
            # from pdb import set_trace
            # set_trace()
            weight = self.cfg.MODEL.get('TEACHER_WEIGHTS',self.cfg.MODEL.WEIGHTS)
            # aa = torch.load(weight)
            self.teacher_checkpointer.resume_or_load(weight,resume=resume)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length
        
        
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.OptimizationHook(
                accumulate_grad_steps=cfg.SOLVER.BATCH_SUBDIVISIONS,
                grad_clipper=None,
                mixed_precision=cfg.TRAINER.FP16.ENABLED
            ),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))
            if self.teacher_checkpointer is not None:
                ret.append(hooks.PeriodicCheckpointer(
                    self.teacher_checkpointer,
                    cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_iter=self.max_iter,
                    max_epoch=self.max_epoch
                ))
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        if not self.only_forward:
            ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=100
            ))
            # Put `PeriodicDumpLog` after writers so that can dump all the files,
            # including the files generated by writers

        return ret


    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        if self.max_epoch is None:
            logger.info("Starting training from iteration {}".format(self.start_iter))
        else:
            logger.info("Starting training from epoch {}".format(self.start_epoch))

        from cvpods.utils.dump.events import EventStorage
        avg_meters = {}
        pbar = tqdm(total=self.max_iter)   

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.iter = self.start_iter
                # setattr(self.model,"teacher_model",self.teacher_model)
                self.before_train()

                self.start_training = True
                for self.iter in range(self.start_iter, self.max_iter):
                    self.inner_iter = 0
                    self.before_step()
                    # by default, a step contains data_loading and model forward,
                    # loss backward is executed in after_step for better expansibility
                    self.run_step(pbar,avg_meters)
                    # if self.max_iter - self.iter <=1:
                        # from pdb import set_trace
                        # set_trace()
                    if self.only_forward:
                        if self.max_iter - self.iter <= 1:
                            from pdb import set_trace
                            set_trace()
                            np.save('iou_score_nomiss_raw.npy',np.concatenate(self.model.iou_score_nomiss_raw,axis=1))
                            np.save('iou_score_miss_raw.npy',np.concatenate(self.model.iou_score_miss_raw,axis=1))
                            np.save('iou_score_gt_raw.npy',np.concatenate(self.model.iou_score_gt_raw,axis=1))
                            np.save('iou_score_nomiss_nor.npy',np.concatenate(self.model.iou_score_nomiss_nor,axis=1))
                            np.save('iou_score_gt_nor.npy',np.concatenate(self.model.iou_score_gt_nor,axis=1))
                            np.save('iou_score_nomiss_str.npy',np.concatenate(self.model.iou_score_nomiss_str,axis=1))
                            np.save('iou_score_miss_str.npy',np.concatenate(self.model.iou_score_miss_str,axis=1))
                            np.save('iou_score_gt_str.npy',np.concatenate(self.model.iou_score_gt_str,axis=1))
                        self.after_step(1)
                    else:
                        self.after_step()


                    # grad_check(self.teacher_model,past)

                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
                # from pdb import set_trace
                # set_trace()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self,pbar=None,avg_meters=None):
        """
        Implement the standard training logic described above.
        """
        # from pdb import set_trace
        # set_trace()
        start_splg = self.cfg.TRAINER.get('START_SPLG_ITER',-1)
        # from pdb import set_trace
        # set_trace()
        if self.start_training or start_splg > self.iter:
            global costom_num
            costom_num['found_num'] = costom_num['all_num'] \
                    = costom_num['wrong_loc_num']= costom_num['ori_num'] \
                    = costom_num['wrong_cls_num'] = 0
            self.start_training = False
        
        if self.splg_start_iter is not None:
            if self.splg_start_iter < self.iter:
                costom_num['with_SPLG'] = True
        else:
            costom_num['with_SPLG'] = False
        if not self.only_forward:
            assert self.model.training, "[IterRunner] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self.epoch += 1
            costom_num['found_num'] = costom_num['all_num'] \
                    = costom_num['wrong_loc_num']= costom_num['ori_num'] \
                    = costom_num['wrong_cls_num'] = 0
                    
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self.epoch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        
        data[0]['iter'] = self.iter
        # from pdb import set_trace
        # set_trace()
        # try:
        loss_dict = self.model(data,self.teacher_model)
        # except:
        #     loss_dict = self.model(data,self.teacher_model)

        if self.only_forward:

            return None

        self.inner_iter += 1

        if isinstance(loss_dict,tuple):
            costom_num = loss_dict[1]
            loss_dict = loss_dict[0]

        losses = sum([
            metrics_value for metrics_value in loss_dict.values()
            if metrics_value.requires_grad
        ])
        self._detect_anomaly(losses, loss_dict)

        self._write_metrics(loss_dict, data_time,custom_metrics=costom_num,pbar=pbar)

        self.step_outputs = {
            "loss_for_backward": losses,
        }

        
        
        # if pbar is not None:
            # pbar.update(1)

    @classmethod
    def test(cls, cfg, model, evaluators=None, output_folder=None,teacher_model=None):
        """
        Args:
            cfg (config dict):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            if teacher_model is not None:
                data_loader = build_test_loader_mutiBranch(cfg)
            else:
                data_loader = build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    
                    from cvpods.evaluation import build_evaluator
                    evaluator = build_evaluator(
                        cfg, dataset_name, data_loader.dataset, output_folder=output_folder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultRunner.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue

            # results_i = inference_on_dataset(model, data_loader, evaluator)
            if cfg.TEST.ON_FILES:
                results_i = inference_on_files(evaluator)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator,teacher_model=teacher_model)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`JSONWriter`.

        Args:
            output_dir: directory to store JSON metrics and tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(
                self.max_iter,
                window_size=self.window_size,
                epoch=self.max_epoch,
            ),
            JSONWriter(
                os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"),
                window_size=self.window_size
            )
        ]