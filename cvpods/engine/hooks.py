#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import datetime
import itertools
import os
import tempfile
import time
from collections import Counter
import megfile
from loguru import logger
from bisect import bisect_right
import torch
from subprocess import TimeoutExpired, Popen, PIPE, STDOUT
import time
HIGH_TEMPERATURE_WARNING = 1

from cvpods.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from cvpods.evaluation.testing import flatten_results_dict
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules, update_bn_stats
from cvpods.utils import EventStorage, EventWriter, Timer, comm, ensure_dir

__all__ = [
    "HookBase",
    "CallbackHook",
    "IterationTimer",
    "OptimizationHook",
    "PeriodicWriter",
    "PeriodicCheckpointer",
    "LRScheduler",
    "AutogradProfiler",
    "EvalHook",
    "PreciseBN",
    "MeanTeacher",
    "gpus_temp_monitor"
]


"""
Implement some common hooks.
"""


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class CallbackHook(HookBase):
    """
    Create a hook using callback functions provided by the user.
    """

    def __init__(self, *, before_train=None, after_train=None, before_step=None, after_step=None):
        """
        Each argument is a function that takes one argument: the trainer.
        """
        self._before_train = before_train
        self._before_step = before_step
        self._after_step = after_step
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        # The functions may be closures that hold reference to the trainer
        # Therefore, delete them to avoid circular reference.
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class OptimizationHook(HookBase):
    def __init__(self, accumulate_grad_steps=1, grad_clipper=None, mixed_precision=False):
        self.accumulate_grad_steps = accumulate_grad_steps
        self.grad_clipper = grad_clipper
        self.mixed_precision = mixed_precision

    def before_step(self):
        self.trainer.optimizer.zero_grad()

    def after_step(self):
        losses = self.trainer.step_outputs["loss_for_backward"]
        losses /= self.accumulate_grad_steps

        if self.mixed_precision:
            from apex import amp
            with amp.scale_loss(losses, self.trainer.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()

        if self.trainer.inner_iter == self.accumulate_grad_steps:
            if self.grad_clipper is not None:
                self.grad_clipper(self.tariner.model.paramters())
            self.trainer.optimizer.step()
            self.trainer.optimizer.zero_grad()


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.

    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage periodically.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, writers, period=100):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (
            (self.trainer.iter + 1) % self._period == 0
            or (self.trainer.iter == self.trainer.max_iter - 1)
            or (self.trainer.iter == 0)
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.close()


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as :class:`cvpods.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        # `self.max_iter` and `self.max_epoch` will be initialized in __init__
        pass

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self._scheduler.step()


class AutogradProfiler(HookBase):
    r"""
    A hook which runs `torch.autograd.profiler.profile`.

    Examples:
    .. code-block:: python

        hooks.AutogradProfiler(
            lambda trainer: trainer.iter > 10 and trainer.iter < 20, self.cfg.OUTPUT_DIR
            )

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    The result files can be loaded in the ``chrome://tracing`` page in chrome browser.

    Note:
        When used together with NCCL on older version of GPUs,
        autograd profiler may cause deadlock because it unnecessarily allocates
        memory on every device it sees. The memory management calls, if
        interleaved with NCCL calls, lead to deadlock on GPUs that do not
        support `cudaLaunchCooperativeKernelMultiDevice`.
    """

    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
            output_dir (str): the output directory to dump tracing files.
            use_cuda (bool): same as in `torch.autograd.profiler.profile`.
        """
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        ensure_dir(self._output_dir)
        out_file = os.path.join(
            self._output_dir, "profiler-trace-iter{}.json".format(self.trainer.iter)
        )
        if "://" not in out_file:
            self._profiler.export_chrome_trace(out_file)
        else:
            # Support non-posix filesystems
            with tempfile.TemporaryDirectory(prefix="cvpods_profiler") as d:
                tmp_file = os.path.join(d, "tmp.json")
                self._profiler.export_chrome_trace(tmp_file)
                with open(tmp_file) as f:
                    content = f.read()
            with megfile.smart_open(out_file, "w") as f:
                f.write(content)


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`.
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = (next_iter == self.trainer.max_iter) and (self._period >= 0)
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_eval()

    def after_train(self):
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class PreciseBN(HookBase):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, period, model, data_loader, num_iter):
        """
        Args:
            period (int): the period this hook is run, or 0 to not run during training.
                The hook will always run in the end of training.
            model (nn.Module): a module whose all BN layers in training mode will be
                updated by precise BN.
                Note that user is responsible for ensuring the BN layers to be
                updated are in training mode when this hook is triggered.
            data_loader (iterable): it will produce data to be run by `model(data)`.
            num_iter (int): number of iterations used to compute the precise
                statistics.
        """
        if len(get_bn_modules(model)) == 0:
            logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._period = period
        self._disabled = False

        self._data_iter = iter(self._data_loader)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        """
        if self._disabled:
            return

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    logger.info(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter)
                    )
                # This way we can reuse the same iterator
                try:
                    item = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    item = next(self._data_iter)

                yield item

        with EventStorage():  # capture events in a new storage to discard them
            logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)


class MeanTeacher(HookBase):
    def __init__(
        self,
        runner,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.runner = runner
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_train(self):
        model = self.runner.model
        teacher_model = self.runner.teacher_model
        # only do it at initial stage

        if self.runner.iter == 0:
            logger.info("Clone all parameters of student to teacher...")
            self.momentum_update(model,teacher_model, 0)

    def before_step(self):
        """Update ema parameter every self.interval iterations."""
        curr_step = self.runner.iter
        if curr_step % self.interval != 0:
            return
        model = self.runner.model
        teacher_model = self.runner.teacher_model
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )

        self.momentum_update(model,teacher_model, momentum)

    def after_step(self):
        curr_step = self.runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model,teacher_model, momentum):

        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.named_parameters(),teacher_model.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)




def log_output(command, ok=(0,)):
    output = []
    p = Popen(command, stdout=PIPE, stderr=STDOUT)
    try:
        p.wait(60)
        for line in p.stdout:
            output.append(line.decode().strip())
    except TimeoutExpired:
        print('Command timed out: ' + ' '.join(command))
        raise
    finally:
        if p.returncode not in ok:
            print('\n'.join(output))
            raise ValueError('Command crashed with return code ' + str(p.returncode) + ': ' + ' '.join(command))
        return '\n'.join(output)


def gpu_buses(gpus:list):

    buses = log_output(['nvidia-smi', '--format=csv,noheader', '--query-gpu=pci.bus_id']).splitlines()
    aa = set(gpus)
    assert len(aa) == len(gpus)
    new_buses = []
    for i in gpus:
        assert i>=0 and i<=3
        new_buses.append(buses[i])
    return new_buses

def query(bus, field):
    [line] = log_output(['nvidia-smi', '--format=csv,noheader', '--query-gpu='+field, '-i', bus]).splitlines()
    return line


def temperature(bus):
    return int(query(bus, 'temperature.gpu'))

class gpus_temp_monitor(HookBase):
    '''
    for High temperature warning
 
    '''
    def __init__(self,
                 gpus=[0,1,2,3],
                 max_temp=78,
                 stop_time=300,
                 try_to_pull_fans=False) -> None:
        aa = set(gpus)
        assert len(aa) == len(gpus)
        for i in gpus:
            assert i>=0 and i<=3
        self.gpus = gpus
        self.buses = gpu_buses(self.gpus)
        assert max_temp >= 70 and max_temp <= 85
        self.max_temp = max_temp
        self.stop_time = stop_time
        self.try_to_pull_fans = try_to_pull_fans


    def before_step(self):

        for bus in self.buses:
            temp = temperature(bus)
            if temp >= self.max_temp:
                time.sleep(self.stop_time)
                return  HIGH_TEMPERATURE_WARNING
        
        return 0


