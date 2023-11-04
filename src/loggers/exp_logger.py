import importlib
import os
from datetime import datetime


class ExperimentLogger:
    """Main class for experiment logging"""

    def __init__(self, log_path, exp_name, begin_time=None):
        self.log_path = log_path
        self.exp_name = exp_name
        self.exp_path = os.path.join(log_path, exp_name)
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        pass

    def log_args(self, args):
        pass

    def log_result(self, array, name, step):
        pass

    def log_figure(self, name, iter, figure, curtime=None):
        pass

    def save_model(self, state_dict, task):
        pass


class MultiLogger(ExperimentLogger):
    """This class allows to use multiple loggers"""

    def __init__(self, log_path, exp_name, loggers=None, save_models=True, tags=None):
        super(MultiLogger, self).__init__(log_path, exp_name)
        if os.path.exists(self.exp_path):
            print("WARNING: {} already exists!".format(self.exp_path))
        else:
            os.makedirs(os.path.join(self.exp_path, "models"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_path, "results"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_path, "figures"), exist_ok=True)

        self.save_models = save_models
        self.loggers = []
        self.wandb_id = None
        for i, l in enumerate(loggers):
            lclass = getattr(
                importlib.import_module(name="loggers." + l + "_logger"), "Logger"
            )
            if l == "wandb":
                self.loggers.append(lclass(self.log_path, self.exp_name, tags))
                self.wandb_id = i
            else:
                self.loggers.append(lclass(self.log_path, self.exp_name))

        self.iter_steps = {}

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        if iter is None:
            key = f"{group}_{task}_{name}"
            if key not in self.iter_steps:
                self.iter_steps[key] = 0
            iter = self.iter_steps[key]
            self.iter_steps[key] += 1

        if curtime is None:
            curtime = datetime.now()
        for l in self.loggers:
            l.log_scalar(task, iter, name, value, group, curtime)

    def log_args(self, args):
        for l in self.loggers:
            l.log_args(args)

    def log_result(self, array, name, step, skip_wandb=False, **kwargs):
        for i, l in enumerate(self.loggers):
            if skip_wandb and self.wandb_id is not None and i == self.wandb_id:
                continue
            l.log_result(array, name, step, **kwargs)

    def log_figure(self, name, iter, figure, curtime=None):
        if curtime is None:
            curtime = datetime.now()
        for l in self.loggers:
            l.log_figure(name, iter, figure, curtime)

    def save_model(self, state_dict, task):
        if self.save_models:
            for l in self.loggers:
                l.save_model(state_dict, task)
