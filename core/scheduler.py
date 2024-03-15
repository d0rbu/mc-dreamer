import torch as th
from bisect import bisect_right
from typing import Any, Self


def step_scheduler(scheduler: th.optim.lr_scheduler.LRScheduler, epoch: int | None = None, metrics: Any = None) -> None:
    if isinstance(scheduler, th.optim.lr_scheduler.ReduceLROnPlateau):
        if metrics is not None:
            scheduler.step(metrics, epoch)

        return

    if isinstance(scheduler, (ChainedScheduler, SequentialLR)):
        scheduler.step(epoch, metrics)
        return

    if isinstance(scheduler, th.optim.lr_scheduler.CosineAnnealingWarmRestarts):
        scheduler.step(epoch)
        return

    scheduler.step()


# why arent these classes compatible
class SequentialLR(th.optim.lr_scheduler.SequentialLR):
    def step(self: Self, epoch: int | None = None, metrics: Any = None) -> None:
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]

        milestone = 0 if idx == 0 else self._milestones[idx - 1]
        step_scheduler(scheduler, epoch - milestone, metrics)

        self._last_lr = scheduler.get_last_lr()

class ChainedScheduler(th.optim.lr_scheduler.ChainedScheduler):
    @property
    def last_epoch(self: Self):
        return self._schedulers[-1].last_epoch
    
    @last_epoch.setter
    def last_epoch(self: Self, value: int):
        for scheduler in self._schedulers:
            scheduler.last_epoch = value

    def step(self: Self, epoch: int | None = None, metrics: Any = None) -> None:
        for scheduler in self._schedulers:
            step_scheduler(scheduler, epoch, metrics)

        self._last_lr = [group['lr'] for group in self._schedulers[-1].optimizer.param_groups]