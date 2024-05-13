import argparse
import yaml
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import WandbLogger
from core.data.samples_module import WorldSampleDataModule
from core.train.structure_module import StructureModule
from core.train.color_module import ColorModule
from itertools import chain


class EMACallback(Callback):
    def on_after_backward(
        trainer: Trainer,
        module: LightningModule,
    ) -> None:
        ema_model = getattr(model, "ema", None)
        ema_ctrl_model = getattr(model, "ctrl_emb_ema", None)

        if ema_model is not None:
            ema_model.update()

        if ema_ctrl_model is not None:
            ema_ctrl_model.update()


def train(args: argparse.Namespace) -> None:
    with open(args.config, "r") as file:
        conf = yaml.safe_load(file)

        # First flatten conf
        defaults = {
            key: value for key, value in chain(*[category.items() for category in conf.values() if isinstance(category, dict)])
        }

        # Then update any args that are None with the defaults
        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    data_module = WorldSampleDataModule(
        tube_length=args.tube_length,
        dataset_mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        device=args.data_device,
    )

    if args.mode == "structure":
        module = StructureModule
    elif args.mode == "color":
        module = ColorModule
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    model_kwargs = vars(args)
    model = module.from_conf(model_kwargs.pop("config"), **model_kwargs)
    best_ckpt_callback = ModelCheckpoint(dirpath=args.ckpt_dir, monitor="val_loss_epoch", save_top_k=1, mode="min")
    last_ckpt_callback = ModelCheckpoint(dirpath=args.ckpt_dir, monitor="step", every_n_train_steps=args.val_check_interval, save_top_k=1, mode="max", save_last="link")
    lr_callback = LearningRateMonitor(logging_interval="step")
    ema_callback = EMACallback()
    logger = WandbLogger(log_model=False, project="mc-dreamer", name=f"{args.mode}_training")

    trainer = Trainer(
        logger = logger,
        callbacks = [best_ckpt_callback, last_ckpt_callback, lr_callback, ema_callback],
        max_epochs = -1,
        max_steps = args.steps,
        accumulate_grad_batches = args.accumulate_grad_batches,
        gradient_clip_val = args.gradient_clip_val,
        precision = args.precision,
        accelerator = args.accelerator,
        devices = int(args.devices) if args.devices.isdigit() else args.devices,
        benchmark = True,
        val_check_interval = args.val_check_interval * args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
