import argparse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from core.data.samples_module import WorldSampleDataModule
from core.train.structure_module import StructureModule
from core.train.color_module import ColorModule


def train(args: argparse.Namespace) -> None:
    data_module = WorldSampleDataModule(
        tube_length=args.tube_length,
        dataset_mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        device=args.device,
    )

    if args.mode == "structure":
        module = StructureModule
    elif args.mode == "color":
        module = ColorModule
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    model = module.from_conf(args.config, **vars(args))
    ckpt_callback = ModelCheckpoint(dirpath=args.ckpt_dir, monitor="val_loss", save_top_k=1, mode="min")
    logger = WandbLogger(log_model="all", project="mc-dreamer", name=f"{args.mode}_training", callbacks=[ckpt_callback])

    trainer = Trainer(
        logger = logger,
        max_steps = args.steps,
        accumulate_grad_batches = args.accumulate_grad_batches,
        gradient_clip_val = args.gradient_clip_val,
        precision = args.precision,
        accelerator = args.accelerator,
        devices = args.devices,
        benchmark = True
    )

    trainer.fit(model, datamodule=data_module)
