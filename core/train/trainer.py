import argparse
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from core.data.samples_module import WorldSampleDataModule
from core.train.structure_module import StructureModule
from core.train.color_module import ColorModule
from itertools import chain


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
    ckpt_callback = ModelCheckpoint(dirpath=args.ckpt_dir, monitor="val_loss", save_top_k=1, mode="min")
    logger = WandbLogger(log_model="all", project="mc-dreamer", name=f"{args.mode}_training")

    trainer = Trainer(
        logger = logger,
        callbacks = [ckpt_callback],
        max_epochs = -1,
        max_steps = args.steps,
        accumulate_grad_batches = args.accumulate_grad_batches,
        gradient_clip_val = args.gradient_clip_val,
        precision = args.precision,
        accelerator = args.accelerator,
        devices = int(args.devices) if args.devices.isdigit() else args.devices,
        benchmark = True
    )

    trainer.fit(model, datamodule=data_module)
