import argparse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from core.data.samples_module import WorldSampleDataModule
from core.train.structure_module import StructureModule
from core.train.color_module import ColorModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--mode", type=str, help="Training mode (structure or color)", default="structure")
    parser.add_argument("--steps", type=int, help="Number of steps to train for", default=100000)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=8)
    parser.add_argument("--num-workers", type=int, help="Number of workers", default=8)
    parser.add_argument("--tube-length", type=int, help="Length of tube", default=8)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="Weight decay", default=1e-3)
    parser.add_argument("--data-dir", type=str, help="Data directory", default="outputs")
    parser.add_argument("--device", type=str, help="Device", default="auto")
    parser.add_argument("--ckpt-dir", type=str, help="Checkpoint directory", default="checkpoints")
    parser.add_argument("--config", type=str, help="Config file", default="config/structure_default.yaml")
    parser.add_argument("--accelerator", type=str, help="Accelerator", default="gpu")
    parser.add_argument("--devices", type=int, help="Number of devices", default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, help="Number of batches to accumulate", default=2)
    parser.add_argument("--gradient-clip-val", type=float, help="Gradient clipping value", default=None)
    parser.add_argument("--precision", type=int, help="Precision", default=16)
    parser.add_argument("--val-check-interval", type=int, help="Validation check interval", default=1000)

    args = parser.parse_args()

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
