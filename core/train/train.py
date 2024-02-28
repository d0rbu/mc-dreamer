import argparse
import lightning as L
from lightning import Trainer

from core.data.samples_module import WorldSampleDataModule
from core.train.structure_module import StructureModule
from core.train.color_module import ColorModule
from core.model.sinkformer import SinkFormerConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--mode", type=str, help="Training mode (structure or color)", default="structure")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train for", default=10)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=8)
    parser.add_argument("--num-workers", type=int, help="Number of workers", default=8)
    parser.add_argument("--tube-length", type=int, help="Length of tube", default=8)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="Weight decay", default=1e-3)
    parser.add_argument("--data-dir", type=str, help="Data directory", default="outputs")
    parser.add_argument("--device", type=str, help="Device", default="auto")
    parser.add_argument("--ckpt-dir", type=str, help="Checkpoint directory", default="checkpoints")
    parser.add_argument("--config", type=str, help="Config file", default="config/structure_default.yaml")
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
    trainer = Trainer()

    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_dir)
