import argparse
from core.train.trainer import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--mode", type=str, help="Training mode (structure or color)", default="structure")
    parser.add_argument("--steps", type=int, help="Number of steps to train for", default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=None)
    parser.add_argument("--num-workers", type=int, help="Number of workers", default=8)
    parser.add_argument("--tube-length", type=int, help="Length of tube", default=None)
    parser.add_argument("--lr", type=float, help="Learning rate", default=None)
    parser.add_argument("--wd", type=float, help="Weight decay", default=None)
    parser.add_argument("--data-dir", type=str, help="Data directory", default="outputs")
    parser.add_argument("--ckpt-dir", type=str, help="Checkpoint directory", default="checkpoints")
    parser.add_argument("--config", type=str, help="Config file", default="core/train/config/structure_default.yaml")
    parser.add_argument("--accelerator", type=str, help="Accelerator", default=None)
    parser.add_argument("--devices", help="Number of devices", default="auto")
    parser.add_argument("--data-device", type=str, help="Device to use for data", default="cpu")
    parser.add_argument("--accumulate-grad-batches", type=int, help="Number of batches to accumulate", default=None)
    parser.add_argument("--gradient-clip-val", type=float, help="Gradient clipping value", default=None)
    parser.add_argument("--precision", type=int, help="Precision", default=None)
    parser.add_argument("--val-check-interval", type=int, help="Validation check interval", default=None)

    args = parser.parse_args()

    train(args)
