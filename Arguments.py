import argparse
import os
import warnings

import torch


warnings.filterwarnings("ignore")


def get_config():
    parser = argparse.ArgumentParser(
        description="Training configuration for DSCH_Net."
    )

    parser.add_argument("--dataset_dir", type=str, default="./dataset/RSID/")
    parser.add_argument("--train", type=str, default="train", help="training folder name")
    parser.add_argument("--test", type=str, default="test", help="testing folder name")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--bs", type=int, default=2, help="batch size")

    parser.add_argument("--model_dir", type=str, default="./trained_models/")
    parser.add_argument("--net", type=str, default="DSCH_Net")

    parser.add_argument("--crop", action="store_true")
    parser.add_argument(
        "--crop_size",
        type=int,
        default=240,
        help="Crop size used only when --crop is enabled"
    )

    parser.add_argument(
        "--no_lr_sche",
        action="store_true",
        help="Disable cosine learning-rate schedule"
    )

    parser.add_argument(
        "--clip",
        action="store_true",
        help="Enable gradient clipping"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id used for training"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="Automatic detection"
    )

    return parser.parse_args()


def prepare_runtime(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.model_dir, exist_ok=True)

    return args


opt = prepare_runtime(get_config())

print(opt)
print("model_dir:", opt.model_dir)