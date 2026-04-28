import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from ptflops import get_model_complexity_info
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchinfo import summary

from metrics import ciede2000
from main_model import DA_Net_t


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained dehazing model.")
    parser.add_argument("--trained_dir", type=str, default="./trained_models/DA-Net_RSID_195.pk")
    parser.add_argument("--test_dir", type=str, default="./dataset")
    parser.add_argument("--hazy_dir", type=str, default="haze")
    parser.add_argument("--GT_dir", type=str, default="GT")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id used for evaluation")
    return parser.parse_args()


class RunningAverage:
    def __init__(self):
        self.reset()

    def reset(self):
        self.latest = 0.0
        self.total = 0.0
        self.samples = 0
        self.average = 0.0

    def update(self, value, n=1):
        self.latest = value
        self.total += value * n
        self.samples += n
        self.average = self.total / self.samples if self.samples > 0 else 0.0


class PairedDehazingDataset(data.Dataset):
    def __init__(self, root_dir, hazy_folder="haze", gt_folder="GT"):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.hazy_root = self.root_dir / hazy_folder
        self.gt_root = self.root_dir / gt_folder

        self.hazy_files = sorted([self.hazy_root / name for name in os.listdir(self.hazy_root)])

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, index):
        hazy_path = self.hazy_files[index]

        hazy_img = Image.open(hazy_path).convert("RGB")

        image_id = hazy_path.name.split("_")[0]
        gt_path = self.gt_root / image_id

        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = transforms.CenterCrop(hazy_img.size[::-1])(gt_img)

        hazy_tensor = self.normalize(self.to_tensor(hazy_img))
        gt_tensor = self.to_tensor(gt_img)

        return hazy_tensor, gt_tensor, gt_path.name


def tensor_to_uint8_image(tensor):
    image = tensor.detach().clamp(0, 1).cpu().squeeze(0)
    image = image.permute(1, 2, 0).numpy()
    image = np.round(image * 255.0).astype(np.uint8)
    return Image.fromarray(image)


def compute_ssim(prediction, reference):
    _, _, height, width = prediction.shape
    scale = max(1, round(min(height, width) / 256))

    pooled_prediction = F.adaptive_avg_pool2d(
        prediction,
        output_size=(int(height / scale), int(width / scale))
    )

    pooled_reference = F.adaptive_avg_pool2d(
        reference,
        output_size=(int(height / scale), int(width / scale))
    )

    return ssim(
        pooled_prediction,
        pooled_reference,
        data_range=1,
        size_average=False
    ).item()


def evaluate(loader, model, device, save_dir="./output"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    meters = {
        "psnr": RunningAverage(),
        "ssim": RunningAverage(),
        "mse": RunningAverage(),
        "ciede": RunningAverage(),
    }

    torch.cuda.empty_cache()
    model.eval()

    timer_start = torch.cuda.Event(enable_timing=True)
    timer_end = torch.cuda.Event(enable_timing=True)

    total_runtime = 0.0

    for step, (hazy, clean, names) in enumerate(loader):
        hazy = hazy.to(device)
        clean = clean.to(device)
        image_name = names[0]

        with torch.no_grad():
            timer_start.record()

            restored = model(hazy)

            timer_end.record()
            torch.cuda.synchronize()

            elapsed = timer_start.elapsed_time(timer_end) / 1000.0
            total_runtime += elapsed

            output_image = tensor_to_uint8_image(restored)
            output_file = save_path / f"{Path(image_name).stem}_{step + 1}.jpg"
            output_image.save(output_file, compress_level=0)

            restored = restored.clamp_(-1, 1)
            clean = clean.clamp_(-1, 1)

            restored = restored * 0.5 + 0.5
            clean = clean * 0.5 + 0.5

            mse_value = F.mse_loss(restored, clean).item()
            psnr_value = 10 * torch.log10(1 / F.mse_loss(restored, clean)).item()
            ssim_value = compute_ssim(restored, clean)
            ciede_value = ciede2000(
                restored.cpu().squeeze(0).numpy(),
                clean.cpu().squeeze(0).numpy()
            )

        meters["mse"].update(mse_value)
        meters["psnr"].update(psnr_value)
        meters["ssim"].update(ssim_value)
        meters["ciede"].update(ciede_value)

        print(
            f"Test: [{step}]\t"
            f"PSNR: {meters['psnr'].latest:.02f} ({meters['psnr'].average:.02f})\t"
            f"SSIM: {meters['ssim'].latest:.03f} ({meters['ssim'].average:.03f})\t"
            f"MSE: {meters['mse'].latest:.06f} ({meters['mse'].average:.06f})\t"
            f"CIEDE: {meters['ciede'].latest:.03f} ({meters['ciede'].average:.03f})"
        )

        print(f"Time: {elapsed} sec")

    total_images = len(loader)
    print(total_images)
    print(f"Time: {total_runtime / total_images} sec")
    print(
        f"FINAL AVERAGES — "
        f"PSNR: {meters['psnr'].average:.02f}  "
        f"SSIM: {meters['ssim'].average:.03f} "
        f"MSE: {meters['mse'].average:.06f} "
        f"CIEDE2000: {meters['ciede'].average:.03f}"
    )


def report_model_statistics(model):
    summary(model, (1, 3, 64, 64), depth=0)

    parameter_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size_mb = (parameter_bytes + buffer_bytes) / 1024 ** 2

    print(f"Size: {total_size_mb:.3f} MB")

    macs, params = get_model_complexity_info(
        model,
        (3, 224, 224),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print("{:<30}  {:<8}".format("Computational complexity (MACs): ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairedDehazingDataset(
        root_dir=args.test_dir,
        hazy_folder=args.hazy_dir,
        gt_folder=args.GT_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    checkpoint = torch.load(args.trained_dir, map_location=device)

    model = DA_Net_t().to(device)
    model = nn.DataParallel(model)

    report_model_statistics(model)

    model.load_state_dict(checkpoint["model"])

    evaluate(loader, model, device)


if __name__ == "__main__":
    main()