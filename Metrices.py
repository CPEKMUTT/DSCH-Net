import math
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2lab, deltaE_ciede2000


def build_gaussian_kernel_1d(kernel_size, sigma):
    center = kernel_size // 2

    values = [
        math.exp(-((position - center) ** 2) / (2.0 * sigma ** 2))
        for position in range(kernel_size)
    ]

    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()


def build_ssim_window(kernel_size, num_channels):
    kernel_1d = build_gaussian_kernel_1d(kernel_size, sigma=1.5).view(kernel_size, 1)
    kernel_2d = torch.mm(kernel_1d, kernel_1d.t())

    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    window = kernel_2d.expand(num_channels, 1, kernel_size, kernel_size).contiguous()

    return window


def compute_ssim_map(img_a, img_b, window, kernel_size, channels):
    padding = kernel_size // 2

    mean_a = F.conv2d(img_a, window, padding=padding, groups=channels)
    mean_b = F.conv2d(img_b, window, padding=padding, groups=channels)

    mean_a_sq = mean_a * mean_a
    mean_b_sq = mean_b * mean_b
    mean_ab = mean_a * mean_b

    var_a = F.conv2d(img_a * img_a, window, padding=padding, groups=channels) - mean_a_sq
    var_b = F.conv2d(img_b * img_b, window, padding=padding, groups=channels) - mean_b_sq
    cov_ab = F.conv2d(img_a * img_b, window, padding=padding, groups=channels) - mean_ab

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2.0 * mean_ab + c1) * (2.0 * cov_ab + c2)
    denominator = (mean_a_sq + mean_b_sq + c1) * (var_a + var_b + c2)

    return numerator / denominator


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)

    channels = img1.size(1)

    window = build_ssim_window(window_size, channels)
    window = window.to(device=img1.device, dtype=img1.dtype)

    score_map = compute_ssim_map(
        img_a=img1,
        img_b=img2,
        window=window,
        kernel_size=window_size,
        channels=channels
    )

    if size_average:
        return score_map.mean()

    return score_map.mean(dim=1).mean(dim=1).mean(dim=1)


def psnr(pred, gt):
    pred_np = pred.clamp(0, 1).detach().cpu().numpy()
    gt_np = gt.clamp(0, 1).detach().cpu().numpy()

    mse = np.mean((pred_np - gt_np) ** 2)

    if mse == 0:
        return 100

    rmse = math.sqrt(mse)
    return 20 * math.log10(1.0 / rmse)


def ciede2000(color_value_expand, color_feat_expand):
    predicted_rgb = np.transpose(color_value_expand, (1, 2, 0))
    reference_rgb = np.transpose(color_feat_expand, (1, 2, 0))

    predicted_lab = rgb2lab(predicted_rgb)
    reference_lab = rgb2lab(reference_rgb)

    delta_e_map = deltaE_ciede2000(predicted_lab, reference_lab)

    return np.mean(delta_e_map)


if __name__ == "__main__":
    pass