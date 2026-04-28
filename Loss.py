import torch
import torch.nn as nn
import torch.nn.functional as F

# This is a smooth version of L1 loss.
# In image restoration, it is often more stable than plain L1,
# especially when the prediction has small pixel-level errors.
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()

        # A very small value is added inside the square root
        # so the gradient remains stable when the difference is close to zero.
        self.eps = epsilon

    def forward(self, pred, target):
        # Difference between restored image and clean image
        diff = pred - target

        # Charbonnier penalty: sqrt(diff^2 + eps)
        loss = torch.sqrt(diff * diff + self.eps)

        # Final scalar loss
        return loss.mean()


# Backward-compatible name.
# Keep this if your old train.py still calls L1_Charbonnier_loss().
L1_Charbonnier_loss = CharbonnierLoss


# ---------------------------------------------------------
# Gradient loss
# ---------------------------------------------------------
# This loss compares edge/gradient information between two images.
# It helps the model preserve sharper structures and object boundaries.
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Laplacian-like kernel.
        # It responds strongly around edges and local intensity changes.
        base_kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]

        # Same kernel is used separately for R, G, and B channels.
        kernel = torch.tensor(
            [base_kernel, base_kernel, base_kernel],
            dtype=torch.float32
        )

        # Shape becomes [3, 1, 3, 3], which works with groups=3.
        kernel = kernel.unsqueeze(1)

        # Register as buffer because this is a fixed filter, not trainable weight.
        self.register_buffer("kernel", kernel)

        # L1 is used to compare the gradient maps.
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Compute gradient/edge response for prediction
        pred_grad = F.conv2d(pred, self.kernel, groups=3)

        # Compute gradient/edge response for ground truth
        target_grad = F.conv2d(target, self.kernel, groups=3)

        # Compare both gradient maps
        return self.l1(pred_grad, target_grad)


# Backward-compatible name.
Gradient_Loss = GradientLoss


# ---------------------------------------------------------
# Gaussian kernel functions
# ---------------------------------------------------------
# These functions are used for SSIM and MS-SSIM calculation.
def gaussian_1d(kernel_size, sigma):
    # Coordinate positions centered around zero
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords = coords - kernel_size // 2

    # Standard Gaussian formula
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))

    # Normalize so all values sum to one
    kernel = kernel / kernel.sum()

    return kernel


def gaussian_2d(kernel_size, sigma):
    # First create a 1D Gaussian vector
    kernel_1d = gaussian_1d(kernel_size, sigma)

    # Outer product gives a 2D Gaussian window
    kernel_2d = torch.outer(kernel_1d, kernel_1d)

    return kernel_2d


def create_window(window_size, channel=1):
    # Create a 2D Gaussian kernel with sigma 1.5
    kernel = gaussian_2d(window_size, sigma=1.5)

    # Convert to convolution shape: [out_channels, in_channels/groups, H, W]
    kernel = kernel.view(1, 1, window_size, window_size)

    # Repeat the same window for every channel
    window = kernel.expand(channel, 1, window_size, window_size).contiguous()

    return window


# ---------------------------------------------------------
# SSIM function
# ---------------------------------------------------------
# Structural Similarity Index compares two images using luminance,
# contrast, and structural information.
def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None
):
    # Decide the dynamic range of image values.
    # For normalized images, the range is usually 1.
    # For 8-bit images, the range can be 255.
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        value_range = max_val - min_val
    else:
        value_range = val_range

    # Get image shape
    _, channel, height, width = img1.size()

    # Padding is kept as zero to match the original behavior.
    padding = 0

    # Create Gaussian window only when it is not provided.
    if window is None:
        real_window_size = min(window_size, height, width)
        window = create_window(real_window_size, channel)
        window = window.to(device=img1.device, dtype=img1.dtype)

    # Local mean of first image
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)

    # Local mean of second image
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    # Mean-related terms
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variance of first image
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq

    # Local variance of second image
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq

    # Local covariance between both images
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    # Stability constants from SSIM paper
    C1 = (0.01 * value_range) ** 2
    C2 = (0.03 * value_range) ** 2

    # Contrast sensitivity term
    cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    # Full SSIM map
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # Either return one scalar or one value per image
    if size_average:
        score = ssim_map.mean()
    else:
        score = ssim_map.mean(dim=1).mean(dim=1).mean(dim=1)

    # Some losses need both SSIM and contrast sensitivity.
    if full:
        return score, cs_map.mean()

    return score


# ---------------------------------------------------------
# Multi-scale SSIM
# ---------------------------------------------------------
# MS-SSIM checks similarity at multiple image scales.
# At each level, the image is downsampled by average pooling.
def msssim(
    img1,
    img2,
    window_size=11,
    size_average=True,
    val_range=None,
    normalize=False
):
    # Standard MS-SSIM weights
    weights = torch.FloatTensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    ).to(img1.device)

    mssim_values = []
    cs_values = []

    # Compute SSIM and contrast sensitivity at each scale
    for _ in range(weights.size(0)):
        sim, cs = ssim(
            img1,
            img2,
            window_size=window_size,
            size_average=size_average,
            full=True,
            val_range=val_range
        )

        mssim_values.append(sim)
        cs_values.append(cs)

        # Move to the next scale
        img1 = F.avg_pool2d(img1, kernel_size=(2, 2))
        img2 = F.avg_pool2d(img2, kernel_size=(2, 2))

    # Convert list to tensors
    mssim_values = torch.stack(mssim_values)
    cs_values = torch.stack(cs_values)

    # Optional normalization.
    # This is sometimes useful when training becomes unstable.
    if normalize:
        mssim_values = (mssim_values + 1) / 2
        cs_values = (cs_values + 1) / 2

    # Follow the MS-SSIM combination rule.
    cs_part = cs_values ** weights
    ssim_part = mssim_values ** weights

    # Same formulation as the source implementation.
    return torch.prod(cs_part[:-1] * ssim_part[-1])


# ---------------------------------------------------------
# SSIM module
# ---------------------------------------------------------
# This class wraps SSIM into nn.Module form.
# Useful when we want to use it like a normal PyTorch loss/module.
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Initial window assumes one channel.
        # It will be recreated automatically if input channels change.
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        # Reuse the old window only if the number of channels and dtype match.
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel)
            window = window.to(device=img1.device, dtype=img1.dtype)

            self.window = window
            self.channel = channel

        return ssim(
            img1,
            img2,
            window_size=self.window_size,
            window=window,
            size_average=self.size_average,
            val_range=self.val_range
        )


# ---------------------------------------------------------
# MS-SSIM module
# ---------------------------------------------------------
class MSSSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(
            img1,
            img2,
            window_size=self.window_size,
            size_average=self.size_average
        )


# ---------------------------------------------------------
# VGG perceptual loss
# ---------------------------------------------------------
# This loss compares deep feature maps instead of only pixels.
# It is useful when the restored image should look perceptually closer
# to the ground truth image.
class LossNetwork(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()

        # Usually this is torchvision.models.vgg16(...).features
        self.vgg_layers = vgg_model

        # Layers selected from VGG feature extractor.
        # These correspond to low-level and mid-level visual features.
        self.layer_name_mapping = {
            "3": "relu1_2",
            "8": "relu2_2",
            "15": "relu3_3"
        }

    def output_features(self, x):
        features = {}

        # Pass image through VGG layer by layer
        for name, module in self.vgg_layers._modules.items():
            x = module(x)

            # Save only selected feature maps
            if name in self.layer_name_mapping:
                features[self.layer_name_mapping[name]] = x

        return list(features.values())

    def forward(self, dehaze, gt):
        # Extract VGG features from model output
        dehaze_features = self.output_features(dehaze)

        # Extract VGG features from ground truth
        gt_features = self.output_features(gt)

        loss_values = []

        # Compare corresponding feature maps
        for pred_feature, gt_feature in zip(dehaze_features, gt_features):
            loss_values.append(F.l1_loss(pred_feature, gt_feature))

        # Average all selected VGG feature losses
        return sum(loss_values) / len(loss_values)


# ---------------------------------------------------------
# MS-SSIM + L1 combined loss
# ---------------------------------------------------------
# This loss mixes perceptual structural similarity with pixel-level L1.
# The implementation follows the common restoration-loss formulation.
class MS_SSIM_L1_LOSS(nn.Module):
    def __init__(
        self,
        gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
        data_range=1.0,
        K=(0.01, 0.03),
        alpha=0.025,
        compensation=200.0,
        cuda_dev=0
    ):
        super().__init__()

        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2

        self.alpha = alpha
        self.compensation = compensation

        # Padding depends on the largest Gaussian sigma.
        self.pad = int(2 * gaussian_sigmas[-1])

        # Kernel size also follows the largest sigma.
        filter_size = int(4 * gaussian_sigmas[-1] + 1)

        # There are 5 scales, and each scale has 3 RGB filters.
        masks = torch.zeros(
            (3 * len(gaussian_sigmas), 1, filter_size, filter_size)
        )

        # Fill the Gaussian masks for all scales and RGB channels.
        for idx, sigma in enumerate(gaussian_sigmas):
            gaussian_mask = self._make_gaussian_2d(filter_size, sigma)

            masks[3 * idx + 0, 0, :, :] = gaussian_mask
            masks[3 * idx + 1, 0, :, :] = gaussian_mask
            masks[3 * idx + 2, 0, :, :] = gaussian_mask

        # Keep same behavior as the original implementation:
        # the masks are directly moved to CUDA device.
        self.g_masks = masks.cuda(cuda_dev)

    def _make_gaussian_1d(self, size, sigma):
        # 1D coordinates centered at zero
        coords = torch.arange(size, dtype=torch.float32)
        coords = coords - size // 2

        # Gaussian curve
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))

        # Normalize
        gaussian = gaussian / gaussian.sum()

        return gaussian.reshape(-1)

    def _make_gaussian_2d(self, size, sigma):
        # Build 2D Gaussian using outer product
        gaussian_1d = self._make_gaussian_1d(size, sigma)
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)

        return gaussian_2d

    def forward(self, x, y):
        # Local means at multiple Gaussian scales
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        # Mean square terms
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        # Local variance and covariance
        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # Luminance component
        luminance = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)

        # Contrast-structure component
        contrast_structure = (2 * sigmaxy + self.C2) / (
            sigmax2 + sigmay2 + self.C2
        )

        # Last scale luminance for R, G, B channels
        last_luminance = (
            luminance[:, -1, :, :]
            * luminance[:, -2, :, :]
            * luminance[:, -3, :, :]
        )

        # Product of contrast-structure terms across all channels/scales
        cs_product = contrast_structure.prod(dim=1)

        # MS-SSIM loss map
        ms_ssim_loss = 1 - last_luminance * cs_product

        # Pixel-wise L1 loss
        l1_map = F.l1_loss(x, y, reduction="none")

        # Smooth L1 map using the largest Gaussian filters
        gaussian_l1 = F.conv2d(
            l1_map,
            self.g_masks.narrow(dim=0, start=-3, length=3),
            groups=3,
            padding=self.pad
        ).mean(1)

        # Final mixed loss
        mixed_loss = (
            self.alpha * ms_ssim_loss
            + (1 - self.alpha) * gaussian_l1 / self.DR
        )

        mixed_loss = self.compensation * mixed_loss

        return mixed_loss.mean()


# ---------------------------------------------------------
# Total variation loss
# ---------------------------------------------------------
# This loss encourages local smoothness in the generated image.
# It penalizes sudden pixel changes in horizontal and vertical directions.
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()

        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        # Number of values used in vertical and horizontal difference maps
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        # Vertical smoothness penalty
        h_tv = torch.pow(
            x[:, :, 1:, :] - x[:, :, :height - 1, :],
            2
        ).sum()

        # Horizontal smoothness penalty
        w_tv = torch.pow(
            x[:, :, :, 1:] - x[:, :, :, :width - 1],
            2
        ).sum()

        # Final TV loss
        loss = self.TVLoss_weight * 2 * (
            h_tv / count_h + w_tv / count_w
        ) / batch_size

        return loss

    def _tensor_size(self, tensor):
        # Number of elements per image excluding batch dimension
        return tensor.size(1) * tensor.size(2) * tensor.size(3)