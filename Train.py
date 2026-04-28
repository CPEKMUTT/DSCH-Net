import os
import math
import random

import numpy as np
import torch
import torch.nn as nn

from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from torchvision.models import vgg16

from option import opt
from data_utils import RS_Dataset
from metrics import ssim, psnr
from perceptual import LossNetwork
from DSCH_Net import DSCH_Net_t

def set_random_seed(seed=1234):
    # Python random seed
    random.seed(seed)

    # Hash seed for more reproducible Python behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA seed for single and multi-GPU training
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These settings make training more deterministic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosine_decay_lr(current_epoch, total_epochs, initial_lr=opt.lr):
    cosine_value = math.cos(current_epoch * math.pi / total_epochs)
    lr = 0.5 * (1 + cosine_value) * initial_lr
    return lr

def train_one_epoch(model, train_loader, optimizer, criterion, perceptual_net, scaler):
    perceptual_weight = 0.04

    torch.cuda.empty_cache()
    model.train()
    epoch_losses = []

    progress_bar = tqdm(train_loader)

    for hazy_img, clean_img in progress_bar:
        # Move input and target images to selected device
        hazy_img = hazy_img.to(opt.device)
        clean_img = clean_img.to(opt.device)

        # Always clear gradients before backward
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast():
            restored_img = model(hazy_img)

            pixel_loss = criterion(restored_img, clean_img)
            feature_loss = perceptual_net(restored_img, clean_img)
            total_loss = pixel_loss + perceptual_weight * feature_loss

        # Backward pass with AMP scaling
        scaler.scale(total_loss).backward()

        # Optional gradient clipping.
        # For AMP, unscale before clipping.
        if opt.clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)

        # Optimizer update
        scaler.step(optimizer)

        # Update scaler for next iteration
        scaler.update()

        # Save current loss value
        epoch_losses.append(total_loss.item())

        # Show live training status
        progress_bar.set_description(
            "Total Loss: {:.5f}, L1 Loss: {:.5f}, Perceptual Loss: {:.5f}".format(
                total_loss.item(),
                pixel_loss.item(),
                (perceptual_weight * feature_loss).item()
            )
        )

    return np.mean(epoch_losses)



def evaluate_model(model, test_loader):
    # Put model in evaluation mode
    model.eval()

    # Clear CUDA cache before validation
    torch.cuda.empty_cache()

    ssim_scores = []
    psnr_scores = []

    with torch.no_grad():
        for hazy_img, clean_img in test_loader:
            hazy_img = hazy_img.to(opt.device)
            clean_img = clean_img.to(opt.device)

            restored_img = model(hazy_img)

            batch_ssim = ssim(restored_img, clean_img).item()
            batch_psnr = psnr(restored_img, clean_img)

            ssim_scores.append(batch_ssim)
            psnr_scores.append(batch_psnr)

    mean_ssim = np.mean(ssim_scores)
    mean_psnr = np.mean(psnr_scores)

    return mean_ssim, mean_psnr


def build_dataloaders():
    train_dir = os.path.join(opt.dataset_dir, opt.train)
    test_dir = os.path.join(opt.dataset_dir, opt.test)

    train_dataset = RS_Dataset(
        train_dir,
        train=True,
        format=".png"
    )

    test_dataset = RS_Dataset(
        test_dir,
        train=False,
        size="whole img",
        format=".png"
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.bs,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_loader, test_loader


def build_perceptual_network():
    # Use first 16 layers of VGG16 feature extractor
    vgg_features = vgg16(pretrained=True).features[:16]
    vgg_features = vgg_features.to(opt.device)

    # VGG is only used as a fixed feature extractor.
    for param in vgg_features.parameters():
        param.requires_grad = False

    perceptual_net = LossNetwork(vgg_features)
    perceptual_net = perceptual_net.to(opt.device)
    perceptual_net.eval()

    return perceptual_net


def build_model():
    model = DSCH_Net_t()
    model = model.to(opt.device)

    trainable_params = sum(
        param.nelement()
        for param in model.parameters()
        if param.requires_grad
    )

    print("Total_params: ==> {:.4f} M".format(trainable_params / 1e6))

    if opt.device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    return model


# ---------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------
def save_checkpoint(model, epoch, best_psnr, best_ssim):
    os.makedirs(opt.model_dir, exist_ok=True)

    save_name = "DSCH-Net_RSID_{}.pk".format(epoch)
    save_path = os.path.join(opt.model_dir, save_name)

    torch.save(
        {
            "model": model.state_dict()
        },
        save_path
    )

    print(
        "\nModel saved at epoch: {} | max_psnr: {:.4f} | max_ssim: {:.4f}".format(
            epoch,
            best_psnr,
            best_ssim
        )
    )

if __name__ == "__main__":
    # Fix seed for reproducible experiments
    set_random_seed(seed=1234)

    print("Batch size:", opt.bs)
    print("Device:", opt.device)
    print("Dataset:", opt.dataset_dir)

    # Prepare data
    loader_train, loader_test = build_dataloaders()

    # Prepare model
    net = build_model()

    # Loss functions
    pixel_criterion = nn.L1Loss().to(opt.device)

    # Perceptual loss network
    loss_network = build_perceptual_network()

    # AMP scaler
    scaler = GradScaler()

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs,
        eta_min=opt.lr * 1e-2
    )

    # Print model summary
    summary(net, depth=5)

    # Best validation scores
    best_ssim = 0.0
    best_psnr = 0.0

    # Training loop
    for epoch in tqdm(range(opt.epochs + 1)):
        torch.cuda.empty_cache()

        avg_train_loss = train_one_epoch(
            model=net,
            train_loader=loader_train,
            optimizer=optimizer,
            criterion=pixel_criterion,
            perceptual_net=loss_network,
            scaler=scaler
        )

        scheduler.step()

        # Validate after selected interval
        if epoch % opt.eval_step == 0:
            val_ssim, val_psnr = evaluate_model(net, loader_test)

            print(
                "\nepoch: {} | train_loss: {:.5f} | ssim: {:.4f} | psnr: {:.4f}".format(
                    epoch,
                    avg_train_loss,
                    val_ssim,
                    val_psnr
                )
            )

            # Save only when both metrics improve
            if val_ssim > best_ssim and val_psnr > best_psnr:
                best_ssim = val_ssim
                best_psnr = val_psnr

                save_checkpoint(
                    model=net,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    best_ssim=best_ssim
                )