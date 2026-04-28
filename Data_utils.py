import os
import sys
import random
from pathlib import Path

from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from metrics import *
from Arguments import args


sys.path.extend(["net", ""])


DEFAULT_CROP_SIZE = args.crop_size if args.crop else "whole_img"


class RS_Dataset(data.Dataset):
    """
    Remote sensing paired image dataset for haze-removal training/testing.

    Expected folder format for RSID:
        root/
            hazy/
            GT/
    """

    def __init__(
        self,
        path,
        train,
        size=DEFAULT_CROP_SIZE,
        format=".png",
        hazy="hazy",
        GT="GT"
    ):
        super().__init__()

        self.root = Path(path)
        self.train = train
        self.crop_size = size
        self.image_format = format

        self.hazy_root = self.root / hazy
        self.gt_root = self.root / GT

        self.hazy_images = [
            self.hazy_root / filename
            for filename in os.listdir(self.hazy_root)
        ]

        print("crop size", self.crop_size)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, index):
        hazy_path = self._select_valid_hazy_image(index)

        hazy_img = Image.open(hazy_path)
        clean_img = self._load_matching_gt(hazy_path)

        clean_img = transforms.CenterCrop(hazy_img.size[::-1])(clean_img)

        if isinstance(self.crop_size, int):
            hazy_img, clean_img = self._paired_random_crop(hazy_img, clean_img)

        hazy_img = hazy_img.convert("RGB")
        clean_img = clean_img.convert("RGB")

        hazy_tensor, clean_tensor = self._apply_transforms(hazy_img, clean_img)

        return hazy_tensor, clean_tensor

    def _select_valid_hazy_image(self, index):
        hazy_path = self.hazy_images[index]
        hazy_img = Image.open(hazy_path)

        if isinstance(self.crop_size, int):
            while hazy_img.size[0] < self.crop_size or hazy_img.size[1] < self.crop_size:
                index = random.randint(0, 20000)
                hazy_path = self.hazy_images[index]
                hazy_img = Image.open(hazy_path)

        return hazy_path

    def _load_matching_gt(self, hazy_path):
        image_id = hazy_path.name.split("_")[0]

        # For RSID, the GT filename is directly obtained from the hazy image id.
        gt_name = image_id

        return Image.open(self.gt_root / gt_name)

    def _paired_random_crop(self, hazy_img, clean_img):
        top, left, height, width = transforms.RandomCrop.get_params(
            hazy_img,
            output_size=(self.crop_size, self.crop_size)
        )

        hazy_crop = TF.crop(hazy_img, top, left, height, width)
        clean_crop = TF.crop(clean_img, top, left, height, width)

        return hazy_crop, clean_crop

    def _apply_transforms(self, hazy_img, clean_img):
        if self.train:
            hazy_img, clean_img = self._augment_pair(hazy_img, clean_img)

        hazy_tensor = self.to_tensor(hazy_img)
        hazy_tensor = self.normalize(hazy_tensor)

        clean_tensor = self.to_tensor(clean_img)

        return hazy_tensor, clean_tensor

    def _augment_pair(self, hazy_img, clean_img):
        flip_flag = random.randint(0, 1)
        rotation_id = random.randint(0, 3)

        hazy_img = transforms.RandomHorizontalFlip(flip_flag)(hazy_img)
        clean_img = transforms.RandomHorizontalFlip(flip_flag)(clean_img)

        if rotation_id:
            angle = 90 * rotation_id
            hazy_img = TF.rotate(hazy_img, angle)
            clean_img = TF.rotate(clean_img, angle)

        return hazy_img, clean_img