"""PyTorch Dataset wrappers for training use."""

from pathlib import Path
from typing import List, Optional

import torch
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

from mri_synth.pipeline import HRLRDataGenerator


class GeneratorDataset(torch.utils.data.Dataset):
    """
    Wrapper that applies HRLRDataGenerator to a base dataset.

    Returns N orthogonal low-resolution stacks for each HR volume.
    """

    def __init__(
        self,
        base_dataset,
        generator: HRLRDataGenerator,
        return_resolution: bool = False,
        balanced_orientation_combos: bool = False,
    ):
        self.base_dataset = base_dataset
        self.generator = generator
        self.return_resolution = return_resolution
        self.balanced_orientation_combos = balanced_orientation_combos
        self._orientation_combo_schedule = None
        self._epoch = 0
        self._last_built_epoch = None
        self._shared_epoch = None

        if self.balanced_orientation_combos:
            import multiprocessing as mp

            self._shared_epoch = mp.Value("i", 0)
            self._build_orientation_combo_schedule()

    def _build_orientation_combo_schedule(self) -> None:
        import random

        num_stacks = self.generator.num_stacks
        # Build all valid combos: keep at least 2 out of num_stacks
        combos = []
        if num_stacks >= 3:
            # Standard combos for 3+ stacks
            for i in range(num_stacks):
                combo = torch.ones(num_stacks, dtype=torch.bool)
                combo[i] = False
                combos.append(combo)
            combos.append(torch.ones(num_stacks, dtype=torch.bool))  # all
        else:
            combos.append(torch.ones(num_stacks, dtype=torch.bool))

        total = len(self.base_dataset)
        base_count = total // len(combos)
        remainder = total % len(combos)

        schedule = []
        for combo in combos:
            schedule.extend([combo.clone() for _ in range(base_count)])
        for i in range(remainder):
            schedule.append(combos[i].clone())

        rng = random.Random(self._epoch)
        rng.shuffle(schedule)
        self._orientation_combo_schedule = schedule
        self._last_built_epoch = self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if self._shared_epoch is not None:
            self._shared_epoch.value = epoch
        if self.balanced_orientation_combos:
            self._build_orientation_combo_schedule()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.balanced_orientation_combos and self._shared_epoch is not None:
            shared_epoch = self._shared_epoch.value
            if self._last_built_epoch != shared_epoch:
                self._epoch = shared_epoch
                self._build_orientation_combo_schedule()

        data = self.base_dataset[idx]
        hr_image = data["image"]

        sample_info = None
        if hasattr(self.base_dataset, "data") and idx < len(self.base_dataset.data):
            volume_path = self.base_dataset.data[idx].get("image", "")
            volume_name = Path(volume_path).stem if volume_path else f"sample_{idx}"
            sample_info = {"volume_name": [volume_name]}

        if hr_image.ndim == 3:
            hr_image = hr_image.unsqueeze(0)
        hr_image = hr_image.unsqueeze(0)  # (1, C, D, H, W)

        result = self.generator.generate_paired_data(
            hr_image,
            return_resolution=self.return_resolution,
            sample_info=sample_info,
        )

        if self.return_resolution:
            lr_stacks, hr_augmented, resolutions, thicknesses, orientation_mask, fov_masks = result
            if self.balanced_orientation_combos:
                orientation_mask = self._orientation_combo_schedule[idx]
            return (
                [stack.squeeze(0) for stack in lr_stacks],
                hr_augmented.squeeze(0),
                [res.squeeze(0) for res in resolutions],
                [thick.squeeze(0) for thick in thicknesses],
                orientation_mask.squeeze(0),
                [mask.squeeze(0) for mask in fov_masks],
            )
        else:
            lr_stacks, hr_augmented, orientation_mask, fov_masks = result
            if self.balanced_orientation_combos:
                orientation_mask = self._orientation_combo_schedule[idx]
            return (
                [stack.squeeze(0) for stack in lr_stacks],
                hr_augmented.squeeze(0),
                orientation_mask.squeeze(0),
                [mask.squeeze(0) for mask in fov_masks],
            )


def create_dataset(
    image_paths: List[str],
    generator: HRLRDataGenerator,
    target_shape: Optional[List[int]] = None,
    target_spacing: Optional[List[float]] = None,
    use_cache: bool = False,
    return_resolution: bool = False,
    is_training: bool = True,
    balanced_orientation_combos: bool = False,
):
    """
    Create a training dataset using HRLRDataGenerator.

    Args:
        image_paths: List of paths to NIfTI volumes.
        generator: Configured HRLRDataGenerator instance.
        target_shape: Target spatial shape for cropping/padding.
        target_spacing: Target voxel spacing for resampling.
        use_cache: If True, use MONAI CacheDataset.
        return_resolution: If True, return resolution info.
        is_training: If True, use random crops; else center crops.
        balanced_orientation_combos: If True, balance orientation combos.
    """
    data_dicts = [{"image": img_path} for img_path in image_paths]

    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
    ]

    if target_spacing is not None:
        transforms.append(
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear")
        )

    if target_shape is not None:
        transforms.append(CropForegroundd(keys=["image"], source_key="image"))
        if is_training:
            transforms.append(
                RandSpatialCropd(
                    keys=["image"], roi_size=target_shape, random_size=False
                )
            )
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))
        else:
            transforms.append(
                CenterSpatialCropd(keys=["image"], roi_size=target_shape)
            )
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))

    if is_training:
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                shear_range=None,
                translate_range=(5, 5, 5),
                mode="bilinear",
                padding_mode="border",
            )
        )

    transforms.append(ToTensord(keys=["image"]))
    transform = Compose(transforms)

    if use_cache:
        dataset = CacheDataset(
            data=data_dicts, transform=transform, cache_rate=1.0, num_workers=4
        )
    else:
        dataset = Dataset(data=data_dicts, transform=transform)

    return GeneratorDataset(
        dataset,
        generator,
        return_resolution,
        balanced_orientation_combos=balanced_orientation_combos,
    )
