# mri-synth

Physics-based MRI simulation pipeline for generating synthetic low-resolution stacks from high-resolution volumes.

## Installation

```bash
pip install -e .
```

**Dependencies:** torch (>=2.0), monai (>=1.3), nibabel (>=5.0), typer (>=0.9), pydantic (>=2.0), pyyaml (>=6.0)

## Quick Start — CLI

Generate synthetic LR stacks from a single HR volume:

```bash
mri-synth -i brain.nii.gz -o output/
```

Process an entire directory:

```bash
mri-synth -i /data/hr_volumes/ -o output/ -n 5 --num-stacks 3
```

### Output structure

```
output/
  brain/
    hr.nii.gz
    variation_000/
      stack_0_axial.nii.gz
      stack_0_axial_fov_mask.nii.gz
      stack_1_coronal.nii.gz
      stack_1_coronal_fov_mask.nii.gz
      stack_2_sagittal.nii.gz
      stack_2_sagittal_fov_mask.nii.gz
      metadata.json
    variation_001/
      ...
  manifest.json
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | (required) | Input NIfTI file or directory |
| `--output-dir`, `-o` | (required) | Output directory |
| `--num-stacks` | `3` | Number of LR stacks per variation |
| `--num-variations`, `-n` | `1` | Number of variations per volume |
| `--psf-type` | `trapezoid` | PSF profile: `boxcar`, `gaussian`, `trapezoid` |
| `--enable-bias-field` / `--no-bias-field` | enabled | Enable bias field corruption |
| `--fov-enable` / `--no-fov-enable` | enabled | Enable FOV simulation |
| `--noise-std` | `0.02` | Noise standard deviation |
| `--min-res` | `1.0 1.0 1.0` | Minimum resolution per axis (3 values) |
| `--max-res-aniso` | `9.0 9.0 9.0` | Maximum anisotropic resolution (3 values) |
| `--obliqueness-range` | `15.0` | Max obliqueness rotation per axis in degrees |
| `--enable-obliqueness` / `--no-obliqueness` | enabled | Simulate oblique acquisitions |
| `--prob-obliqueness` | `0.5` | Probability of applying obliqueness per stack |
| `--tight-fov` / `--no-tight-fov` | enabled | Size LR scan FOV to brain bbox |
| `--tight-fov-threshold` | `1e-3` | Foreground intensity threshold for bbox detection |
| `--tight-fov-margin` | `0` | Voxel margin around the brain bbox |
| `--save-native-res` / `--no-save-native-res` | disabled | Save native-resolution LR stacks (pre-upsample) |
| `--clip-to-unit-range` / `--no-clip-to-unit-range` | enabled | Clip outputs to [0, 1] |
| `--apply-intensity-aug` / `--no-intensity-aug` | disabled | Apply intensity augmentation |
| `--randomise-res` / `--no-randomise-res` | enabled | Randomize acquisition resolution |
| `--return-intermediate` / `--no-return-intermediate` | disabled | Return native-resolution LR (pre-upsample) |
| `--upsample-mode` | `trilinear` | Interpolation mode for upsampling |
| `--device` | `cpu` | Device: `cpu` or `cuda` |
| `--seed` | `None` | Random seed for reproducibility |
| `--config`, `-c` | `None` | YAML config file (overrides CLI flags) |

### Using a YAML config

```yaml
# config.yaml
num_stacks: 3
num_variations: 5
min_resolution: [1.0, 1.0, 1.0]
max_res_aniso: [9.0, 9.0, 9.0]
device: cuda
physics:
  psf_type: trapezoid
  prob_bias_field: 0.5
artifacts:
  noise_std: 0.02
  prob_motion: 0.2
fov:
  enable: true
  prob: 0.7
  min_keep: 0.40
  max_keep: 0.70
  obliqueness_range: 15.0
  enable_obliqueness: true
  prob_obliqueness: 0.5
```

```bash
mri-synth -i /data/hr_volumes/ -o output/ --config config.yaml
```

## Quick Start — Training (Python API)

### Using `create_dataset()` with NIfTI paths

```python
from mri_synth.config import GenerationConfig
from mri_synth.dataset import create_dataset
from mri_synth.pipeline import HRLRDataGenerator

cfg = GenerationConfig(num_stacks=3, device="cuda")
generator = HRLRDataGenerator.from_config(cfg)

dataset = create_dataset(
    image_paths=["sub-01.nii.gz", "sub-02.nii.gz"],
    generator=generator,
    target_shape=[128, 128, 128],
    target_spacing=[1.0, 1.0, 1.0],
    use_cache=True,
)

lr_stacks, hr, orientation_mask, fov_masks = dataset[0]
# lr_stacks: list of N tensors (C, D, H, W)
# hr: tensor (C, D, H, W)
# orientation_mask: bool tensor (num_stacks,)
# fov_masks: list of N binary tensors (1, D, H, W) — 1=missing (out-of-bounds), 0=valid
```

### Using `GeneratorDataset` with a MONAI base dataset

```python
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from mri_synth.dataset import GeneratorDataset

base_dataset = Dataset(
    data=[{"image": p} for p in nifti_paths],
    transform=Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
    ]),
)

dataset = GeneratorDataset(
    base_dataset,
    generator,
    return_resolution=True,
    balanced_orientation_combos=True,
)

lr_stacks, hr, resolutions, thicknesses, orientation_mask, fov_masks = dataset[0]
```

### Using `HRLRDataGenerator` directly

```python
import torch
from mri_synth.pipeline import HRLRDataGenerator

generator = HRLRDataGenerator(
    num_stacks=3,
    min_resolution=[1.0, 1.0, 1.0],
    max_res_aniso=[9.0, 9.0, 9.0],
    psf_profile_type="trapezoid",
)

hr_batch = torch.randn(2, 1, 128, 128, 128)  # (B, C, D, H, W)
lr_stacks, hr_aug, orientation_mask, fov_masks = generator.generate_paired_data(hr_batch)
```

### Balanced orientation combos

When `balanced_orientation_combos=True`, the dataset cycles through orientation dropout patterns (drop one orientation at a time, plus all-present) evenly across the epoch. Use `set_epoch()` to reshuffle each epoch:

```python
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)  # reshuffles orientation schedule
    for batch in dataloader:
        ...
```

For distributed training, call `set_epoch(epoch)` on each rank to keep schedules synchronized.

## Configuration

### `GenerationConfig` fields

| Field | Default | Description |
|---|---|---|
| `num_stacks` | `3` | Number of LR stacks per sample |
| `num_variations` | `1` | Number of variations per volume (CLI mode) |
| `atlas_res` | `[1, 1, 1]` | Resolution of input HR images (mm) |
| `target_res` | `[1, 1, 1]` | Target output resolution (mm) |
| `min_resolution` | `[1, 1, 1]` | Minimum in-plane resolution (mm) |
| `max_res_aniso` | `[9, 9, 9]` | Maximum through-plane resolution (mm) |
| `randomise_res` | `true` | Randomize acquisition resolution |
| `apply_intensity_aug` | `false` | Apply gamma/intensity augmentation |
| `clip_to_unit_range` | `true` | Clip outputs to [0, 1] |
| `upsample_mode` | `trilinear` | Interpolation mode for upsampling |
| `return_intermediate` | `false` | Return native-resolution LR (pre-upsample) |
| `save_native_res` | `false` | Save native-res stacks to disk (CLI) |
| `orientation_dropout_prob` | `0.0` | Probability of dropping orientations |
| `min_orientations` | `1` | Minimum orientations to keep after dropout |
| `drop_orientations` | `null` | List of orientation indices to always drop |
| `device` | `cpu` | Torch device |
| `seed` | `null` | Random seed |

### `PhysicsConfig`

| Field | Default | Description |
|---|---|---|
| `psf_type` | `trapezoid` | Slice profile: `boxcar`, `gaussian`, `trapezoid` |
| `edge_width` | `0.1` | Edge width for trapezoid PSF |
| `bias_field_std` | `0.3` | Bias field intensity |
| `prob_bias_field` | `0.5` | Probability of applying bias field |

### `ArtifactConfig`

| Field | Default | Description |
|---|---|---|
| `prob_motion` | `0.2` | Probability of motion ghosting |
| `prob_spike` | `0.05` | Probability of RF spike artifact |
| `prob_aliasing` | `0.1` | Probability of aliasing |
| `prob_noise` | `0.8` | Probability of adding noise |
| `noise_std` | `0.02` | Noise standard deviation |
| `motion_intensity` | `0.5` | Motion artifact intensity |
| `spike_intensity` | `0.04` | Spike artifact intensity |

### `FOVConfig`

| Field | Default | Description |
|---|---|---|
| `enable` | `true` | Enable FOV simulation |
| `prob` | `0.7` | Probability of FOV cropping per stack |
| `min_keep` | `0.40` | Minimum fraction of slices to keep |
| `max_keep` | `0.70` | Maximum fraction of slices to keep |
| `ensure_coverage` | `true` | Ensure complementary FOV coverage across stacks |
| `force_both_sides` | `true` | Drop slices from both ends |
| `obliqueness_range` | `15.0` | Max rotation per axis in degrees |
| `enable_obliqueness` | `true` | Enable oblique acquisition simulation |
| `prob_obliqueness` | `0.5` | Probability of applying obliqueness per stack |
| `tight_fov` | `true` | Size LR scan FOV to brain bbox (mimics radiographer-sized FOV) |
| `tight_fov_threshold` | `1e-3` | Foreground intensity threshold for brain bbox detection |
| `tight_fov_margin` | `0` | Voxel margin around the brain bbox |

### Loading and saving configs

```python
from mri_synth.config import GenerationConfig

cfg = GenerationConfig.from_yaml("config.yaml")
cfg.physics.psf_type = "gaussian"
cfg.to_yaml("updated_config.yaml")
```

## Simulation Pipeline Overview

The pipeline applies the following steps to each HR volume:

1. **Percentile normalization** — Scale intensities to [0, 1] using 0.5th-99.5th percentiles
2. **Bias field corruption** — Smooth multiplicative field applied once (shared across all stacks)
3. **Intensity/gamma augmentation** — Random gamma and intensity shifts (shared across all stacks)
4. **Per-stack simulation** (repeated for each of N stacks):
   - Orthogonal resolution assignment — cycles through axial (axis 2) -> coronal (axis 1) -> sagittal (axis 0)
   - PSF blurring with configurable slice profile (boxcar/gaussian/trapezoid)
   - K-space artifact injection (motion ghosting, RF spikes, aliasing)
   - FFT-based downsampling via k-space cropping
   - FOV slice drop on native LR (optional, simulates incomplete coverage)
   - Affine-based resampling to HR grid with obliqueness simulation
   - FOV mask generation via the "dummy mask trick"
   - Additive Rician noise

### FOV Mask and Obliqueness

The FOV mask captures the physical difference between the LR stack's native coordinate system and the HR target grid. It is generated using the **support-mask trick**:

1. After FFT downsampling, we have the true LR volume in its native resolution
2. A binary support volume of the same shape is created. By default this is the brain-bbox mask resampled into LR space (see `tight_fov` below); with `tight_fov=False` it is just an all-ones "dummy"
3. Any FOV slice drop is applied to *both* the image and the support, so dropped slabs propagate into the mask
4. Both are resampled to the HR grid using affine-based `grid_sample` (nearest-neighbor for the support)
5. The support is inverted to produce the final mask: **1 where voxels are missing** (out-of-bounds in the LR stack, in the air around the brain bbox, or in a dropped slab), **0 where valid LR data exists**

**Obliqueness simulation** makes the mask non-trivial: small random rotations are applied to each LR stack's affine matrix (with configurable probability and range), simulating the real-world tilt of clinical MRI acquisitions relative to the atlas grid. This creates characteristic triangular empty regions at volume corners after resampling.

**Brain-tight FOV (`tight_fov`)** mirrors what a radiographer does at the scanner: the LR scan FOV is sized around the brain so the slabs barely contain it. We approximate this by computing the axis-aligned bounding box of the foreground in the input HR volume (via MONAI's `generate_spatial_bounding_box`) and using it as the LR scan's support mask. After resampling to the HR grid, HR voxels falling outside the bbox become "new air" and show up in the FOV mask alongside any obliqueness corners. Without `tight_fov`, the FOV mask only captures the obliqueness corners, which understates how much of the HR grid would be missing in a real acquisition.

The mask is most useful when:
- The LR stack is oblique (rotated) relative to the target grid
- The LR scan FOV is smaller than the HR grid (typical with `tight_fov`)
- FOV slice dropping removes edge slices before resampling

The network uses the FOV mask in the loss function to ignore artificial zero-padding and focus only on valid data regions.

## License

MIT
