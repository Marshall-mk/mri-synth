"""Multi-contrast simulation: contrasts of one subject share an acquisition.

The point of the feature is that stack ``i`` of T1 and stack ``i`` of T2 describe
the *same* physical acquisition — same orientation, thickness, coverage and
tilt — so a reconstruction can use them jointly.
"""

import json

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from mri_synth.cli import app
from mri_synth.fov.resampling import compute_shared_support_mask
from mri_synth.pipeline import HRLRDataGenerator

runner = CliRunner()


def _phantom(n=48, bright_core=True, seed=0):
    g = torch.arange(n, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    r = ((zz - 23) / 14) ** 2 + ((yy - 24) / 12) ** 2 + ((xx - 23) / 13) ** 2
    body = torch.exp(-3.0 * r)
    core = torch.exp(-14.0 * r)
    # Contrast inversion between "T1-like" and "T2-like".
    vol = body - 0.6 * core if bright_core else 0.55 * body + 0.9 * core
    return vol.clamp(min=0).unsqueeze(0).unsqueeze(0)


class TestSharedSupportMask:
    def test_covers_every_input(self):
        a = torch.zeros(1, 16, 16, 16)
        a[:, 2:6, 2:6, 2:6] = 1.0
        b = torch.zeros(1, 16, 16, 16)
        b[:, 8:12, 8:12, 8:12] = 1.0
        shared = compute_shared_support_mask([a, b], threshold=0.5)
        # The prescribed FOV must contain both contrasts' foreground.
        assert shared[:, 2:12, 2:12, 2:12].min() == 1.0
        assert shared.shape == (1, 16, 16, 16)

    def test_is_a_rectangular_slab(self):
        a = torch.zeros(1, 16, 16, 16)
        a[:, 2:6, 2:6, 2:6] = 1.0
        b = torch.zeros(1, 16, 16, 16)
        b[:, 8:12, 8:12, 8:12] = 1.0
        shared = compute_shared_support_mask([a, b], threshold=0.5)
        idx = shared[0].nonzero()
        lo, hi = idx.min(dim=0).values, idx.max(dim=0).values
        expected = int((hi - lo + 1).prod().item())
        assert int(shared.sum().item()) == expected

    def test_rejects_mismatched_grids(self):
        with pytest.raises(ValueError, match="share a spatial grid"):
            compute_shared_support_mask(
                [torch.ones(1, 8, 8, 8), torch.ones(1, 8, 8, 9)]
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one image"):
            compute_shared_support_mask([])


class TestStructuralPlanReuse:
    def _generator(self):
        return HRLRDataGenerator(
            num_stacks=3,
            structural_only=True,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[6.0, 6.0, 6.0],
            fov_augmentation_prob=1.0,
            enable_obliqueness=True,
            prob_obliqueness=1.0,
            obliqueness_range=12.0,
            tight_fov=False,
        )

    def test_same_plan_gives_same_geometry(self):
        gen = self._generator()
        torch.manual_seed(0)
        plan = gen.sample_structural_plan(batch_size=1)

        t1, t2 = _phantom(bright_core=True), _phantom(bright_core=False)

        _, _, res_a, thk_a, _, masks_a = gen.generate_paired_data(
            t1, return_resolution=True, structural_plan=plan
        )
        rot_a = [gen._rotation_angles_per_stack[s][0] for s in range(3)]

        _, _, res_b, thk_b, _, masks_b = gen.generate_paired_data(
            t2, return_resolution=True, structural_plan=plan
        )
        rot_b = [gen._rotation_angles_per_stack[s][0] for s in range(3)]

        for s in range(3):
            assert torch.equal(res_a[s], res_b[s])
            assert torch.equal(thk_a[s], thk_b[s])
            assert rot_a[s] == rot_b[s]
            # Identical geometry -> identical FOV masks, bit for bit.
            assert torch.equal(masks_a[s], masks_b[s])

    def test_without_plan_geometry_is_resampled(self):
        """Sanity check that the plan is what pins the geometry, not luck."""
        gen = self._generator()
        torch.manual_seed(0)
        _, _, res_a, _, _, _ = gen.generate_paired_data(
            _phantom(), return_resolution=True
        )
        _, _, res_b, _, _, _ = gen.generate_paired_data(
            _phantom(), return_resolution=True
        )
        assert not all(torch.equal(res_a[s], res_b[s]) for s in range(3))

    def test_shared_support_keeps_fov_masks_in_sync(self):
        """With tight_fov, a per-contrast bbox would desynchronise the masks."""
        gen = HRLRDataGenerator(
            num_stacks=3,
            structural_only=True,
            min_resolution=[1.0, 1.0, 1.0],
            max_res_aniso=[5.0, 5.0, 5.0],
            fov_augmentation_prob=0.0,
            enable_obliqueness=False,
            tight_fov=True,
        )
        t1, t2 = _phantom(bright_core=True), _phantom(bright_core=False)
        support = compute_shared_support_mask([t1[0], t2[0]], threshold=1e-3)

        torch.manual_seed(1)
        plan = gen.sample_structural_plan(
            batch_size=1, support_masks=support.unsqueeze(0)
        )
        _, _, _, _, _, masks_a = gen.generate_paired_data(
            t1, return_resolution=True, structural_plan=plan
        )
        _, _, _, _, _, masks_b = gen.generate_paired_data(
            t2, return_resolution=True, structural_plan=plan
        )
        for s in range(3):
            assert torch.equal(masks_a[s], masks_b[s])


class TestMultiContrastCLI:
    def _write_subject(self, d, n=32):
        import nibabel as nib

        d.mkdir(parents=True, exist_ok=True)
        g = np.arange(n, dtype=np.float32)
        zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
        r = ((zz - 15) / 9) ** 2 + ((yy - 16) / 8) ** 2 + ((xx - 15) / 9) ** 2
        body, core = np.exp(-3.0 * r), np.exp(-14.0 * r)
        for name, data in [
            ("T1", body - 0.6 * core),
            ("T2", 0.55 * body + 0.9 * core),
        ]:
            nib.save(
                nib.Nifti1Image(data.astype(np.float32), np.eye(4)),
                str(d / f"{name}.nii.gz"),
            )

    def test_contrasts_share_geometry(self, tmp_path):
        self._write_subject(tmp_path / "in" / "sub-01")
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--input", str(tmp_path / "in"),
                "--output-dir", str(out),
                "--multi-contrast",
                "--structural-only",
                "--num-variations", "2",
                "--seed", "5",
            ],
        )
        assert result.exit_code == 0, result.output

        for var in ["variation_000", "variation_001"]:
            meta = json.loads((out / "sub-01" / var / "metadata.json").read_text())
            assert meta["shared_geometry"] is True
            assert set(meta["contrasts"]) == {"T1", "T2"}
            t1, t2 = meta["contrasts"]["T1"], meta["contrasts"]["T2"]
            for a, b in zip(t1["stacks"], t2["stacks"]):
                for key in (
                    "resolution",
                    "thickness",
                    "fov_dropped",
                    "fov_keep_fraction",
                    "obliqueness_deg",
                ):
                    assert a[key] == b[key], f"{key} differs between contrasts"

        assert (out / "sub-01" / "hr_T1.nii.gz").exists()
        assert (out / "sub-01" / "hr_T2.nii.gz").exists()
        assert (out / "sub-01" / "variation_000" / "T1" / "stack_0_axial.nii.gz").exists()

    def test_flat_directory_is_one_subject(self, tmp_path):
        self._write_subject(tmp_path / "session")
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--input", str(tmp_path / "session"),
                "--output-dir", str(out),
                "--multi-contrast",
                "--structural-only",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "session" / "hr_T1.nii.gz").exists()

    def test_mismatched_grids_are_skipped_with_a_message(self, tmp_path):
        import nibabel as nib

        d = tmp_path / "in" / "sub-01"
        d.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.random.rand(24, 24, 24).astype(np.float32), np.eye(4)), str(d / "T1.nii.gz"))
        nib.save(nib.Nifti1Image(np.random.rand(24, 24, 20).astype(np.float32), np.eye(4)), str(d / "T2.nii.gz"))
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--input", str(tmp_path / "in"),
                "--output-dir", str(out),
                "--multi-contrast",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "co-registered" in result.output
        assert not (out / "sub-01" / "hr_T1.nii.gz").exists()

    def test_rejects_single_file(self, tmp_path):
        import nibabel as nib

        f = tmp_path / "solo.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32), np.eye(4)), str(f))
        result = runner.invoke(
            app,
            ["--input", str(f), "--output-dir", str(tmp_path / "out"), "--multi-contrast"],
        )
        assert result.exit_code == 1
