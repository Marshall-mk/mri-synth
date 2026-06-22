"""Tests for CLI."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from mri_synth.cli import app
from mri_synth.config import GenerationConfig

runner = CliRunner()


def _create_test_nifti(path: Path, shape=(32, 32, 32)):
    """Create a minimal NIfTI file for testing."""
    import nibabel as nib

    data = np.random.rand(*shape).astype(np.float32)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, str(path))


class TestCLI:
    def test_single_file(self, tmp_path):
        nifti_path = tmp_path / "test.nii.gz"
        _create_test_nifti(nifti_path)
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "--input", str(nifti_path),
                "--output-dir", str(output_dir),
                "--num-stacks", "3",
                "--num-variations", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (output_dir / "manifest.json").exists()

    def test_directory_input(self, tmp_path):
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        for i in range(2):
            _create_test_nifti(input_dir / f"vol_{i}.nii.gz")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "--input", str(input_dir),
                "--output-dir", str(output_dir),
                "--num-variations", "1",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_yaml_config(self, tmp_path):
        cfg = GenerationConfig(num_stacks=3, num_variations=1)
        cfg.fov.enable = False
        cfg_path = tmp_path / "config.yaml"
        cfg.to_yaml(str(cfg_path))

        nifti_path = tmp_path / "test.nii.gz"
        _create_test_nifti(nifti_path)
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "--input", str(nifti_path),
                "--output-dir", str(output_dir),
                "--config", str(cfg_path),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_structural_only_flag(self, tmp_path):
        nifti_path = tmp_path / "test.nii.gz"
        _create_test_nifti(nifti_path)
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "--input", str(nifti_path),
                "--output-dir", str(output_dir),
                "--num-stacks", "3",
                "--num-variations", "1",
                "--structural-only",
                "--seed", "0",
            ],
        )
        assert result.exit_code == 0, result.output

        meta_path = output_dir / "test" / "variation_000" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["structural_only"] is True
        # No appearance corruption may be applied in geometry-only mode.
        assert not any(meta["applied_artifacts"].values())
        # Enriched per-stack metadata must be present.
        for stack in meta["stacks"]:
            assert "fov_dropped" in stack
            assert "obliqueness_deg" in stack

    def test_nonexistent_input(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "--input", "/nonexistent/path.nii.gz",
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
