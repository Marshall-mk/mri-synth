"""Tests for I/O module."""

import json
from pathlib import Path

import numpy as np
import torch

from mri_synth.io import (
    create_output_structure,
    get_stack_filename,
    load_volume,
    save_volume,
    write_manifest,
    write_metadata,
)


class TestNIfTIIO:
    def test_save_load_roundtrip(self, tmp_path):
        tensor = torch.rand(1, 16, 16, 16)
        affine = np.eye(4)
        path = tmp_path / "test.nii.gz"

        save_volume(tensor, path, affine)
        loaded, loaded_affine, _ = load_volume(path)

        assert loaded.shape == tensor.shape
        np.testing.assert_array_almost_equal(loaded_affine, affine)
        # Values should be close (float32 round-trip)
        assert torch.allclose(loaded, tensor, atol=1e-5)

    def test_save_3d_tensor(self, tmp_path):
        tensor = torch.rand(16, 16, 16)
        path = tmp_path / "test3d.nii.gz"
        save_volume(tensor, path)
        loaded, _, _ = load_volume(path)
        # load_volume adds channel dim
        assert loaded.shape == (1, 16, 16, 16)

    def test_default_affine(self, tmp_path):
        tensor = torch.rand(1, 8, 8, 8)
        path = tmp_path / "test_default.nii.gz"
        save_volume(tensor, path)
        _, affine, _ = load_volume(path)
        np.testing.assert_array_equal(affine, np.eye(4))


class TestOutputStructure:
    def test_create_structure(self, tmp_path):
        dirs = create_output_structure(tmp_path, "vol001", 3)
        assert dirs["volume_dir"].exists()
        assert len(dirs["variation_dirs"]) == 3
        for vd in dirs["variation_dirs"]:
            assert vd.exists()

    def test_stack_filename(self):
        assert get_stack_filename(0) == "stack_0_axial.nii.gz"
        assert get_stack_filename(1) == "stack_1_coronal.nii.gz"
        assert get_stack_filename(2) == "stack_2_sagittal.nii.gz"
        # Cycles for N > 3
        assert get_stack_filename(3) == "stack_3_axial.nii.gz"


class TestMetadata:
    def test_write_metadata(self, tmp_path):
        meta = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "meta.json"
        write_metadata(path, meta)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == meta

    def test_write_manifest(self, tmp_path):
        manifest = {"volumes": [{"name": "test"}]}
        write_manifest(tmp_path, manifest)
        with open(tmp_path / "manifest.json") as f:
            loaded = json.load(f)
        assert loaded == manifest
