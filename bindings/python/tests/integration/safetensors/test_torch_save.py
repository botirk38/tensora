"""End-to-end SafeTensors save tests with PyTorch."""

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file

from tensora.torch import load_safetensors as pytorch_load_safetensors
from tensora.torch import save_safetensors as pytorch_save_safetensors


class TestPyTorchSave:
    """Test PyTorch tensor saving to SafeTensors."""

    def test_save_single_tensor(self, tmp_path):
        """Test saving a single tensor."""
        tensors = {"weight": torch.randn(10, 20)}
        path = tmp_path / "single.safetensors"

        pytorch_save_safetensors(tensors, path)
        assert path.exists()

        loaded = load_file(path)
        assert "weight" in loaded
        assert torch.allclose(loaded["weight"], tensors["weight"])

    def test_save_multiple_tensors(self, tmp_path, hidden_dim):
        """Test saving multiple tensors."""
        tensors = {
            "wte": torch.randn(1024, hidden_dim),
            "wpe": torch.randn(128, hidden_dim),
            "ln_1.weight": torch.ones(hidden_dim),
            "ln_1.bias": torch.zeros(hidden_dim),
        }
        path = tmp_path / "multiple.safetensors"

        pytorch_save_safetensors(tensors, path)

        loaded = load_file(path)
        assert set(loaded.keys()) == set(tensors.keys())
        for name, tensor in tensors.items():
            assert torch.allclose(loaded[name], tensor, atol=1e-6)

    def test_save_with_metadata(self, tmp_path):
        """Test saving tensors with metadata."""
        tensors = {"weight": torch.randn(5, 5)}
        metadata = {"model_name": "test", "version": "1.0"}
        path = tmp_path / "meta.safetensors"

        pytorch_save_safetensors(tensors, path, metadata=metadata)

        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            assert f.metadata() == metadata

    def test_save_empty_tensor(self, tmp_path):
        """Test saving an empty tensor."""
        tensors = {"empty": torch.zeros(0)}
        path = tmp_path / "empty.safetensors"

        pytorch_save_safetensors(tensors, path)

        loaded = load_file(path)
        assert "empty" in loaded
        assert loaded["empty"].shape == (0,)

    def test_save_scalar_tensor(self, tmp_path):
        """Test saving a scalar tensor."""
        tensors = {"scalar": torch.tensor(42.0)}
        path = tmp_path / "scalar.safetensors"

        pytorch_save_safetensors(tensors, path)

        loaded = load_file(path)
        assert torch.allclose(loaded["scalar"], tensors["scalar"])

    def test_save_various_dtypes(self, tmp_path):
        """Test saving tensors with various dtypes."""
        tensors = {
            "f32": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "f64": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "i32": torch.tensor([1, 2], dtype=torch.int32),
            "i64": torch.tensor([1, 2], dtype=torch.int64),
            "bool": torch.tensor([True, False], dtype=torch.bool),
        }
        path = tmp_path / "dtypes.safetensors"

        pytorch_save_safetensors(tensors, path)

        loaded = load_file(path)
        for name, tensor in tensors.items():
            assert loaded[name].dtype == tensor.dtype
            assert torch.allclose(loaded[name], tensor)


class TestPyTorchRoundtrip:
    """Test load/save roundtrips with PyTorch."""

    def test_load_save_roundtrip(self, tmp_path):
        """Test that tensors can be saved and loaded back correctly."""
        tensors = {
            "weight1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "weight2": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }
        path = tmp_path / "roundtrip.safetensors"

        pytorch_save_safetensors(tensors, path)
        loaded = load_file(path)

        assert set(loaded.keys()) == {"weight1", "weight2"}
        assert torch.allclose(loaded["weight1"], tensors["weight1"])
        assert torch.allclose(loaded["weight2"], tensors["weight2"])

    def test_large_tensor_preservation(self, tmp_path, hidden_dim):
        """Test that large tensors are preserved correctly."""
        tensors = {
            "large": torch.randn(1024, hidden_dim * 12),
        }
        path = tmp_path / "large.safetensors"

        pytorch_save_safetensors(tensors, path)
        loaded = load_file(path)

        assert loaded["large"].shape == tensors["large"].shape
        assert torch.allclose(loaded["large"], tensors["large"], atol=1e-6)


class TestPyTorchGPUTensors:
    """Test saving GPU tensors (if available)."""

    def test_save_cuda_tensor(self, tmp_path):
        """Test saving a CUDA tensor if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensors = {"weight": torch.randn(10, 20, device="cuda")}
        path = tmp_path / "cuda.safetensors"

        pytorch_save_safetensors(tensors, path)

        loaded = load_file(path)
        assert "weight" in loaded
        assert loaded["weight"].device.type == "cpu"
        assert torch.allclose(loaded["weight"].cpu(), tensors["weight"].cpu())
