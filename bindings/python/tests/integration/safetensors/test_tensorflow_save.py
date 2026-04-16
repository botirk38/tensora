"""End-to-end SafeTensors save tests with TensorFlow."""

import pytest

tf = pytest.importorskip("tensorflow")

from tensora.tensorflow import load_safetensors as tf_load_safetensors
from tensora.tensorflow import save_safetensors as tf_save_safetensors


class TestTensorFlowSave:
    """Test TensorFlow tensor saving to SafeTensors."""

    def test_save_single_tensor(self, tmp_path):
        """Test saving a single tensor."""
        tensors = {"weight": tf.constant([[1.0, 2.0], [3.0, 4.0]])}
        path = tmp_path / "single.safetensors"

        tf_save_safetensors(tensors, path)
        assert path.exists()

        loaded = tf_load_safetensors(path)
        assert "weight" in loaded
        assert tf.reduce_all(tf.equal(loaded["weight"], tensors["weight"]))

    def test_save_multiple_tensors(self, tmp_path, hidden_dim):
        """Test saving multiple tensors."""
        tensors = {
            "wte": tf.random.normal((1024, hidden_dim)),
            "wpe": tf.random.normal((128, hidden_dim)),
            "ln_1_weight": tf.ones((hidden_dim,)),
            "ln_1_bias": tf.zeros((hidden_dim,)),
        }
        path = tmp_path / "multiple.safetensors"

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        assert set(loaded.keys()) == set(tensors.keys())
        for name, tensor in tensors.items():
            assert tf.reduce_all(tf.equal(loaded[name], tensor))

    def test_save_with_metadata(self, tmp_path):
        """Test saving tensors with metadata."""
        tensors = {"weight": tf.constant([[1.0, 2.0], [3.0, 4.0]])}
        metadata = {"model_name": "test", "version": "1.0"}
        path = tmp_path / "meta.safetensors"

        tf_save_safetensors(tensors, path, metadata=metadata)

        from safetensors import safe_open

        with safe_open(path, framework="numpy") as f:
            assert f.metadata() == metadata

    def test_save_empty_tensor(self, tmp_path):
        """Test saving an empty tensor."""
        tensors = {"empty": tf.zeros((0,))}
        path = tmp_path / "empty.safetensors"

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        assert "empty" in loaded
        assert loaded["empty"].shape == (0,)

    def test_save_scalar_tensor(self, tmp_path):
        """Test saving a scalar tensor."""
        tensors = {"scalar": tf.constant(42.0)}
        path = tmp_path / "scalar.safetensors"

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        assert tf.reduce_all(tf.equal(loaded["scalar"], tensors["scalar"]))

    def test_save_various_dtypes(self, tmp_path):
        """Test saving tensors with various dtypes."""
        tensors = {
            "f32": tf.constant([1.0, 2.0], dtype=tf.float32),
            "f64": tf.constant([1.0, 2.0], dtype=tf.float64),
            "i32": tf.constant([1, 2], dtype=tf.int32),
            "i64": tf.constant([1, 2], dtype=tf.int64),
            "bool": tf.constant([True, False]),
        }
        path = tmp_path / "dtypes.safetensors"

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        for name, tensor in tensors.items():
            assert loaded[name].dtype == tensor.dtype
            assert tf.reduce_all(tf.equal(loaded[name], tensor))


class TestTensorFlowRoundtrip:
    """Test load/save roundtrips with TensorFlow."""

    def test_load_save_roundtrip(self, tmp_path):
        """Test that tensors can be saved and loaded back correctly."""
        tensors = {
            "weight1": tf.constant([[1.0, 2.0], [3.0, 4.0]]),
            "weight2": tf.constant([[5.0, 6.0], [7.0, 8.0]]),
        }
        path = tmp_path / "roundtrip.safetensors"

        tf_save_safetensors(tensors, path)
        loaded = tf_load_safetensors(path)

        assert set(loaded.keys()) == {"weight1", "weight2"}
        assert tf.reduce_all(tf.equal(loaded["weight1"], tensors["weight1"]))
        assert tf.reduce_all(tf.equal(loaded["weight2"], tensors["weight2"]))

    def test_large_tensor_preservation(self, tmp_path, hidden_dim):
        """Test that large tensors are preserved correctly."""
        tensors = {
            "large": tf.random.normal((1024, hidden_dim * 12)),
        }
        path = tmp_path / "large.safetensors"

        tf_save_safetensors(tensors, path)
        loaded = tf_load_safetensors(path)

        assert loaded["large"].shape == tensors["large"].shape
        assert tf.reduce_all(tf.abs(loaded["large"] - tensors["large"]) < 1e-5)


class TestTensorFlowGPUTensors:
    """Test saving GPU tensors (if available)."""

    def test_save_gpu_tensor(self, tmp_path):
        """Test saving a GPU tensor if GPU is available."""
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            pytest.skip("GPU not available")

        tensors = {"weight": tf.constant([[1.0, 2.0], [3.0, 4.0]])}
        path = tmp_path / "gpu.safetensors"

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        assert "weight" in loaded
        assert tf.reduce_all(tf.equal(loaded["weight"], tensors["weight"]))


class TestTensorFlowComplex:
    """Test complex tensor handling."""

    def test_save_complex64(self, tmp_path):
        """Test saving complex64 tensors."""
        real = tf.constant([1.0, 2.0])
        imag = tf.constant([3.0, 4.0])
        tensors = {"complex": tf.complex(real, imag)}
        path = tmp_path / "complex.safetensors"

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        assert "complex" in loaded
