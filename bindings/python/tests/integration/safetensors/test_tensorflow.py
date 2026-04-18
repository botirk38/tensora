"""End-to-end SafeTensors integration tests with TensorFlow."""

import pytest

tf = pytest.importorskip("tensorflow")

from tensora.tensorflow import load_safetensors as tf_load_safetensors


FIXED_SEED = 42


def _seeded_tensor(shape, seed):
    """Create a deterministic tensor for testing."""
    tf.random.set_seed(seed)
    return tf.random.normal(shape)


def _create_reference_tensors(hidden_dim):
    """Create reference tensors matching the fixture structure."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(FIXED_SEED)
    return {
        "wte": torch.randn(1024, hidden_dim),
        "wpe": torch.randn(128, hidden_dim),
        "ln_1.weight": torch.ones(hidden_dim),
        "ln_1.bias": torch.zeros(hidden_dim),
        "attn.c_attn.weight": torch.randn(hidden_dim * 2, hidden_dim),
        "attn.c_attn.bias": torch.zeros(hidden_dim * 2),
        "attn.c_proj.weight": torch.randn(hidden_dim, hidden_dim),
        "attn.c_proj.bias": torch.zeros(hidden_dim),
    }


def _attention_computation(weights, layer_idx=0, hidden_dim=128):
    """Simple attention computation for testing."""
    c_attn = weights[f"h.{layer_idx}.attn.c_attn.weight"]
    c_proj = weights[f"h.{layer_idx}.attn.c_proj.weight"]
    x = tf.random.normal((1, 10, hidden_dim))
    qkv = tf.linalg.matmul(x, c_attn)
    q, k, v = tf.split(qkv, 3, axis=-1)
    attn_scores = tf.matmul(q, k, transpose_b=True) / 8.0
    attn_weights = tf.nn.softmax(attn_scores, axis=-1)
    attn_out = tf.matmul(attn_weights, v)
    output = tf.linalg.matmul(attn_out, c_proj)
    return output


class TestTensorFlowLoad:
    """Test TensorFlow tensor loading from SafeTensors files."""

    def test_load_safetensors_sync_basic(self, safetensors_path):
        """Test basic sync loading with TensorFlow."""
        weights = tf_load_safetensors(safetensors_path)
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for name, tensor in weights.items():
            assert isinstance(tensor, tf.Tensor), f"{name} is not a tf.Tensor"

    def test_load_safetensors_async_basic(self, safetensors_path):
        """Test basic async loading with TensorFlow."""
        weights = tf_load_safetensors(safetensors_path, backend="async")
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for name, tensor in weights.items():
            assert isinstance(tensor, tf.Tensor), f"{name} is not a tf.Tensor"

    def test_tensor_shapes(self, safetensors_path, hidden_dim):
        """Test that tensor shapes match PyTorch loads for the same file."""
        from tensora.torch import load_safetensors as pytorch_load_safetensors

        pt_weights = pytorch_load_safetensors(safetensors_path)
        tf_weights = tf_load_safetensors(safetensors_path)
        for name, pt in pt_weights.items():
            assert name in tf_weights
            assert tuple(tf_weights[name].shape) == tuple(pt.shape)

    def test_attention_computation_sync(self, safetensors_path, hidden_dim):
        """Test attention computation with sync-loaded TensorFlow weights."""
        tf_weights = tf_load_safetensors(safetensors_path)
        output = _attention_computation(tf_weights, hidden_dim=hidden_dim)
        assert output.shape == (1, 10, hidden_dim)

    def test_attention_computation_async(self, safetensors_path, hidden_dim):
        """Test attention computation with async-loaded TensorFlow weights."""
        tf_weights = tf_load_safetensors(safetensors_path, backend="async")
        output = _attention_computation(tf_weights, hidden_dim=hidden_dim)
        assert output.shape == (1, 10, hidden_dim)


class TestTensorFlowDtypes:
    """Test dtype handling for TensorFlow."""

    def test_float32(self, safetensors_path_dtypes):
        """Test float32 tensor loading."""
        weights = tf_load_safetensors(safetensors_path_dtypes)
        if "f32" in weights:
            assert weights["f32"].dtype == tf.float32

    def test_int64(self, safetensors_path_dtypes):
        """Test int64 tensor loading."""
        weights = tf_load_safetensors(safetensors_path_dtypes)
        if "i64" in weights:
            assert weights["i64"].dtype == tf.int64

    def test_bool(self, safetensors_path_dtypes):
        """Test bool tensor loading."""
        weights = tf_load_safetensors(safetensors_path_dtypes)
        if "bool" in weights:
            assert weights["bool"].dtype == tf.bool


class TestTensorFlowRoundtrip:
    """Test save/load roundtrips with TensorFlow."""

    def test_save_load_roundtrip(self, tmp_path, hidden_dim):
        """Test that tensors can be saved and loaded back correctly."""
        tensors = {
            "weight1": tf.constant([[1.0, 2.0], [3.0, 4.0]]),
            "weight2": tf.constant([[5.0, 6.0], [7.0, 8.0]]),
        }
        path = tmp_path / "test.safetensors"

        from tensora.tensorflow import save_safetensors as tf_save_safetensors

        tf_save_safetensors(tensors, path)

        loaded = tf_load_safetensors(path)
        assert set(loaded.keys()) == {"weight1", "weight2"}
        assert tf.reduce_all(tf.equal(loaded["weight1"], tensors["weight1"]))
        assert tf.reduce_all(tf.equal(loaded["weight2"], tensors["weight2"]))
