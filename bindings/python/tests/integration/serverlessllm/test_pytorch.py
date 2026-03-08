"""End-to-end ServerlessLLM integration tests with actual PyTorch computations."""

import pytest
import torch

from tensor_store_py import (
    load_serverlessllm,
    load_serverlessllm_mmap,
    load_serverlessllm_sync,
)


FIXED_SEED = 42


def _seeded_tensor(shape, seed):
    """Create a deterministic tensor for testing."""
    torch.manual_seed(seed)
    return torch.randn(shape)


# =============================================================================
# Embedding Lookup Tests
# =============================================================================

def test_embedding_lookup_sync(serverlessllm_dir):
    """Test embedding lookup with sync-loaded weights."""
    torch.manual_seed(FIXED_SEED)
    weights = load_serverlessllm_sync(serverlessllm_dir)
    wte = weights["wte"]
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    embedded = torch.nn.functional.embedding(input_ids, wte)
    assert embedded.shape == (1, 5, 768)


def test_embedding_lookup_mmap(serverlessllm_dir):
    """Test embedding lookup with mmap-loaded weights."""
    torch.manual_seed(FIXED_SEED)
    weights = load_serverlessllm_mmap(serverlessllm_dir)
    wte = weights["wte"]
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    embedded = torch.nn.functional.embedding(input_ids, wte)
    assert embedded.shape == (1, 5, 768)


@pytest.mark.asyncio
async def test_embedding_lookup_async(serverlessllm_dir):
    """Test embedding lookup with async-loaded weights."""
    torch.manual_seed(FIXED_SEED)
    weights = await load_serverlessllm(serverlessllm_dir)
    wte = weights["wte"]
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    embedded = torch.nn.functional.embedding(input_ids, wte)
    assert embedded.shape == (1, 5, 768)


# =============================================================================
# Attention Computation Tests
# =============================================================================

def _attention_computation(weights, layer_idx=0):
    """Simple attention computation for testing."""
    c_attn = weights[f"h.{layer_idx}.attn.c_attn.weight"]
    c_proj = weights[f"h.{layer_idx}.attn.c_proj.weight"]
    x = torch.randn(1, 10, 768)
    qkv = torch.matmul(x, c_attn)
    q, k, v = qkv.split(768, dim=-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    output = torch.matmul(attn_out, c_proj)
    return output


def test_attention_computation_sync(serverlessllm_dir):
    """Test attention computation with sync-loaded weights."""
    weights = load_serverlessllm_sync(serverlessllm_dir)
    output = _attention_computation(weights)
    assert output.shape == (1, 10, 768)


def test_attention_computation_mmap(serverlessllm_dir):
    """Test attention computation with mmap-loaded weights."""
    weights = load_serverlessllm_mmap(serverlessllm_dir)
    output = _attention_computation(weights)
    assert output.shape == (1, 10, 768)


@pytest.mark.asyncio
async def test_attention_computation_async(serverlessllm_dir):
    """Test attention computation with async-loaded weights."""
    weights = await load_serverlessllm(serverlessllm_dir)
    output = _attention_computation(weights)
    assert output.shape == (1, 10, 768)


# =============================================================================
# Full Layer Forward Tests
# =============================================================================

def _full_layer_forward(weights, layer_idx=0):
    """Full transformer layer: LayerNorm + Attention + LayerNorm + MLP."""
    ln1_w = weights[f"h.{layer_idx}.ln_1.weight"]
    ln1_b = weights[f"h.{layer_idx}.ln_1.bias"]
    c_attn_w = weights[f"h.{layer_idx}.attn.c_attn.weight"]
    c_attn_b = weights[f"h.{layer_idx}.attn.c_attn.bias"]
    c_proj_w = weights[f"h.{layer_idx}.attn.c_proj.weight"]
    c_proj_b = weights[f"h.{layer_idx}.attn.c_proj.bias"]
    ln2_w = weights[f"h.{layer_idx}.ln_2.weight"]
    ln2_b = weights[f"h.{layer_idx}.ln_2.bias"]
    mlp_fc_w = weights[f"h.{layer_idx}.mlp.c_fc.weight"]
    mlp_fc_b = weights[f"h.{layer_idx}.mlp.c_fc.bias"]
    mlp_proj_w = weights[f"h.{layer_idx}.mlp.c_proj.weight"]
    mlp_proj_b = weights[f"h.{layer_idx}.mlp.c_proj.bias"]

    x = torch.randn(2, 5, 768, requires_grad=True)

    ln1 = torch.nn.LayerNorm(768, eps=1e-5)
    ln1.weight.data = ln1_w
    ln1.bias.data = ln1_b
    x = ln1(x)

    attn_c = torch.nn.Linear(768, 2304)
    attn_c.weight.data = c_attn_w.T
    attn_c.bias.data = c_attn_b
    attn_proj = torch.nn.Linear(768, 768)
    attn_proj.weight.data = c_proj_w.T
    attn_proj.bias.data = c_proj_b
    qkv = attn_c(x)
    q, k, v = qkv.split(768, dim=-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    x = x + attn_proj(attn_out)

    ln2 = torch.nn.LayerNorm(768, eps=1e-5)
    ln2.weight.data = ln2_w
    ln2.bias.data = ln2_b
    x = ln2(x)

    mlp_fc = torch.nn.Linear(768, 3072)
    mlp_fc.weight.data = mlp_fc_w.T
    mlp_fc.bias.data = mlp_fc_b
    mlp_proj = torch.nn.Linear(3072, 768)
    mlp_proj.weight.data = mlp_proj_w.T
    mlp_proj.bias.data = mlp_proj_b
    x = x + mlp_proj(torch.nn.functional.gelu(mlp_fc(x)))

    return x


def test_full_layer_forward_sync(serverlessllm_dir):
    """Test full layer forward with sync-loaded weights."""
    output = _full_layer_forward(load_serverlessllm_sync(serverlessllm_dir))
    assert output.shape == (2, 5, 768)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_full_layer_forward_mmap(serverlessllm_dir):
    """Test full layer forward with mmap-loaded weights."""
    output = _full_layer_forward(load_serverlessllm_mmap(serverlessllm_dir))
    assert output.shape == (2, 5, 768)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.asyncio
async def test_full_layer_forward_async(serverlessllm_dir):
    """Test full layer forward with async-loaded weights."""
    output = _full_layer_forward(await load_serverlessllm(serverlessllm_dir))
    assert output.shape == (2, 5, 768)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


# =============================================================================
# Gradient Flow Tests
# =============================================================================

def _gradient_test(weights, layer_idx=0):
    """Test that gradients flow through the computation."""
    c_attn = weights[f"h.{layer_idx}.attn.c_attn.weight"].requires_grad_(True)
    c_proj = weights[f"h.{layer_idx}.attn.c_proj.weight"].requires_grad_(True)
    x = torch.randn(1, 10, 768, requires_grad=True)
    qkv = torch.matmul(x, c_attn)
    q, k, v = qkv.split(768, dim=-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    output = torch.matmul(attn_out, c_proj)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert c_attn.grad is not None
    assert c_proj.grad is not None
    return x.grad, c_attn.grad, c_proj.grad


def test_gradient_flow_sync(serverlessllm_dir):
    """Test gradient flow with sync-loaded weights."""
    x_grad, c_attn_grad, c_proj_grad = _gradient_test(
        load_serverlessllm_sync(serverlessllm_dir)
    )
    assert x_grad.shape == (1, 10, 768)
    assert c_attn_grad.shape == (768, 2304)
    assert c_proj_grad.shape == (768, 768)


def test_gradient_flow_mmap(serverlessllm_dir):
    """Test gradient flow with mmap-loaded weights."""
    x_grad, c_attn_grad, c_proj_grad = _gradient_test(
        load_serverlessllm_mmap(serverlessllm_dir)
    )
    assert x_grad.shape == (1, 10, 768)
    assert c_attn_grad.shape == (768, 2304)
    assert c_proj_grad.shape == (768, 768)


@pytest.mark.asyncio
async def test_gradient_flow_async(serverlessllm_dir):
    """Test gradient flow with async-loaded weights."""
    x_grad, c_attn_grad, c_proj_grad = _gradient_test(
        await load_serverlessllm(serverlessllm_dir)
    )
    assert x_grad.shape == (1, 10, 768)
    assert c_attn_grad.shape == (768, 2304)
    assert c_proj_grad.shape == (768, 768)
