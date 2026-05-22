# tensora (Python package)

Core Python package providing framework integrations on top of the Rust extension.

## Modules

| Module | Description |
|--------|-------------|
| `_tensora_rust` | Native extension — low-level load/save functions |
| `torch` | PyTorch `state_dict` loading for SafeTensors and ServerlessLLM |
| `tensorflow` | TensorFlow tensor loading |

## Usage

```python
# PyTorch
from tensora_py.torch import load_safetensors, load_serverlessllm
state_dict = load_safetensors("model_dir", device="cuda")

# TensorFlow
from tensora_py.tensorflow import load_safetensors as tf_load
tensors = tf_load("model_dir")
```
