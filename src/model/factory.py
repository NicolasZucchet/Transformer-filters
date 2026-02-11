"""Model factory."""
from typing import Any
import jax.numpy as jnp
import flax.linen as nn
from src.model.config import ModelConfig
from src.data.config import DataConfig
from src.model.backbone import TransformerBackbone
from src.model.layers import EmbeddingInput, LinearInput, PatchingInput
from src.model.heads import ClassificationHead, RegressionHead


class SequenceModel(nn.Module):
    """Composes input layer + backbone + output head."""
    input_layer: nn.Module
    backbone: TransformerBackbone
    head: nn.Module
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x, mask=None, deterministic=True,
        init_cache=False, cache=None, decode_step=None, max_seq_len=None,
    ):
        x = self.input_layer(x, decode=(decode_step is not None))
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        backbone_out = self.backbone(
            x, mask=mask, deterministic=deterministic,
            init_cache=init_cache, cache=cache,
            decode_step=decode_step, max_seq_len=max_seq_len,
        )

        if init_cache or cache is not None:
            x, new_caches = backbone_out
        else:
            x = backbone_out

        logits = self.head(x)

        if init_cache or cache is not None:
            return logits, new_caches
        return logits


def _get_dtype(dtype_str: str):
    return {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}[dtype_str]


def create_model(model_config: ModelConfig, data_config: DataConfig) -> SequenceModel:
    """Create a SequenceModel from config."""
    dtype = _get_dtype(model_config.dtype)
    mlp_dim = model_config.model_dim * model_config.mlp_coefficient

    backbone = TransformerBackbone(
        num_heads=model_config.num_heads,
        model_dim=model_config.model_dim,
        num_layers=model_config.num_layers,
        mlp_dim=mlp_dim,
        dropout_rate=model_config.dropout_rate,
        positional_encoding=model_config.positional_encoding,
        max_seq_len=model_config.max_seq_len,
        activation=model_config.activation,
        dtype=dtype,
    )

    if data_config.task_type == "bigram":
        input_layer = EmbeddingInput(
            vocab_size=data_config.vocab_size,
            model_dim=model_config.model_dim,
            dtype=dtype,
        )
        head = ClassificationHead(vocab_size=data_config.vocab_size, dtype=dtype)
    elif data_config.task_type in ("lds", "physics"):
        input_layer = PatchingInput(
            model_dim=model_config.model_dim,
            patch_size=model_config.patch_size,
            dtype=dtype,
        )
        head = RegressionHead(
            output_dim=model_config.patch_size * data_config.dim_y,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown task_type: {data_config.task_type}")

    return SequenceModel(
        input_layer=input_layer,
        backbone=backbone,
        head=head,
        dropout_rate=model_config.dropout_rate,
        dtype=dtype,
    )
