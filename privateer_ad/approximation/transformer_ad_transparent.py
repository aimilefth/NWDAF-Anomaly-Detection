# privateer_ad/approximation/transformer_ad_transparent.py
import torch
from torch import nn
from typing import Optional

from .FxPyTorch.transparent.trans_transformer_encoder import (
    TransformerEncoderLayerTransparent,
)
from .FxPyTorch.transparent.trans_layernorm import LayerNormTransparent
from .FxPyTorch.transparent.trans_linear import LinearTransparent
from ..architectures.layers import PositionalEncoding
from ..config import ModelConfig
from .FxPyTorch.transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)

class TransformerADTransparent(nn.Module):
    """
    Transparent version of the TransformerAD model for activation logging.
    This class mirrors the structure of the base TransformerAD but uses
    transparent layers that support detailed logging of intermediate tensors.
    """

    def __init__(self, config: ModelConfig):
        super(TransformerADTransparent, self).__init__()
        self.config = config
        self.input_size = self.config.input_size
        self.embed_dim = self.config.embed_dim  # Changed from hidden_dim
        self.latent_dim = self.config.latent_dim
        self.dropout = self.config.dropout
        self.num_heads = self.config.num_heads
        self.num_layers = self.config.num_layers
        self.seq_len = self.config.seq_len

        self.embed = LinearTransparent(self.input_size, self.embed_dim)
        self.pos_enc = PositionalEncoding(
            d_model=self.embed_dim, max_seq_length=self.seq_len, dropout=self.dropout
        )
        self.encoder_layer = TransformerEncoderLayerTransparent(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.latent_dim,
            batch_first=True,
            dropout=self.dropout,
            att_dropout=self.dropout,
            bias=True,
            add_bias_kv=True,
            add_bias_q=True,
        )
        self.norm_layer = LayerNormTransparent(self.embed_dim)

        self.compress = nn.Sequential(
            LinearTransparent(self.embed_dim, self.latent_dim), nn.ReLU()
        )
        self.output = LinearTransparent(self.latent_dim, self.input_size)

    def forward(
        self,
        x,
        logger: Optional[ActivationLogger] = None,
    ) -> torch.Tensor:
        with ActivationLoggingScope(logger, type(self).__name__):  # Top-level scope
            if logger:
                logger.log("input_src", x, self)

            x = self.embed(x, logger=logger)

            x = self.pos_enc(x)
            if logger:
                logger.log("pos_encoded_x", x, self.pos_enc)

            x = self.encoder_layer(x, logger=logger)

            x = self.norm_layer(x, logger=logger)

            x = self.compress(x)
            if logger:
                logger.log("compressed_x", x, self.compress)

            x = self.output(x)
            if logger:
                logger.log("output", x, self.output)

        return x

    def load_state_dict(self, state_dict, strict=True):
        """
        Remaps keys from a state_dict of a standard TransformerAD (or the old
        AttentionAutoencoder) into the names used by TransformerADTransparent
        before calling the parent load_state_dict.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Remap the encoder layer keys from the standard nn.TransformerEncoder
            if key.startswith("transformer_encoder.layers.0."):
                # e.g., "transformer_encoder.layers.0.self_attn..." -> "encoder_layer.self_attn..."
                new_key = "encoder_layer." + key[len("transformer_encoder.layers.0.") :]
            elif key.startswith("transformer_encoder.norm."):
                # e.g., "transformer_encoder.norm.weight" -> "norm_layer.weight"
                new_key = "norm_layer." + key[len("transformer_encoder.norm.") :]
            # Other keys like "embed.weight", "compress.0.weight", "output.weight" should match.
            new_state_dict[new_key] = value

        # Now call the parent's load_state_dict with the remapped state dict.
        super(TransformerADTransparent, self).load_state_dict(
            new_state_dict, strict=strict
        )