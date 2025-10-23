# privateer_ad/approximation/transformer_ad_fxp.py
import torch
from torch import nn
import copy
from typing import Optional, Literal, Union, Type

from ..architectures.layers import PositionalEncoding
from ..config import ModelConfig
from .transformer_ad_transparent import TransformerADTransparent
from .FxPyTorch.transparent.activation_logger import (
    ActivationLogger,
    ActivationLoggingScope,
)
from .FxPyTorch.fxp.symmetric_quant import (
    QType,
    QConfig,
    apply_quantize,
    QMethod,
)
from pydantic import Field
from .FxPyTorch.fxp.fxp_dropout import FxPDropout, DropoutQConfig
from .FxPyTorch.fxp.fxp_softmax import FxPSoftmax, SoftmaxQConfig
from .FxPyTorch.fxp.fxp_linear import FxPLinear, LinearQConfig
from .FxPyTorch.fxp.fxp_layernorm import FxPLayerNorm, LayerNormQConfig
from .FxPyTorch.fxp.fxp_transformer_encoder import (
    FxPTransformerEncoderLayer,
    TransformerEncoderLayerQConfig,
)
from .FxPyTorch.fxp.fxp_multiheadattention import MultiheadAttentionQConfig
from .FxPyTorch.fxp.calibration import (
    set_calibrated_activation_quant,
    CalibrationType,
)


class FxPTransformerADConfig(ModelConfig):
    # These are module **classes** (constructors), not instances:
    dropout_fun: type[nn.Module] = Field(default=FxPDropout, exclude=True)
    softmax_fun: type[nn.Module] = Field(default=FxPSoftmax, exclude=True)
    layernorm_fun: type[nn.Module] = Field(default=FxPLayerNorm, exclude=True)

    # Let Pydantic accept these arbitrary types
    model_config = {**ModelConfig.model_config, 'arbitrary_types_allowed': True}

    def to_base_config(self) -> ModelConfig:
        # Don’t serialize the function-class fields
        return ModelConfig(**self.model_dump(exclude={'dropout_fun',
                                                      'softmax_fun',
                                                      'layernorm_fun'}))


class TransformerADQConfig(QConfig):
    layer_type: Literal["transformer_encoder_layer"] = "transformer_encoder_layer"
    input: QType = Field(default_factory=QType)
    embed: LinearQConfig = Field(default_factory=LinearQConfig)
    pos_enc: QType = Field(default_factory=QType)
    encoder_layer: TransformerEncoderLayerQConfig = Field(
        default_factory=TransformerEncoderLayerQConfig
    )
    norm_layer: LayerNormQConfig = Field(default_factory=LayerNormQConfig)
    compress_linear: LinearQConfig = Field(default_factory=LinearQConfig)
    output: LinearQConfig = Field(default_factory=LinearQConfig)


class FxpTransformerAD(TransformerADTransparent):
    """Fixed-point quantized version of the TransformerAD model."""

    def __init__(
        self,
        config: FxPTransformerADConfig,
        q_config: TransformerADQConfig = None,
    ):
        super(FxpTransformerAD, self).__init__(config.to_base_config())
        self._q_config = q_config
        if self._q_config is not None:
            self.dropout_fun = config.dropout_fun
            self.softmax_fun = config.softmax_fun
            self.layernorm_fun = config.layernorm_fun

            self.embed = FxPLinear(
                self.input_size, self.embed_dim, q_config=self._q_config.embed
            )
            self.pos_enc = PositionalEncoding(
                d_model=self.embed_dim,
                max_seq_length=self.seq_len,
                dropout=self.dropout,
            )
            self.encoder_layer = FxPTransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.latent_dim,
                batch_first=True,
                dropout=self.dropout,
                att_dropout=self.dropout,
                bias=True,
                add_bias_kv=True,
                add_bias_q=True,
                dropout_fun=self.dropout_fun,
                softmax_fun=self.softmax_fun,
                layernorm_fun=self.layernorm_fun,
                q_config=self._q_config.encoder_layer,
            )
            self.norm_layer = FxPLayerNorm(
                self.embed_dim, q_config=self._q_config.norm_layer
            )
            self.compress = nn.Sequential(
                FxPLinear(
                    self.embed_dim,
                    self.latent_dim,
                    q_config=self._q_config.compress_linear,
                ),
                nn.ReLU(),
            )
            self.output = FxPLinear(
                self.latent_dim, self.input_size, q_config=self._q_config.output
            )

    @property
    def q_config(self) -> TransformerADQConfig:
        return self._q_config

    @q_config.setter
    def q_config(self, new_q_config: TransformerADQConfig):
        if self._q_config is None:
            self._q_config = new_q_config
        else:
            self._q_config.input = new_q_config.input
            self._q_config.pos_enc = new_q_config.pos_enc
        # Re‑wire each submodule so it holds the same nested config objects
        self.embed.q_config = new_q_config.embed
        self.encoder_layer.q_config = new_q_config.encoder_layer
        self.norm_layer.q_config = new_q_config.norm_layer
        self.compress[0].q_config = new_q_config.compress_linear
        self.output.q_config = new_q_config.output

    def forward(
        self,
        x,
        logger: Optional[ActivationLogger] = None,
        apply_ste: bool = True,
        calibrate: bool = False,
        calibration_type: Union[str, CalibrationType] = CalibrationType.NO_OVERFLOW,
    ) -> torch.Tensor:
        if self._q_config is None:
            # Floating Point
            return super(FxpTransformerAD, self).forward(x, logger)
        with ActivationLoggingScope(logger, type(self).__name__):  # Top-level scope
            if calibrate:
                set_calibrated_activation_quant(
                    x, self._q_config.input, calibration_type
                )
            input_quant = apply_quantize(x, self._q_config.input, apply_ste)
            if logger:
                logger.log("input_src", x, self)
                logger.log("input_src_quant", input_quant, self)

            x = self.embed(
                input_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )

            x = self.pos_enc(x)
            if calibrate:
                set_calibrated_activation_quant(
                    x, self._q_config.pos_enc, calibration_type
                )
            pos_enc_quant = apply_quantize(x, self._q_config.pos_enc, apply_ste)
            if logger:
                logger.log("pos_encoded", x, self.pos_enc)
                logger.log("pos_encoded_quant", pos_enc_quant, self)

            x = self.encoder_layer(
                pos_enc_quant,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )

            x = self.norm_layer(
                x,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )

            x = self.compress[0](
                x,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            x = self.compress[1](x)
            if logger:
                logger.log("compressed_x", x, self.compress)

            x = self.output(
                x,
                logger=logger,
                apply_ste=apply_ste,
                calibrate=calibrate,
                calibration_type=calibration_type,
            )
            if logger:
                logger.log("output", x, self.output)

        return x

    def quantize_weights_bias(self) -> None:
        self.embed.quantize_weights_bias()
        self.encoder_layer.quantize_weights_bias()
        self.norm_layer.quantize_weights_bias()
        self.compress[0].quantize_weights_bias()
        self.output.quantize_weights_bias()

    def set_high_precision_quant(
        self,
        same_ff: bool = False,
        same_all_attn: bool = False,
        same_qkv: bool = False,
        same_wb: bool = False,
    ) -> None:
        self.embed.set_high_precision_quant(same_wb)
        self.encoder_layer.set_high_precision_quant(
            same_ff=same_ff,
            same_all_attn=same_all_attn,
            same_qkv=same_qkv,
            same_wb=same_wb,
        )
        self.norm_layer.set_high_precision_quant(same_wb)
        self.compress[0].set_high_precision_quant(same_wb)
        self.output.set_high_precision_quant(same_wb)

    def set_no_overflow_quant(
        self,
        same_ff: bool = False,
        same_all_attn: bool = False,
        same_qkv: bool = False,
        same_wb: bool = False,
    ) -> None:
        self.embed.set_no_overflow_quant(same_wb)
        self.encoder_layer.set_no_overflow_quant(
            same_ff=same_ff,
            same_all_attn=same_all_attn,
            same_qkv=same_qkv,
            same_wb=same_wb,
        )
        self.norm_layer.set_no_overflow_quant(same_wb)
        self.compress[0].set_no_overflow_quant(same_wb)
        self.output.set_no_overflow_quant(same_wb)


def create_dynamic_qconfig(
    weight_bits: int,
    activation_bits: int,
    q_method: QMethod = QMethod.ROUND_SATURATE,
) -> TransformerADQConfig:
    """
    Helper function to create an FxPTransformerADConfig with specified
    total_bits for weights and activations. Fractional bits shall be determined later 
   (e.g. by set_no_overflow_quant for weights and calibration for activations.)
    """

    # Define base QTypes for weights and activations
    # For weights, only total_bits is set
    w_qtype = QType(total_bits=weight_bits, q_method=q_method)
    # For activations, total_bits is set
    a_qtype = QType(total_bits=activation_bits, q_method=q_method)
    x2a_qtype = QType(total_bits=2 * activation_bits, q_method=q_method)
    # --- LinearQConfig template ---
    linear_qc = LinearQConfig(
        input=QType(),  # Input to linear layer
        weight=copy.deepcopy(w_qtype),
        bias=copy.deepcopy(w_qtype),  # Assuming bias exists and uses same bits
        activation=copy.deepcopy(
            a_qtype
        ),  # Output of linear layer (before non-linearity if separate)
    )

    # --- LayerNormQConfig template ---
    ln_qc = LayerNormQConfig(
        input=QType(),
        weight=copy.deepcopy(w_qtype),  # Gamma
        bias=copy.deepcopy(w_qtype),  # Beta
        mean_tensor=copy.deepcopy(a_qtype),
        var_tensor=copy.deepcopy(x2a_qtype),
        input_normalized=copy.deepcopy(x2a_qtype),
        activation=copy.deepcopy(a_qtype),  # Output of LayerNorm
    )

    # --- DropoutQConfig template ---
    dropout_qc = DropoutQConfig(input=QType(), activation=copy.deepcopy(a_qtype))

    softmax_qc = SoftmaxQConfig(
        input=QType(),  # Input to softmax (e.g., attention scores)
        activation=copy.deepcopy(a_qtype),  # Output of softmax (probabilities)
    )

    # --- MultiheadAttentionQConfig template ---
    mha_qc = MultiheadAttentionQConfig(
        input_query=QType(),
        input_key=QType(),
        input_value=QType(),
        qlinear=copy.deepcopy(linear_qc),
        klinear=copy.deepcopy(linear_qc),
        vlinear=copy.deepcopy(linear_qc),
        q_scaled=copy.deepcopy(a_qtype),
        attn_scores_raw=copy.deepcopy(a_qtype),
        softmax=copy.deepcopy(softmax_qc),
        dropout=copy.deepcopy(dropout_qc),  # Dropout on attention weights * V
        attn_output=copy.deepcopy(
            a_qtype
        ),  # Output of attention mechanism (before out_proj)
        out_proj=copy.deepcopy(linear_qc),  # Final projection
    )

    # --- TransformerEncoderLayerQConfig template ---
    tel_qc = TransformerEncoderLayerQConfig(
        input=QType(),
        norm1=copy.deepcopy(ln_qc),
        self_attn=copy.deepcopy(mha_qc),
        self_attn_dropout=copy.deepcopy(dropout_qc),  # Dropout after MHA + residual
        residual_1=copy.deepcopy(a_qtype),
        norm2=copy.deepcopy(ln_qc),
        linear1=copy.deepcopy(linear_qc),  # FFN linear 1
        ff_activation=copy.deepcopy(a_qtype),  # Activation in FFN (e.g. ReLU output)
        dropout1=copy.deepcopy(dropout_qc),  # Dropout after FFN activation
        linear2=copy.deepcopy(linear_qc),  # FFN linear 2
        dropout2=copy.deepcopy(dropout_qc),  # Dropout after FFN linear 2
        residual_2=copy.deepcopy(a_qtype),
    )

    # --- TransformerADQConfig ---
    ae_qconfig = TransformerADQConfig(
        input=copy.deepcopy(a_qtype),  # Input to the whole AE
        embed=copy.deepcopy(linear_qc),  # Embedding layer
        pos_enc=copy.deepcopy(a_qtype),  # Output of positional encoding
        encoder_layer=copy.deepcopy(tel_qc),  # The transformer encoder layer
        norm_layer=copy.deepcopy(ln_qc),  # Final LayerNorm in encoder
        compress_linear=copy.deepcopy(linear_qc),  # Linear layer in compression block
        # Note: compress has nn.ReLU() which is not explicitly quantized here.
        # Its input will be `compress_linear.activation`.
        # Its output will be the input to `output` linear layer.
        output=copy.deepcopy(linear_qc),  # Final output linear layer
    )
    return ae_qconfig