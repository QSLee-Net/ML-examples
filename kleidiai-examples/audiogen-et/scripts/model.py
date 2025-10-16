#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.models.utils import copy_state_dict

DEVICE = torch.device("cpu")

logging.basicConfig(level=logging.INFO)

## Model loading
def load_model(
    model_config: Optional[Dict[str, Any]] = None,
    model_ckpt_path: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    pretransform_ckpt_path: Optional[str] = None,
    device: torch.device = DEVICE,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load the AudioGen model and its configuration.

    Either a pretrained model (via `pretrained_name`) or a freshly constructed one
    (via `model_config` + `model_ckpt_path`) will be loaded.

    Args:
        model_config: Configuration dict for creating the model.
        model_ckpt_path: Path to a model checkpoint file.
        pretrained_name: Name of a model to load from the repo.
        pretransform_ckpt_path: Optional path to a pretransform checkpoint.
        device: Torch device to map the model to.

     Returns:
        A tuple of (model, model_config), where `model` is in eval mode
        and cast to float, and `model_config` contains sample_rate/size, etc.
    """

    if pretrained_name is not None:
        logging.info("Loading pretrained model: %s", pretrained_name)
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None:
        if model_ckpt_path is None:
            raise ValueError(
                "model_ckpt_path must be provided when specifying model_config"
            )
        logging.info("Creating model from config")
        model = create_model_from_config(model_config)

        logging.info("Loading model checkpoint from: %s", model_ckpt_path)
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        logging.info("Done loading model checkpoint")

    SAMPLE_RATE = model_config["sample_rate"]
    SAMPLE_SIZE = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        logging.info("Loading pretransform checkpoint from %r", pretransform_ckpt_path)
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(pretransform_ckpt_path), strict=False
        )
        logging.info("Done loading pretransform.")

    model.to(device).eval().requires_grad_(False)
    model = model.to(torch.float)

    return model, model_config


## ----------------- Conditioners Utility Functions -------------------
def get_conditioners(model):
    """Load the conditioners module from Stable Audio Open Small model.
    Args:
        model: Stable Audio Open Small model.
    Returns:
        sao_t5_cond: The T5 encoder.
        sao_seconds_total_cond: The seconds_total conditioner.
    """
    cond_model = model.conditioner
    t5_cond = cond_model.conditioners["prompt"]
    seconds_total_cond = cond_model.conditioners["seconds_total"]

    return t5_cond, seconds_total_cond

## ----------------- Wrapper Class -------------------
class ExportableNumberConditioner(torch.nn.Module):
    """NumberConditioner Module. Take a list of floats,
    normalizes them for a given range, and returns a list of embeddings.
    """

    def __init__(
        self,
        numberConditioner,
    ):
        super(ExportableNumberConditioner, self).__init__()

        self.min_val = numberConditioner.min_val
        self.max_val = numberConditioner.max_val

        self.embedder = numberConditioner.embedder

    def forward(self, floats: torch.tensor) -> Any:
        floats = floats.clamp(self.min_val, self.max_val)

        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)

        return float_embeds, torch.ones(float_embeds.shape[0], 1)

class ConditionersModule(torch.nn.Module):
    """Conditioners Module. Takes the T5 encoder and seconds_total conditioner,
    and returns the cross-attention inputs and global conditioning inputs.
    """

    def __init__(
        self,
        sao_t5_cond: torch.nn.Module,
        sao_seconds_total_cond: torch.nn.Module,
        dtype: torch.dtype = torch.float
    ):
        super().__init__()
        self.sao_t5 = sao_t5_cond
        self.sao_seconds_total_cond = ExportableNumberConditioner(
            sao_seconds_total_cond
        )
        self.dtype = dtype

        # Use float
        self.sao_t5 = (
            self.sao_t5.to("cpu").to(dtype).eval().requires_grad_(False)
        )
        self.sao_seconds_total_cond = self.sao_seconds_total_cond.to(dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seconds_total: torch.Tensor,
    ):
        # Get the projections and conditioner results
        with torch.no_grad():
            t5_embeddings = self.sao_t5.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]
            # Resize the embeddings and attention mask to 64 to match DiT model
            t5_embeddings = t5_embeddings[:, :64, :]
            attention_mask = attention_mask[:, :64]
            # Get the T5 projections
            t5_proj = self.sao_t5.proj_out(t5_embeddings).to(dtype=self.dtype)
            t5_proj = t5_proj * attention_mask.unsqueeze(-1).to(dtype=self.dtype)
            t5_mask = attention_mask

        # Get seconds_total conditioner results
        seconds_total_embedding, seconds_total_mask = self.sao_seconds_total_cond(
            seconds_total
        )

        # Concatenate all cross-attention inputs (t5_embedding, seconds_total) over the sequence dimension
        # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
        cross_attention_input = torch.cat(
            [
                t5_proj,
                seconds_total_embedding,
            ],
            dim=1,
        )
        cross_attention_masks = torch.cat(
            [
                t5_mask,
                seconds_total_mask.to(torch.long),
            ],
            dim=1,
        )

        # Concatenate all global conditioning inputs (seconds_start, seconds_total) over the channel dimension
        # Assumes that the global conditioning inputs are of shape (batch, channels)
        global_cond = torch.cat(
            [
                seconds_total_embedding
            ],
            dim=-1,
        )
        global_cond = global_cond.squeeze(1)

        return cross_attention_input, cross_attention_masks, global_cond

def get_conditioners_module(model, dtype = torch.float):
    """
    Wrap both the T5 encoder and seconds_total conditioner in a single module.
    """
    # Load the SAO conditioners
    sao_t5_cond, sao_seconds_total_cond = get_conditioners(model)

    # Return the conditioners module
    return ConditionersModule(
        sao_t5_cond=sao_t5_cond,
        sao_seconds_total_cond=sao_seconds_total_cond,
        dtype=dtype
    )

def get_conditioners_example_input(seconds_total: float, seq_length: int, dtype=torch.float):
    """Provide example input tensors for the AudioGen Conditioners submodule.
    Args:
        seconds_total (float): The total seconds for the audio.
        seq_length (int): The sequence length for the T5 encoder.
    Returns:
        input_ids (torch.Tensor): The input IDs tensor for the T5 encoder.
        attention_mask (torch.Tensor): The attention mask tensor for the T5 encoder.
        seconds_total (torch.Tensor): The seconds_total tensor.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    encoded = tokenizer(
        text="birds singing in the morning",
        truncation=True,
        max_length=seq_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Create the input_ids and attention_mask tensors for sao conditioners
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Create the seconds_total tensor
    seconds_total = torch.tensor([seconds_total], dtype=dtype)

    return (
        input_ids,
        attention_mask,
        seconds_total,
    )

## ----------------- Utility Functions DiT -------------------
def get_dit_example_input_mapping(dtype=torch.float):
    """Provide example input tensors for the DiT model as a dictionary.
    Args:
        dtype (torch.dtype): The data type for the input tensors.
    Returns:
        dict: A dictionary containing the example input tensors for the DiT model.
        x (torch.Tensor): The input tensor for the DiT model.
        t (torch.Tensor): The time tensor for the DiT model.
        cross_attn_cond (torch.Tensor): The cross attention conditioning tensor for the DiT model. Output of the Conditioner T5 Encoder.
        global_cond (torch.Tensor): The global conditioning tensor for the DiT model. Output of the Conditioner Number Encoder.
    """
    return {
        "x": torch.rand(size=(1, 64, 256), dtype=dtype, requires_grad=False),  # x
        "t": torch.tensor([0.154], dtype=dtype, requires_grad=False),  # t
        "cross_attn_cond": torch.rand(
            size=(1, 65, 768), dtype=dtype, requires_grad=False
        ),  # cross_attn_cond
        "global_cond": torch.rand(size=(1, 768), dtype=dtype, requires_grad=False),  # global_cond
    }

def get_dit_module(model, dtype = torch.float32):
    dit_model = model.model
    dit_model = dit_model.to(dtype).eval().requires_grad_(False)
    return dit_model


## ----------------- Utility Functions AutoEncoder -------------------
def get_autoencoder_decoder_module(model):
    """Get the AutoEncoder module from the AudioGen model."""
    return AutoEncoderDecoderModule(model.pretransform)

def get_autoencoder_decoder_example_input(dtype=torch.float):
    """Get example input for the AutoEncoder module."""
    return (torch.rand((1, 64, 256), dtype=torch.float),)

class AutoEncoderDecoderModule(torch.nn.Module):
    """Wrap the AutoEncoder Module. Takes the AutoEncoder and returns the audio.
    Args:
        autoencoder (torch.nn.Module): The AutoEncoder module.
    Returns:
        audio (torch.Tensor): The decoded audio tensor.
    """

    def __init__(self, autoencoder):
        super(AutoEncoderDecoderModule, self).__init__()
        self.autoencoder = autoencoder

        # Use Half
        self.autoencoder = (
            self.autoencoder.to(dtype=torch.half).eval().requires_grad_(False)
        )

    def forward(self, sampled: torch.Tensor):
        sampled = sampled.to(torch.half)
        sampled_uncompressed = self.autoencoder.decode(sampled)

        audio = rearrange(sampled_uncompressed, "b d n -> d (b n)")
        audio = audio.to(torch.float)
        return audio
