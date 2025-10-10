#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import logging
import os

import torch

from .model import (get_dit_module, load_model,
                    get_autoencoder_decoder_module,
                    get_autoencoder_decoder_example_input,
                    get_conditioners_module,
                    get_conditioners_example_input,
                    get_dit_example_input_mapping)

from stable_audio_tools.models.utils import remove_weight_norm_from_model

from executorch.exir import to_edge_transform_and_lower

from torch.export import export, ExportedProgram

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (XnnpackPartitioner,
                                                                       XnnpackDynamicallyQuantizedPartitioner)

from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def export_conditioners(model, output_path) -> None:
    logging.info("Starting Conditioners Model conversion...\n")
    conditioners = get_conditioners_module(model=model,dtype=torch.float)
    conditioners_example_input = get_conditioners_example_input(seq_length=64, seconds_total=10.0, dtype=torch.float)

    # Export the model to ExecuTorch format
    exported_program: ExportedProgram = torch.export.export(conditioners, conditioners_example_input, dynamic_shapes=None)
    edge: EdgeProgramManager = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    with open(os.path.join(output_path, "conditioners_model.pte"), "wb") as file:
        exec_prog.write_to_file(file)

    logging.info("Finished Conditioners Model conversion.\n")

def export_dit(model, output_path) -> None:
    dit_model = get_dit_module(model=model)
    dit_example_mapping = get_dit_example_input_mapping()

    # Quantize the models' linear layers to int8 per-channel
    logging.info("Starting Dit Model conversion...\n")

    from torchao.quantization.granularity import PerAxis, PerGroup
    from torchao.quantization.quant_api import (
        Int8DynamicActivationIntxWeightConfig,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    with torch.no_grad():
        quantize_(
            dit_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int8,
                weight_granularity=PerAxis(0),
            ),
        )
        dit_model = unwrap_tensor_subclass(dit_model)

    print("quantized model:", dit_model)

    # Export the model to ExecuTorch format
    exported_program: ExportedProgram = torch.export.export(dit_model, args=(), kwargs=dit_example_mapping, dynamic_shapes=None)
    edge: EdgeProgramManager = to_edge_transform_and_lower(
        exported_program,
        partitioner=[
            XnnpackDynamicallyQuantizedPartitioner(),
            XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    with open(os.path.join(output_path, "dit_model.pte"), "wb") as file:
        exec_prog.write_to_file(file)

    logging.info("Finished Dit Model conversion.\n")

def export_autoencoder(model, output_path) -> None:
    # Load the AutoEncoder part of the model
    logging.info("Starting AutoEncoder Decoder conversion...\n")

    # Export the model in fp16, however the input/output is still fp32 for easy of use on application side
    # Casting to fp16 is done inside the model
    autoencoder_decoder_example_input = get_autoencoder_decoder_example_input(dtype=torch.float)
    model.pretransform.model_half=True
    model = model.to(torch.half)

    # Removing weight norm from the model as it is causing issues during export
    remove_weight_norm_from_model(model.pretransform)

    autoencoder_decoder = get_autoencoder_decoder_module(model)
    autoencoder_decoder = autoencoder_decoder.to(torch.half).eval().requires_grad_(False)

    # Export the model to ExecuTorch format
    exported_program: ExportedProgram = torch.export.export(autoencoder_decoder, autoencoder_decoder_example_input, dynamic_shapes=None)
    edge: EdgeProgramManager = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    with open(os.path.join(output_path, "autoencoder_model.pte"), "wb") as file:
        exec_prog.write_to_file(file)

    logging.info("Finished AutoEncoder Model conversion.\n")

def export(args) -> None:

    torch.manual_seed(0)
    device = torch.device("cpu")

    # Load the model configuration
    logging.info("Loading the AudioGen Checkpoint...")
    with open(args.model_config, encoding="utf-8") as f:
        model_config = json.load(f)
    model, model_config = load_model(
        model_config,
        args.ckpt_path,
        pretrained_name=None,
        device=device,
    )
    logging.info("Model is loaded...")

    # --------- Conditioners Model ---------
    export_conditioners(model, args.output_path)

    # --------- Dit Model ----------------
    export_dit(model, args.output_path)

    # --------- AutoEncoder Model ---------
    export_autoencoder(model, args.output_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        help="Path to the model configuration file.",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the model checkpoint file.",
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the model checkpoint file.",
        default=".",
        required=False,
    )

    export(parser.parse_args())

if __name__ == "__main__":
    main()
