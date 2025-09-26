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

from model import (get_dit_module, load_model,
                    get_autoencoder_decoder_module,
                    get_autoencoder_decoder_example_input,
                    get_conditioners_module,
                    get_conditioners_example_input,
                    get_dit_example_input_tuple)

from executorch.exir import to_edge_transform_and_lower

from torch.export import export, ExportedProgram

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (XnnpackPartitioner,
                                                                       XnnpackDynamicallyQuantizedPartitioner) 

from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def export(args) -> None:

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

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

    ## --------- AutoEncoder Model ---------
    # Load the AutoEncoder part of the model
    logging.info("Starting AutoEncoder Decoder conversion...\n")
    autoencoder_decoder = get_autoencoder_decoder_module(model)
    autoencoder_decoder = autoencoder_decoder.to(dtype).eval().requires_grad_(False)
    autoencoder_decoder_example_input = get_autoencoder_decoder_example_input(dtype)

    # Export the model to ExecuTorch format
    exported_program: ExportedProgram = torch.export.export(autoencoder_decoder, autoencoder_decoder_example_input, dynamic_shapes=None)
    edge: EdgeProgramManager = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    with open("xnnpack_autoencoder.pte", "wb") as file:
        exec_prog.write_to_file(file)

    logging.info("Finished AutoEncoder Model conversion.\n")

    ## --------- Conditioners Model ---------
    # Load the Conditioners part of the model
    logging.info("Starting Conditioners Model conversion...\n")
    conditioners = get_conditioners_module(model=model)
    conditioners_example_input = get_conditioners_example_input(seq_length=128, seconds_total=10.0)

    # Export the model to ExecuTorch format
    exported_program: ExportedProgram = torch.export.export(conditioners, conditioners_example_input, dynamic_shapes=None)
    edge: EdgeProgramManager = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],)
    exec_prog = edge.to_executorch()

    with open("xnnpack_conditioners.pte", "wb") as file:
        exec_prog.write_to_file(file)

    logging.info("Finished Conditioners Model conversion.\n")

    ## --------- Dit Model ----------------
    dit_model = get_dit_module(model=model)
    dit_example_input = get_dit_example_input_tuple()

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
    exported_program: ExportedProgram = torch.export.export(dit_model, dit_example_input, dynamic_shapes=None, )
    edge: EdgeProgramManager = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackDynamicallyQuantizedPartitioner(),XnnpackPartitioner()],
    )
    exec_prog = edge.to_executorch()

    with open("xnnpack_dit_int8.pte", "wb") as file:
        exec_prog.write_to_file(file)

    logging.info("Finished Dit Model conversion.\n")


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
        default="./sao_model.pte",
        required=False,
    )

    export(parser.parse_args())

if __name__ == "__main__":
    main()
