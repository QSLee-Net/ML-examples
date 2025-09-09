# Overview
Conformer is a popular Transformer based speech recognition network, suitable for embedded devices. This repository contains instructions how to train and quantize a [Conformer](https://github.com/sooftware/conformer) speech recognition model.
For the quantization of the model, we use ExecuTorch with the Arm&reg; Ethos&trade;-U quantizer. 

To train the model, follow the instructions in the `training` folder.
To quantize the model, follow the instructions in the `post_training_quantization` folder.
