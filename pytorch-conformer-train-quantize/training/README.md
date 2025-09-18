# Conformer-S Model Training

This repository provides an example of training the **Conformer-S** model on the **LibriSpeech** dataset.

## External dependencies
- **Model**: https://github.com/sooftware/conformer implementation of Conformer-S
- **Dataset**: LibriSpeech (downloaded via `torchaudio`) - used both to generate Tokenizer and Conformer model
- **Tokenizer**: Generated using https://github.com/google/sentencepiece/
- **Python Dependencies**: Python packages listed in **requirements.txt**.

## Environment description
- AWS g5.24xlarge instance 
- Python version 3.12.7
- AWS AMI - Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)

## Setup
1) Make sure the Conformer repository is cloned in the same directory as the training script:
```angular2html
git clone https://github.com/sooftware/conformer.git
```
2) Generate SentencePiece Tokenizer
- More information on what is SentencePiece tokenizer and how to use it can be found at https://github.com/google/sentencepiece?tab=readme-ov-file#overview
- Generate the tokenizer using the following command
```angular2html
!python build_sp_128_librispeech.py \
  --root ./data \
  --subset train-clean-100 \
  --output_dir ./tokenizer_out \
  --vocab_size 128 \
  --model_type unigram \
  --lowercase \
  --disable_bos_eos \
  --pad_id -1
```
- Pass the tokenizer path to the training script via the --sp-model argument
3) create an empty data folder in the same directory as the training script
## Training
Run the following command to start training:
```angular2html
!CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--train-sets "train-clean-100,train-clean-360,train-other-500" \
--valid-set "dev-clean" \
--epochs 160 \
--batch-size 96 \
--lr=0.0005 \
--betas 0.9,0.98 \
--weight-decay 1e-6 \
--warmup-epochs 2.0 \
--grad-clip 5 \
--root "data" \
--save-dir "checkpoints" \
--num-workers=32 \
--accum-steps 16 \
2>&1 | tee train_log.txt
```
## Notes and recommendations
- Hyperparameter tuning and active monitoring (“model babysitting”) are strongly recommended to achieve optimal performance
- We should be able to reach WER in the range of 6%-7% on the test clean dataset.
- Ckeckpoints will be saved under the checkpoints/ directory
- Logs are written to train_log.txt for convenience