# NYCU Computer Vision 2026 HW1

- Student ID: 314553039
- Name: 蔡博崴

## Introduction

This project tackles a 100-class fine-grained image classification task under the constraint that the model must remain ResNet-based and under 100M parameters. The final system is built on SE-ResNet50 with ImageNet pretraining, and the training pipeline includes data augmentation, weighted sampling, and optional test-time augmentation.

## Environment Setup

Install the required Python packages before training or inference.

```bash
pip install -r requirements.txt
```

## Usage

### Training

The default final model is SE-ResNet50. The command below trains with AutoAugment at 256 resolution.

```bash
python train.py --model_type se --image_size 256 --transform_type AA --lr 1e-4
```

If you want to enable weighted sampling for class imbalance:

```bash
python train.py --model_type se --image_size 256 --transform_type AA --lr 1e-4 --use_weighted_sampler --sampler_power 1.0
```

If you want to test the GeM + Dropout variant:

```bash
python train.py --model_type fg_gem --dropout 0.5 --gem_p 3.0 --image_size 256 --transform_type AA --lr 1e-4
```

### Inference

Run inference with the selected experiment name. The checkpoint is loaded from `checkpoints/<EXP_NAME>/best_model.pth`.

```bash
python inference.py --exp_name "<EXP_NAME>" --img_size 256 --output_csv prediction.csv
```

To enable TTA during inference:

```bash
python inference.py --exp_name "<EXP_NAME>" --img_size 256 --use_tta --tta_scales "1.0,1.1" --output_csv prediction_tta.csv
```

## Performance Snapshot
![alt text](image.png)

## Notes

- `strict=False` is used when loading pretrained ResNet50 weights into SE-ResNet50 because the architecture is modified.
- `WeightedRandomSampler` is used to reduce class-imbalance errors.
- TTA is available in inference through multi-scale and horizontal-flip averaging.
- `checkpoint_path` currently loads model weights only and does not restore full optimizer or scheduler state.