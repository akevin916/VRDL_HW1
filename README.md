# HW1 Classification Notes

## Project Goal
- Fine-grained image classification task (100 classes).
- Backbone must be ResNet-based.
- Model size must be under 100M parameters.

## Current Model Setup
- Backbone: `SE-ResNet50` (ResNet50-based, with SE blocks).
- Pretrain: ImageNet pretrained weights from `ResNet50_Weights.DEFAULT`.
- Transfer strategy: load ResNet50 pretrained weights into SE-ResNet50 with `strict=False`.
- Classifier head: `num_classes=100`.

## Current Training Decisions
- Keep `strict=False` because architecture is modified from ResNet50 to SE-ResNet50.
- Do not implement full resume checkpoint structure for now.
- Use experiment naming with augmentation and image size tags.
- Keep `README` and `requirements.txt` cleanup as low priority.

## Data/Augmentation Status
- Two augmentation modes are available:
	- `AA`: AutoAugment (ImageNet policy)
	- `manual`: custom augmentation pipeline
- Manual pipeline currently includes:
	- `RandomResizedCrop(scale=(0.85, 1.0))`
	- `RandomHorizontalFlip`
	- `RandomRotation(15)`
	- `RandomAdjustSharpness`
	- `ColorJitter`
	- `RandomErasing` (after `ToTensor`)

## Observed Error Patterns
1. Fine-grained confusion: difficult to separate close sub-species.
2. Background distraction: foreground bee classified as flower class.
3. High intra-class variance: same class appears as stem/fruit/flower with very different visual forms.
4. Low resolution: some samples are too blurry for reliable recognition.

## Key Findings
- Manual augmentation direction is reasonable for this task.
- Compared with AutoAugment, manual policy is easier to control for fine-grained features.
- `224` and `256` gave similar results in practice.
	- `256` was slightly better on val.
	- test performance was almost the same.

## Priority Next Steps (No major architecture change first)
1. Keep backbone as `SE-ResNet50` and optimize training strategy first.
2. Run a clean baseline without mixup/cutmix, then compare.
3. Reduce `ColorJitter` hue strength (suggest `hue=0.03~0.05`) for fine-grained color cues.
4. Add class-imbalance strategy:
	 - `WeightedRandomSampler` or class-weighted CE (start with one).
5. Add TTA at inference (flip / multi-crop logits averaging).
6. If still saturated, then try a larger ResNet-based variant (still <100M), e.g. SE-ResNet101.

## Why No Immediate Architecture Switch
- Current model already satisfies constraints (ResNet-based, under 100M).
- Existing error modes are more likely data/optimization issues than backbone capacity limit.
- Faster gains are expected from sampling, augmentation tuning, and inference stabilization.

## Useful Commands

Train with manual augmentation:

```bash
python train.py --lr 1e-4 --image_size 256 --transform_type manual
```

Train with AutoAugment:

```bash
python train.py --lr 1e-4 --image_size 256 --transform_type AA
```

Inference by experiment name:

```bash
python inference.py --exp_name "<EXP_NAME>" --img_size 256 --output_csv prediction.csv
```

## Notes
- Current `--checkpoint_path` behavior is initialization from checkpoint weights, not full training-state resume.
- If full resume is needed later, save/load `model + optimizer + scheduler + epoch + best_val_acc` together.


## 指令
# 原本 SE
python train.py --model_type se --image_size 256 --transform_type manual --lr 1e-4

# 新的 GeM + Dropout
python train.py --model_type fg_gem --dropout 0.5 --gem_p 3.0 --image_size 256 --transform_type AA --lr 1e-4