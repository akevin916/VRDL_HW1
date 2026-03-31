import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms  # 新增 datasets

from models.resnet50 import get_finegrained_resnet50, get_resnet50, get_se_resnet50


def get_args():
    parser = argparse.ArgumentParser(description="Run inference on test set.")
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (folder in checkpoints/).",
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="Image size for inference."
    )
    parser.add_argument(
        "--test_dir", type=str, default="./data/test", help="Path to test directory."
    )
    parser.add_argument(
        "--output_csv", type=str, default="prediction.csv", help="Output CSV file path."
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Enable test-time augmentation (multi-scale + hflip).",
    )
    parser.add_argument(
        "--tta_scales",
        type=str,
        default="1.0,1.1",
        help="Comma-separated scales for TTA, e.g. '1.0,1.1'.",
    )
    return parser.parse_args()


def parse_tta_scales(scale_text):
    scales = []
    for item in scale_text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("tta_scales values must be > 0")
        scales.append(value)
    if not scales:
        raise ValueError("tta_scales must contain at least one value")
    return scales


def build_tta_transforms(img_size, scales):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tta_transforms = []
    for scale in scales:
        scaled_size = max(1, int(round(img_size * scale)))
        tta_transforms.append(
            transforms.Compose(
                [
                    transforms.Resize((scaled_size, scaled_size)),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        )
    return tta_transforms


def predict():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 取得 cls mapping table ---
    train_data = datasets.ImageFolder(root="./data/train")
    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
    # print("Index to Class Mapping:", idx_to_class)

    model_path = os.path.join("checkpoints", args.exp_name, "last_model.pth")
    model = get_se_resnet50()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if args.use_tta:
        scales = parse_tta_scales(args.tta_scales)
        tta_transforms = build_tta_transforms(args.img_size, scales)
        print(
            f"TTA enabled. scales={scales}, views_per_image={len(tta_transforms) * 2}"
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    results = []
    test_files = sorted(os.listdir(args.test_dir))

    with torch.no_grad():
        for file_name in test_files:
            img_path = os.path.join(args.test_dir, file_name)
            img = Image.open(img_path).convert("RGB")

            if args.use_tta:
                logits_sum = None
                view_count = 0
                for tta_transform in tta_transforms:
                    view = tta_transform(img).unsqueeze(0).to(device)
                    logits = model(view)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                    view_count += 1

                    # Horizontal flip view
                    view_flip = torch.flip(view, dims=[3])
                    logits_flip = model(view_flip)
                    logits_sum = logits_sum + logits_flip
                    view_count += 1

                output = logits_sum / view_count
            else:
                view = test_transform(img).unsqueeze(0).to(device)
                output = model(view)

            _, pred = torch.max(output, 1)

            # --- 關鍵步驟 2：將 Index 轉回資料夾名稱 ---
            actual_class_name = idx_to_class[pred.item()]

            # 儲存正確的類別名稱 (字串)
            results.append(
                {
                    "image_name": os.path.splitext(file_name)[0],
                    "pred_label": actual_class_name,  # 這裡改用轉換後的名稱
                }
            )

    df = pd.DataFrame(results)
    # 確保 pred_label 是字串或正確的 ID 格式
    df.to_csv(args.output_csv, index=False)
    print(f"Prediction finished. Results saved to {args.output_csv}")


if __name__ == "__main__":
    predict()
