import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resnet50 import get_se_resnet50


def save_confusion_heatmap(matrix, class_names, save_path, title):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 100 classes are too dense for readable ticks; keep axes clean in that case.
    if len(class_names) <= 30:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    else:
        ax.set_xlabel("Predicted Label Index")
        ax.set_ylabel("True Label Index")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def analyze_validation(model_path, val_dir, train_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = datasets.ImageFolder(root=train_dir)
    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
    num_classes = len(idx_to_class)
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = get_se_resnet50(num_classes=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    analysis_results = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    print("Starting Validation Analysis...")
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(val_loader)):
            image = image.to(device)
            file_path = val_dataset.imgs[i][0]
            file_name = os.path.basename(file_path)
            
            output = model(image)
            
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            true_idx = label.item()
            pred_idx = pred.item()
            confusion[true_idx, pred_idx] += 1
            
            true_class = idx_to_class[true_idx]
            pred_class = idx_to_class[pred_idx]
            is_correct = (true_class == pred_class)
            
            analysis_results.append({
                "file_name": file_name,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_label": true_class,
                "pred_label": pred_class,
                "confidence": conf.item(),
                "is_correct": is_correct
            })

    # 儲存並分析
    df = pd.DataFrame(analysis_results)
    val_analysis_path = os.path.join(output_dir, "val_analysis.csv")
    df.to_csv(val_analysis_path, index=False)
    
    print("\n--- Analysis Summary ---")
    print(f"Overall Accuracy: {df['is_correct'].mean()*100:.2f}%")
    
    # Top 10 Overconfident Mistakes
    wrong_df = df[df['is_correct'] == False].sort_values(by='confidence', ascending=False)
    print("\nTop 10 Overconfident Mistakes (Potential Hard Examples):")
    print(wrong_df.head(10)[['file_name', 'true_label', 'pred_label', 'confidence']])

    # 輸出混淆矩陣
    class_names = [idx_to_class[i] for i in range(num_classes)]
    # 輸出熱圖（raw count）
    cm_png_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_heatmap(confusion, class_names, cm_png_path, "Confusion Matrix (Raw Counts)")

    # 顯示最常見錯分對（排除對角線）
    off_diag = confusion.copy()
    np.fill_diagonal(off_diag, 0)
    top_k = min(10, (off_diag > 0).sum())
    if top_k > 0:
        flat_idx = np.argpartition(off_diag.ravel(), -top_k)[-top_k:]
        ranked_idx = flat_idx[np.argsort(off_diag.ravel()[flat_idx])[::-1]]
        print("\nTop Confusion Pairs (True -> Pred):")
        for idx in ranked_idx:
            t, p = np.unravel_index(idx, off_diag.shape)
            print(f"{idx_to_class[t]} -> {idx_to_class[p]}: {off_diag[t, p]}")

    print(f"\nSaved: {val_analysis_path}")
    print(f"Saved: {cm_png_path}")
    
    return df

if __name__ == "__main__":
    analyze_validation(
        model_path="checkpoints/033009_SER50_mixup0.8_LR0.0001_AA_256_ws1.0/best_model.pth",
        val_dir="./data/val",
        train_dir="./data/train"
    )