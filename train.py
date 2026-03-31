import argparse
import os
import datetime
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from models.resnet50 import get_se_resnet50, get_finegrained_resnet50
from utils.dataset import get_dataloaders

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_args():
    parser = argparse.ArgumentParser(description="Train a ResNet50 model on the given dataset.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for training and validation.")

    parser.add_argument("--mix_type", type=str, choices=["mixup", "cutmix"], help="Use Mixup or CutMix data augmentation.")
    parser.add_argument("--mix_alpha", type=float, default=0.9, help="Alpha value for Mixup/CutMix.")
    parser.add_argument("--transform_type", type=str, default="manual", choices=["AA", "manual"], help="Augmentation strategy: AA (AutoAugment) or manual.")
    parser.add_argument("--model_type", type=str, default="se", choices=["se", "fg_gem"], help="Model architecture: se or fg_gem.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for fg_gem head.")
    parser.add_argument("--gem_p", type=float, default=3.0, help="Initial GeM p for fg_gem model.")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use WeightedRandomSampler for class imbalance.")
    parser.add_argument("--sampler_power", type=float, default=1.0, help="Strength of weighted sampling. 1.0 means inverse-frequency.")

    return parser.parse_args()

def evaluate(model, dataloader, criterion, device):
    """ 評估函式：計算 Loss 與 Accuracy """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 按樣本數加權，避免最後一個 batch 偏差
            total_loss += loss.item() * labels.size(0)
            
            # 計算 Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return True
    return False

def train():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. Data and Model ---
    train_loader, val_loader = get_dataloaders(
        data_dir="./data",
        batch_size=64,
        image_size=args.image_size,
        transform_type=args.transform_type,
        use_weighted_sampler=args.use_weighted_sampler,
        sampler_power=args.sampler_power,
    )

    if args.model_type == "fg_gem":
        model = get_finegrained_resnet50(dropout=args.dropout, gem_p=args.gem_p).to(device)
    else:
        model = get_se_resnet50().to(device)

    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # --- 3. Optimizer & Scheduler ---
    total_epochs = 100
    warmup_epochs = 10
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-warmup_epochs)

    # --- 4. Experiment Setup ---
    mix_tag = args.mix_type + str(args.mix_alpha) if args.mix_type else 'basic'
    sampler_tag = f"_ws{args.sampler_power}" if args.use_weighted_sampler else ""
    model_tag = f"{args.model_type}"
    exp_name = f"{datetime.datetime.now().strftime('%m%d%H')}_SER50_{mix_tag}_LR{args.lr}_{args.transform_type}_{args.image_size}_{model_tag}{sampler_tag}"
    exp_dir = f"checkpoints/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    # --- 5. Training loop ---
    best_val_acc = 0.0
    
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        is_warmup = adjust_learning_rate(optimizer, epoch, warmup_epochs, args.lr)
        
        # 使用 enumerate 取得 batch_idx
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            # --- 紀錄第一批增強後的圖片到 TensorBoard ---
            if epoch == 0 and batch_idx == 0:
                img_grid = torchvision.utils.make_grid(images[:8], normalize=True) 
                writer.add_image('Augmented_Images', img_grid, 0)

            # --- 混合增強邏輯 ---
            mix_ratio = 1.0 
            labels_a, labels_b = labels, labels
            do_mix = args.mix_type is not None and np.random.rand() < 0.5

            if do_mix:
                mix_ratio = np.random.beta(args.mix_alpha, args.mix_alpha)
                index = torch.randperm(images.size(0)).to(device)
                labels_b = labels[index]

                if args.mix_type == "mixup":
                    images = mix_ratio * images + (1 - mix_ratio) * images[index]
                elif args.mix_type == "cutmix":
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), mix_ratio)
                    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
                    mix_ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
            
            optimizer.zero_grad()
            outputs = model(images)
            
            if do_mix:
                loss = mix_ratio * criterion(outputs, labels_a) + (1 - mix_ratio) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # 紀錄 Step Loss
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_step', loss.item(), global_step)

            # 按樣本數加權累積 loss
            running_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # --- Epoch 結束後的處理 ---
        train_loss_avg = running_loss / total_train_samples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if not is_warmup:    
            scheduler.step()
        
        # --- TensorBoard Logging (Epoch Level) ---
        writer.add_scalar('Loss/train_epoch', train_loss_avg, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # --- 儲存與日誌 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            with open(os.path.join(exp_dir, "results.txt"), "a") as f:
                f.write(f"Epoch {epoch+1}: Best Val Acc = {val_acc:.2f}%\n")
        
        torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pth"))
        print(f"Epoch {epoch+1} Summary: Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    train()