import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir,
    image_size=224,
    batch_size=32,
    transform_type="AA",
    use_weighted_sampler=False,
    sampler_power=1.0,
):
    if transform_type == "AA":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                # 幾何變換
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                # 畫質增強
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # 隨機擦除
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.ImageFolder(
        root=f"{data_dir}/train", transform=train_transform
    )
    val_set = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)

    if use_weighted_sampler:
        targets = torch.tensor(train_set.targets, dtype=torch.long)
        class_counts = torch.bincount(targets).float().clamp_min(1)
        class_weights = (1.0 / class_counts) ** sampler_power
        sample_weights = class_weights[targets]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=4,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4
        )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
