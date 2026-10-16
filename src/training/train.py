import os
import csv
import random

import numpy as np
import rasterio
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.lite_rsnet import LiteRSNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"D:\Bapan\auto_digi\track\LiteRSNet_Dataset"

TRAIN_IMAGE_DIR = os.path.join(
    DATASET_DIR, "images", "train"
)

TRAIN_MASK_DIR = os.path.join(
    DATASET_DIR, "masks", "train"
)

VAL_IMAGE_DIR = os.path.join(
    DATASET_DIR, "images", "val"
)

VAL_MASK_DIR = os.path.join(
    DATASET_DIR, "masks", "val"
)

# New experiment directory
OUTPUT_DIR = r"D:\Bapan\VS\rail_track_detection\runs\lite_rsnet_v2"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR, "checkpoints"
)

LOG_FILE = os.path.join(
    OUTPUT_DIR, "training_log.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 1024

BATCH_SIZE = 1

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

class RailDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = image_dir

        self.mask_dir = mask_dir

        self.image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith(
                    (".tif", ".tiff")
                )
            ]
        )

        if len(self.image_files) == 0:

            raise RuntimeError(
                f"No TIFF images found in:\n{image_dir}"
            )

        # ----------------------------------------------------
        # Check matching masks
        # ----------------------------------------------------

        valid_images = []

        for image_file in self.image_files:

            base_name = os.path.splitext(
                image_file
            )[0]

            mask_file = (
                base_name + "_mask.png"
            )

            mask_path = os.path.join(
                mask_dir,
                mask_file
            )

            if os.path.exists(mask_path):

                valid_images.append(
                    image_file
                )

            else:

                print(
                    f"WARNING: Mask missing for "
                    f"{image_file}"
                )

        self.image_files = valid_images

        if len(self.image_files) == 0:

            raise RuntimeError(
                "No image/mask pairs found."
            )

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):

        image_name = self.image_files[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        base_name = os.path.splitext(
            image_name
        )[0]

        mask_name = (
            base_name + "_mask.png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        # ----------------------------------------------------
        # Read GeoTIFF
        # ----------------------------------------------------

        with rasterio.open(image_path) as src:

            image = src.read(
                [1, 2, 3]
            )

        # ----------------------------------------------------
        # Convert H,W,C
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (1, 2, 0)
        )

        image = image.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Normalize RGB
        # ----------------------------------------------------

        image = image / 255.0

        image = np.clip(
            image,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = np.array(
            Image.open(mask_path)
        )

        # ----------------------------------------------------
        # Binary mask
        #
        # 0 = background
        # 1 = rail
        # ----------------------------------------------------

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # H,W,C -> C,H,W
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        ).float()

        mask = torch.from_numpy(
            mask.copy()
        ).float()

        # Add channel to mask
        mask = mask.unsqueeze(0)

        return image, mask, image_name


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.contiguous().view(
            probabilities.shape[0],
            -1
        )

        targets = targets.contiguous().view(
            targets.shape[0],
            -1
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            +
            targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            +
            self.smooth
        ) / (
            denominator
            +
            self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# BCE + DICE LOSS
# ============================================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.7,
        bce_weight=0.3
    ):

        super().__init__()

        self.dice_weight = dice_weight

        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss()

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits,
        targets
    ):

        dice = self.dice_loss(
            logits,
            targets
        )

        bce = self.bce_loss(
            logits,
            targets
        )

        loss = (
            self.dice_weight * dice
            +
            self.bce_weight * bce
        )

        return loss


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    prediction_sum = predictions.sum()

    target_sum = targets.sum()

    union = (
        prediction_sum
        +
        target_sum
        -
        intersection
    )

    dice = (
        2.0 * intersection
        +
        1e-7
    ) / (
        prediction_sum
        +
        target_sum
        +
        1e-7
    )

    iou = (
        intersection
        +
        1e-7
    ) / (
        union
        +
        1e-7
    )

    precision = (
        intersection
        +
        1e-7
    ) / (
        prediction_sum
        +
        1e-7
    )

    recall = (
        intersection
        +
        1e-7
    ) / (
        target_sum
        +
        1e-7
    )

    return (
        dice.item(),
        iou.item(),
        precision.item(),
        recall.item()
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    epoch
):

    model.train()

    running_loss = 0.0

    running_dice = 0.0

    running_iou = 0.0

    running_precision = 0.0

    running_recall = 0.0

    total_batches = len(loader)

    for batch_index, (
        images,
        masks,
        names
    ) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        dice, iou, precision, recall = calculate_metrics(
            logits.detach(),
            masks
        )

        running_loss += loss.item()

        running_dice += dice

        running_iou += iou

        running_precision += precision

        running_recall += recall

        print(
            f"\rEpoch {epoch:03d} | "
            f"Batch {batch_index + 1:02d}/{total_batches:02d} | "
            f"Loss {loss.item():.4f} | "
            f"Dice {dice:.4f} | "
            f"IoU {iou:.4f}",
            end=""
        )

    print()

    return (
        running_loss / total_batches,
        running_dice / total_batches,
        running_iou / total_batches,
        running_precision / total_batches,
        running_recall / total_batches
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    running_dice = 0.0

    running_iou = 0.0

    running_precision = 0.0

    running_recall = 0.0

    total_batches = len(loader)

    for images, masks, names in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

        dice, iou, precision, recall = calculate_metrics(
            logits,
            masks
        )

        running_loss += loss.item()

        running_dice += dice

        running_iou += iou

        running_precision += precision

        running_recall += recall

    return (
        running_loss / total_batches,
        running_dice / total_batches,
        running_iou / total_batches,
        running_precision / total_batches,
        running_recall / total_batches
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("LITE-RSNET V2 - RAIL SEGMENTATION TRAINING")

    print("=" * 70)

    print()

    print("Dataset:", DATASET_DIR)

    print("Output:", OUTPUT_DIR)

    print()

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    print()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = RailDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )

    val_dataset = RailDataset(
        VAL_IMAGE_DIR,
        VAL_MASK_DIR
    )

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print()

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LiteRSNet(
        in_channels=3,
        num_classes=1
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameters:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = CombinedLoss(
        dice_weight=0.7,
        bce_weight=0.3
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # AMP SCALER
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # LOG FILE
    # --------------------------------------------------------

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_dice",
            "train_iou",
            "train_precision",
            "train_recall",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_precision",
            "val_recall",
            "learning_rate"
        ])

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_val_dice = -1.0

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()

        print("-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("-" * 70)

        train_results = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            epoch
        )

        (
            train_loss,
            train_dice,
            train_iou,
            train_precision,
            train_recall
        ) = train_results

        val_results = validate(
            model,
            val_loader,
            criterion
        )

        (
            val_loss,
            val_dice,
            val_iou,
            val_precision,
            val_recall
        ) = val_results

        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Train Dice      : {train_dice:.4f}"
        )

        print(
            f"Train IoU       : {train_iou:.4f}"
        )

        print(
            f"Train Precision : {train_precision:.4f}"
        )

        print(
            f"Train Recall    : {train_recall:.4f}"
        )

        print()

        print(
            f"Val Loss        : {val_loss:.4f}"
        )

        print(
            f"Val Dice        : {val_dice:.4f}"
        )

        print(
            f"Val IoU         : {val_iou:.4f}"
        )

        print(
            f"Val Precision   : {val_precision:.4f}"
        )

        print(
            f"Val Recall      : {val_recall:.4f}"
        )

        print(
            f"Learning Rate   : {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "latest.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou
            },
            latest_path
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou
                },
                best_path
            )

            print()

            print(
                "*** NEW BEST MODEL SAVED ***"
            )

            print(
                f"Best validation Dice: "
                f"{best_val_dice:.4f}"
            )

        # ----------------------------------------------------
        # CSV LOG
        # ----------------------------------------------------

        with open(
            LOG_FILE,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                train_dice,
                train_iou,
                train_precision,
                train_recall,
                val_loss,
                val_dice,
                val_iou,
                val_precision,
                val_recall,
                current_lr
            ])

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        if torch.cuda.is_available():

            memory = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            print(
                f"GPU memory allocated: "
                f"{memory:.2f} GB"
            )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print("LITE-RSNET V2 TRAINING COMPLETED")

    print("=" * 70)

    print()

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )

    print()

    print("Best model:")

    print(
        os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )
    )

    print()

    print("Training log:")

    print(
        LOG_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
