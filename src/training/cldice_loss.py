import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SOFT MORPHOLOGICAL OPERATIONS
# ============================================================

def soft_erode(img):
    """
    Differentiable approximation of morphological erosion.
    """

    p1 = -F.max_pool2d(
        -img,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0)
    )

    p2 = -F.max_pool2d(
        -img,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1)
    )

    return torch.minimum(p1, p2)


def soft_dilate(img):
    """
    Differentiable approximation of morphological dilation.
    """

    return F.max_pool2d(
        img,
        kernel_size=3,
        stride=1,
        padding=1
    )


def soft_open(img):
    """
    Soft morphological opening.
    """

    return soft_dilate(
        soft_erode(img)
    )


# ============================================================
# SOFT SKELETONIZATION
# ============================================================

def soft_skeletonize(img, iterations=5):
    """
    Differentiable soft skeletonization.

    Produces a soft centerline representation
    while keeping the operation differentiable.
    """

    img = img.float()

    opened = soft_open(img)

    skeleton = F.relu(
        img - opened
    )

    for _ in range(iterations):

        img = soft_erode(img)

        opened = soft_open(img)

        delta = F.relu(
            img - opened
        )

        skeleton = skeleton + F.relu(
            delta - skeleton * delta
        )

    return skeleton


# ============================================================
# CLDICE LOSS
# ============================================================

class CLDiceLoss(nn.Module):

    def __init__(
        self,
        iterations=5,
        smooth=1e-6
    ):
        super().__init__()

        self.iterations = iterations
        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):
        """
        logits:
            Model output
            Shape: [B, 1, H, W]

        targets:
            Ground-truth mask
            Shape: [B, 1, H, W]
        """

        # ----------------------------------------------------
        # Convert logits to probability
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        # ----------------------------------------------------
        # Use float32 for stable topology calculation
        # ----------------------------------------------------

        probabilities = probabilities.float()
        targets = targets.float()

        # ----------------------------------------------------
        # Soft skeletons
        # ----------------------------------------------------

        skeleton_pred = soft_skeletonize(
            probabilities,
            iterations=self.iterations
        )

        skeleton_true = soft_skeletonize(
            targets,
            iterations=self.iterations
        )

        # ----------------------------------------------------
        # Topology precision
        #
        # How much predicted centerline is inside
        # the ground-truth rail?
        # ----------------------------------------------------

        tprec = (
            (
                skeleton_pred * targets
            ).sum(dim=(1, 2, 3))
            +
            self.smooth
        ) / (
            skeleton_pred.sum(dim=(1, 2, 3))
            +
            self.smooth
        )

        # ----------------------------------------------------
        # Topology sensitivity
        #
        # How much ground-truth centerline is covered
        # by the prediction?
        # ----------------------------------------------------

        tsens = (
            (
                skeleton_true * probabilities
            ).sum(dim=(1, 2, 3))
            +
            self.smooth
        ) / (
            skeleton_true.sum(dim=(1, 2, 3))
            +
            self.smooth
        )

        # ----------------------------------------------------
        # clDice
        # ----------------------------------------------------

        cldice = (
            2.0 * tprec * tsens
        ) / (
            tprec + tsens + self.smooth
        )

        # ----------------------------------------------------
        # Convert score to loss
        # ----------------------------------------------------

        loss = 1.0 - cldice.mean()

        return loss
