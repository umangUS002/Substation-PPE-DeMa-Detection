"""
Few-Shot Data Augmentation Engine with Synthetic Occlusion Cutout.
Implements data augmentation strategies for small datasets (~50 images per class)
as detailed in Section III-B of the paper:
Zhao & Barati (IEEE Transactions on Industry Applications, 2023).
"""

from typing import List
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw


class FewShotAugmenter:
    """
    Applies diverse geometric, color space, and synthetic occlusion transformations
    to expand a small ("few-shot") industrial dataset into thousands of training samples.
    """

    def __init__(self, target_size=(800, 600)):
        self.target_size = target_size

    def resize_standard(self, image: Image.Image) -> Image.Image:
        """Resizes image to standard 800x600 resolution."""
        return image.resize(self.target_size, Image.Resampling.BILINEAR)

    def apply_color_jitter(self, image: Image.Image) -> Image.Image:
        """
        Applies color space transformations (brightness, contrast, sharpness)
        simulating different industrial plant lighting conditions.
        """
        img = image.copy()

        # Brightness Jittering (0.8x to 1.2x)
        b_factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(b_factor)

        # Contrast Jittering (0.8x to 1.2x)
        c_factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(c_factor)

        # Sharpness Adjustment (0.8x to 1.3x)
        s_factor = random.uniform(0.8, 1.3)
        img = ImageEnhance.Sharpness(img).enhance(s_factor)

        return img

    def apply_synthetic_occlusion_cutout(self, image: Image.Image) -> Image.Image:
        """
        Applies synthetic cutout masking (occluding 15-25% of image region).
        Teaches the AI model to recognize helmets, gloves, and boots even when
        partially blocked by objects or workers.
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Random cutout box
        cutout_w = random.randint(int(w * 0.1), int(w * 0.25))
        cutout_h = random.randint(int(h * 0.1), int(h * 0.25))
        x1 = random.randint(0, w - cutout_w)
        y1 = random.randint(0, h - cutout_h)

        draw.rectangle([x1, y1, x1 + cutout_w, y1 + cutout_h], fill=(30, 35, 40))
        return img

    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Combines color jitter, synthetic occlusion cutout, and resizing."""
        img = self.apply_color_jitter(image)
        if random.random() > 0.5:
            img = self.apply_synthetic_occlusion_cutout(img)
        return self.resize_standard(img)

    def generate_augmented_batch(self, image: Image.Image, num_samples: int = 5) -> List[Image.Image]:
        """Generates multiple augmented variants from a single image sample."""
        augmented_images = [self.resize_standard(image)]
        for _ in range(num_samples - 1):
            augmented_images.append(self.apply_augmentation(image))
        return augmented_images
