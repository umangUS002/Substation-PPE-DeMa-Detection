"""
Few-Shot Data Augmentation Engine.
Implements data augmentation strategies for small datasets (~50 images per class)
as detailed in Section III-B of the paper:
Zhao & Barati (IEEE Transactions on Industry Applications, 2023).
"""

import random
from PIL import Image, ImageEnhance, ImageOps

class FewShotAugmenter:
    """
    Applies diverse geometric and color transformations to expand a few-shot dataset.
    """

    def __init__(self, target_size=(800, 600)):
        self.target_size = target_size

    def resize_standard(self, image: Image.Image) -> Image.Image:
        """Resizes image to paper standard 800x600x3."""
        return image.resize(self.target_size, Image.Resampling.BILINEAR)

    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Applies color space transformations (brightness, contrast, sharpness)
        which preserve exact bounding box spatial alignment.
        """
        img = image.copy()

        # Brightness Jittering
        brightness_factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # Contrast Jittering
        contrast_factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # Sharpness Adjustment
        sharpness_factor = random.uniform(0.8, 1.3)
        img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)

        return self.resize_standard(img)

    def generate_augmented_batch(self, image: Image.Image, num_samples: int = 5):
        """Generates multiple augmented variants of a single image sample."""
        augmented_images = [self.resize_standard(image)]
        for _ in range(num_samples - 1):
            augmented_images.append(self.apply_augmentation(image))
        return augmented_images
