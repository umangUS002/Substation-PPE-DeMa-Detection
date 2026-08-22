"""
Few-Shot Data Augmentation Engine.
Implements geometric and color-space augmentation strategies for small datasets
(~50 images per class) as detailed in Section III-B of the paper:
Zhao & Barati (IEEE Transactions on Industry Applications, 2023).

Geometric transforms (horizontal flip, random crop/scale jitter) keep bounding
boxes aligned with the transformed image; boxes are clipped to the crop and
dropped if they no longer have meaningful area.
"""

import random
from PIL import Image, ImageEnhance


class FewShotAugmenter:
    """
    Applies diverse geometric and color transformations to expand a few-shot dataset.
    """

    def __init__(self, target_size=(800, 600), hflip_prob=0.5, crop_prob=0.5, min_crop_scale=0.7):
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.crop_prob = crop_prob
        self.min_crop_scale = min_crop_scale

    def resize_standard(self, image: Image.Image, boxes: list = None):
        """Resizes image (and, if given, boxes) to the paper standard 800x600x3."""
        orig_w, orig_h = image.size
        target_w, target_h = self.target_size
        resized = image.resize(self.target_size, Image.Resampling.BILINEAR)

        if boxes is None:
            return resized

        scale_x = target_w / max(float(orig_w), 1.0)
        scale_y = target_h / max(float(orig_h), 1.0)
        scaled_boxes = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for x1, y1, x2, y2 in boxes
        ]
        return resized, scaled_boxes

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """Brightness / contrast / sharpness jittering. Does not move box coordinates."""
        img = image.copy()
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.3))
        return img

    def _horizontal_flip(self, image: Image.Image, boxes: list, labels: list):
        w, _ = image.size
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]
        return flipped, flipped_boxes, labels

    def _random_crop(self, image: Image.Image, boxes: list, labels: list):
        """
        Crops a random sub-region (scale jitter). Boxes are clipped to the crop
        and dropped if the visible remainder is negligible. If the crop would
        remove every box, the original (uncropped) sample is returned instead,
        since few-shot datasets can't afford to lose annotations to augmentation.
        """
        w, h = image.size
        scale = random.uniform(self.min_crop_scale, 1.0)
        crop_w, crop_h = int(w * scale), int(h * scale)
        if crop_w < 1 or crop_h < 1:
            return image, boxes, labels

        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        cropped = image.crop((left, top, left + crop_w, top + crop_h))

        new_boxes, new_labels = [], []
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            nx1, ny1 = max(x1, left), max(y1, top)
            nx2, ny2 = min(x2, left + crop_w), min(y2, top + crop_h)
            if (nx2 - nx1) > 2 and (ny2 - ny1) > 2:
                new_boxes.append([nx1 - left, ny1 - top, nx2 - left, ny2 - top])
                new_labels.append(label)

        if boxes and not new_boxes:
            return image, boxes, labels

        return cropped, new_boxes, new_labels

    def apply_augmentation(self, image: Image.Image, boxes: list, labels: list):
        """
        Applies color jittering plus geometric transforms (horizontal flip,
        random crop/scale jitter), keeping bounding boxes aligned throughout,
        then resizes to the target size.
        """
        img = self._color_jitter(image)

        if random.random() < self.hflip_prob:
            img, boxes, labels = self._horizontal_flip(img, boxes, labels)

        if random.random() < self.crop_prob:
            img, boxes, labels = self._random_crop(img, boxes, labels)

        img, boxes = self.resize_standard(img, boxes)
        return img, boxes, labels

    def generate_augmented_batch(self, image: Image.Image, boxes: list, labels: list, num_samples: int = 5):
        """Generates multiple augmented (image, boxes, labels) variants of a single sample."""
        base_image, base_boxes = self.resize_standard(image, boxes)
        samples = [(base_image, base_boxes, labels)]
        for _ in range(num_samples - 1):
            samples.append(self.apply_augmentation(image, boxes, labels))
        return samples
