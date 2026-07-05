"""
PFDAug: Prominent Fisheye Distortion Augmentation.

Implements online data augmentation from:
  "Towards Better Distortion Feature Learning for Object Detection
   in Top-View Fisheye Cameras" (Guo et al., IEEE TMM 2026).

PFDAug applies additional barrel distortion to fisheye images and
adapts oriented bounding-box annotations accordingly during training.
Data volume is unchanged — it works as an online transform, not an
offline dataset generator.

Usage::

    from datasets.pfdaug import PFDAug
    aug = PFDAug(k=0.5, p=0.5)
    img_aug, boxes_aug = aug(image, boxes)
"""

import numpy as np
import cv2


class PFDAug:
    """Prominent Fisheye Distortion Augmentation.

    Args:
        k  : Tuple of distortion coefficients. Randomly sampled per image.
             Larger k → stronger barrel distortion. E.g. (0.3, 0.5, 0.7).
        p  : Probability of applying the augmentation.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, k: tuple = (0.5,), p: float = 0.5, seed: int | None = None):
        self.k = k
        self.p = p
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        """Apply PFDAug with the configured probability.

        Args:
            image: (H, W, 3) uint8 numpy array.
            boxes: (N, 5) float32 array ``[cx, cy, w, h, R]`` in pixel coords,
                   R in **degrees**.

        Returns:
            (image, boxes) — the augmented pair (may be unchanged
            when the random draw exceeds ``self.p``).
        """
        if self.p < 1.0 and self.rng.random() > self.p:
            return image, boxes
        # Randomly select k from the tuple
        k = self.rng.choice(self.k)
        return self.forward(image, boxes, k)

    def forward(self, image: np.ndarray, boxes: np.ndarray, k: float):
        """Deterministic forward pass (always applies distortion)."""
        H, W = image.shape[:2]
        image_distorted = self._warp_image(image, W, H, k)
        boxes_distorted = self._warp_boxes(boxes, W, H, k)
        return image_distorted, boxes_distorted

    # ------------------------------------------------------------------
    # Image warp (barrel distortion via backward mapping + Newton)
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_image(image: np.ndarray, W: int, H: int, k: float) -> np.ndarray:
        """Apply barrel distortion to the image (backward mapping).

        The forward map follows the paper's radial model:

            r_dst = r_src · (1 + k · r_src²)

        where x and y are normalized independently by W/2 and H/2.
        This means r = 1 lies on the middle of each image edge, while
        the corners are at r = sqrt(2). We invert this mapping with
        Newton's method so every output pixel samples the correct source
        location.

        For valid pixels we solve  k·r_src³ + r_src - r_dst = 0
        via Newton's method.
        """
        cx, cy = W / 2.0, H / 2.0
        scale_x = cx
        scale_y = cy

        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        xn = (xs - cx) / scale_x
        yn = (ys - cy) / scale_y
        r_dst = np.sqrt(xn ** 2 + yn ** 2)

        valid = r_dst > 1e-8
        r_src = np.zeros_like(r_dst)
        r_src[valid] = r_dst[valid]
        for _ in range(10):
            r2 = r_src * r_src
            f = k * r_src * r2 + r_src - r_dst
            fprime = 3.0 * k * r2 + 1.0
            delta = np.zeros_like(r_src)
            np.divide(f, fprime, out=delta, where=valid & (np.abs(fprime) > 1e-8))
            r_src[valid] -= delta[valid]

        ratio = np.zeros_like(r_dst)
        ratio[valid] = r_src[valid] / r_dst[valid]
        xs_src = ratio * xn * scale_x + cx
        ys_src = ratio * yn * scale_y + cy

        return cv2.remap(image, xs_src.astype(np.float32), ys_src.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # ------------------------------------------------------------------
    # Box warp — Algorithm 1 from the paper
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_boxes(boxes: np.ndarray, W: int, H: int, k: float) -> np.ndarray:
        """Adjust oriented bounding boxes after barrel distortion.

        For each box: compute four rotated corners, distort each corner
        (Alg. 2), then compute the minimum area bounding rectangle (MBR).
        """
        if len(boxes) == 0:
            return boxes.copy()

        boxes = np.asarray(boxes, dtype=np.float32)
        result = np.empty_like(boxes)

        for i, box in enumerate(boxes):
            cx, cy, w, h, R = box
            R_rad = np.deg2rad(R)
            half_w, half_h = w / 2.0, h / 2.0

            # ---- corner order fix: clockwise from top-left ----
            local_corners = np.array([
                [-half_w, -half_h],
                [ half_w, -half_h],
                [ half_w,  half_h],
                [-half_w,  half_h],
            ], dtype=np.float32)

            cos_a, sin_a = np.cos(R_rad), np.sin(R_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            rotated_corners = local_corners @ rot.T + np.array([cx, cy])

            distorted_corners = np.empty_like(rotated_corners)
            for j in range(4):
                distorted_corners[j] = PFDAug._point_distortion(
                    rotated_corners[j, 0], rotated_corners[j, 1], W, H, k)

            contour = distorted_corners.reshape(-1, 1, 2)
            rect = cv2.minAreaRect(contour)
            (new_cx, new_cy), (new_w, new_h), new_angle = rect

            if new_w < new_h:
                new_w, new_h = new_h, new_w
                new_angle += 90.0
            new_angle = (new_angle + 90.0) % 180.0 - 90.0

            result[i] = [new_cx, new_cy, new_w, new_h, new_angle]

        return result

    # ------------------------------------------------------------------
    # Point distortion — Algorithm 2 from the paper
    # ------------------------------------------------------------------

    @staticmethod
    def _point_distortion(x: float, y: float, W: int, H: int, k: float):
        """Distort a single point (Alg. 2) — barrel distortion: edges pushed outward.

        The point warp uses the same axis-wise normalization as image warp.
        """
        cx, cy = W / 2.0, H / 2.0
        scale_x = cx
        scale_y = cy

        xn = (x - cx) / scale_x
        yn = (y - cy) / scale_y
        r = np.sqrt(xn * xn + yn * yn)
        theta = np.arctan2(yn, xn)
        r_dis = r * (1.0 + k * r * r)

        x_dis = r_dis * np.cos(theta) * scale_x + cx
        y_dis = r_dis * np.sin(theta) * scale_y + cy

        return np.float32(x_dis), np.float32(y_dis)


