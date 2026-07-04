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
        k  : Distortion coefficient.  Larger k → stronger barrel distortion.
             Paper default is 0.5; 0.8 may hurt on "easy" scenes.
        p  : Probability of applying the augmentation.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, k: float = 0.5, p: float = 0.5, seed: int | None = None):
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
        return self.forward(image, boxes, self.k)

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
        """Apply barrel distortion to the whole image via backward mapping.

        For every output pixel at normalised radius r_u, solve for the
        source radius r_d:  k·r_d³ + r_d - r_u = 0   (Newton's method).
        """
        cx, cy = W / 2.0, H / 2.0
        scale = max(cx, cy)

        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        xn = (xs - cx) / scale
        yn = (ys - cy) / scale
        r_u = np.sqrt(xn ** 2 + yn ** 2)

        eps = 1e-8
        valid = r_u > eps
        r_d = r_u.copy()

        for _ in range(5):
            r2 = r_d ** 2
            f = k * r_d * r2 + r_d - r_u
            fprime = 3.0 * k * r2 + 1.0
            r_d = np.where(valid, r_d - f / fprime, r_d)

        ratio = np.ones_like(r_u, dtype=np.float32)
        ratio[valid] = r_d[valid] / r_u[valid]
        xs_src = ratio * xn * scale + cx
        ys_src = ratio * yn * scale + cy

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

            local_corners = np.array([
                [-half_w, -half_h],
                [ half_w, -half_h],
                [-half_w,  half_h],
                [ half_w,  half_h],
            ], dtype=np.float32)

            cos_a, sin_a = np.cos(R_rad), np.sin(R_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            rotated_corners = cx + local_corners @ rot.T

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
        """Distort a single pixel coordinate (Alg. 2)."""
        halfW, halfH = W / 2.0, H / 2.0

        xn = (x - halfW) / halfW
        yn = (y - halfH) / halfH
        r = np.sqrt(xn * xn + yn * yn)
        theta = np.arctan2(yn, xn)
        r_dis = r * (1.0 + k * r * r)

        x_dis = r_dis * np.cos(theta) * halfW + halfW
        y_dis = r_dis * np.sin(theta) * halfH + halfH

        return np.float32(x_dis), np.float32(y_dis)
