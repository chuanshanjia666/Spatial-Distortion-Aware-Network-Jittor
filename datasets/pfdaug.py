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
        """Apply barrel distortion to the image via backward mapping.

        For every output pixel at radius r_u, solves k·r_d³ + r_d - r_u = 0
        to find the source pixel (Newton's method).
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
        """Distort a single point (Alg. 2) — same normalisation as _warp_image."""
        cx, cy = W / 2.0, H / 2.0
        scale = max(cx, cy)

        xn = (x - cx) / scale
        yn = (y - cy) / scale
        r = np.sqrt(xn * xn + yn * yn)
        theta = np.arctan2(yn, xn)
        r_dis = r * (1.0 + k * r * r)

        x_dis = r_dis * np.cos(theta) * scale + cx
        y_dis = r_dis * np.sin(theta) * scale + cy

        return np.float32(x_dis), np.float32(y_dis)


# ===================================================================
#  Self-test: synthetic grid → visualize distortion w/ box alignment
# ===================================================================
if __name__ == "__main__":
    import os

    W, H = 1920, 1080
    k = 0.5
    out_dir = "output/pfdaug_test"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Build test image ----
    img = np.full((H, W, 3), 64, dtype=np.uint8)
    cx, cy = W // 2, H // 2
    for r in range(100, 700, 100):
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)
    cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
    cv2.line(img, (0, cy), (W, cy), (255, 255, 255), 1)
    cv2.line(img, (cx, 0), (cx, H), (255, 255, 255), 1)
    for deg in range(0, 180, 30):
        a = np.deg2rad(deg)
        cv2.line(img, (int(cx - 600 * np.cos(a)), int(cy - 600 * np.sin(a))),
                 (int(cx + 600 * np.cos(a)), int(cy + 600 * np.sin(a))),
                 (200, 200, 200), 1)

    # ---- Test boxes ----
    boxes = np.array([
        [300, 540, 120, 60, 30],
        [960, 540, 200, 100, 0],
        [1600, 540, 100, 70, -45],
    ], dtype=np.float32)

    # ---- Draw original ----
    img_orig = img.copy()
    for (bx, by, bw, bh, bR) in boxes:
        pts = cv2.boxPoints(((float(bx), float(by)), (float(bw), float(bh)), float(bR)))
        cv2.drawContours(img_orig, [np.intp(pts)], 0, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "01_original.png"),
                cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR))

    # ---- Apply PFDAug ----
    aug = PFDAug(k=k, p=1.0)
    img_aug, boxes_aug = aug.forward(img, boxes, k)

    # ---- Draw augmented ----
    img_out = img_aug.copy()
    for (bx, by, bw, bh, bR) in boxes_aug:
        pts = cv2.boxPoints(((float(bx), float(by)), (float(bw), float(bh)), float(bR)))
        cv2.drawContours(img_out, [np.intp(pts)], 0, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "02_augmented.png"),
                cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))

    # ---- Consistency check: forward-distort original corners, draw as dots ----
    img_check = img_aug.copy()
    for (bx, by, bw, bh, bR) in boxes:
        half_w, half_h = bw / 2, bh / 2
        a_rad = np.deg2rad(bR)
        cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
        corners = np.array([
            [bx + (-half_w) * cos_a - (-half_h) * sin_a,
             by + (-half_w) * sin_a + (-half_h) * cos_a],
            [bx + ( half_w) * cos_a - (-half_h) * sin_a,
             by + ( half_w) * sin_a + (-half_h) * cos_a],
            [bx + ( half_w) * cos_a - ( half_h) * sin_a,
             by + ( half_w) * sin_a + ( half_h) * cos_a],
            [bx + (-half_w) * cos_a - ( half_h) * sin_a,
             by + (-half_w) * sin_a + ( half_h) * cos_a],
        ], dtype=np.float32)
        for (cx, cy) in corners:
            dx, dy = PFDAug._point_distortion(cx, cy, W, H, k)
            cv2.circle(img_check, (int(dx), int(dy)), 4, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(out_dir, "03_dots.png"),
                cv2.cvtColor(img_check, cv2.COLOR_RGB2BGR))

    # ---- Print comparison ----
    print(f"Image: {W}×{H}, k={k}")
    for i, (orig, aug) in enumerate(zip(boxes, boxes_aug)):
        err = np.sqrt((orig[0] - aug[0]) ** 2 + (orig[1] - aug[1]) ** 2)
        print(f"  Box {i}:  [{orig[0]:6.0f} {orig[1]:6.0f} {orig[2]:6.0f} {orig[3]:6.0f} {orig[4]:6.0f}]"
              f"  →  [{aug[0]:6.0f} {aug[1]:6.0f} {aug[2]:6.0f} {aug[3]:6.0f} {aug[4]:6.0f}]"
              f"  Δcenter={err:.0f}px")
    print(f"\nSaved to {out_dir}/")
    print("  01_original  = before,  02_augmented = after")
    print("  03_dots      = red dots = forward-distorted original corners")
    print("  If box alignment is correct, green boxes should contain red dots.")
