import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    class_id: int
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float
    confidence: float

    def to_bbox_px(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        x_center = self.x_center_norm * image_width
        y_center = self.y_center_norm * image_height
        w = self.width_norm * image_width
        h = self.height_norm * image_height
        x1 = int(round(x_center - w / 2))
        y1 = int(round(y_center - h / 2))
        x2 = int(round(x_center + w / 2))
        y2 = int(round(y_center + h / 2))
        return x1, y1, x2, y2

    def center_px(self, image_width: int, image_height: int) -> Tuple[int, int]:
        return (
            int(round(self.x_center_norm * image_width)),
            int(round(self.y_center_norm * image_height)),
        )


def load_yolo_labels(labels_path: str) -> List[Detection]:
    detections: List[Detection] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # Some exporters include confidence as the 6th field; default to 1.0 if missing
            class_id = int(float(parts[0]))
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            confidence = float(parts[5]) if len(parts) > 5 else 1.0
            detections.append(
                Detection(
                    class_id=class_id,
                    x_center_norm=x_center_norm,
                    y_center_norm=y_center_norm,
                    width_norm=width_norm,
                    height_norm=height_norm,
                    confidence=confidence,
                )
            )
    return detections


def group_rows_by_y(detections: List[Detection]) -> List[List[Detection]]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.y_center_norm)
    median_h = float(np.median([d.height_norm for d in detections])) if detections else 0.1
    # Row threshold tuned to be forgiving but robust across different scales
    row_threshold = max(0.03, min(0.12, median_h * 0.75))

    rows: List[List[Detection]] = []
    current_row: List[Detection] = []
    current_row_y_sum = 0.0

    for det in sorted_dets:
        if not current_row:
            current_row = [det]
            current_row_y_sum = det.y_center_norm
            continue

        row_mean_y = current_row_y_sum / len(current_row)
        if abs(det.y_center_norm - row_mean_y) <= row_threshold:
            current_row.append(det)
            current_row_y_sum += det.y_center_norm
        else:
            rows.append(current_row)
            current_row = [det]
            current_row_y_sum = det.y_center_norm

    if current_row:
        rows.append(current_row)

    return rows


def match_avatars_to_guesses(avatars: List[Detection], guesses: List[Detection]) -> List[Tuple[Detection, Detection]]:
    matches: List[Tuple[Detection, Detection]] = []
    if not avatars or not guesses:
        return matches

    # Build rows using both avatars and guesses for robust grouping
    combined = avatars + guesses
    rows = group_rows_by_y(combined)

    for row in rows:
        row_avatars = [d for d in row if d.class_id == 0]
        row_guesses = [d for d in row if d.class_id == 1]
        if not row_avatars or not row_guesses:
            # Fallback: try nearest neighbor across all guesses if one side missing in this row
            for av in row_avatars:
                nearest = min(guesses, key=lambda g: (abs(g.y_center_norm - av.y_center_norm) * 2.0 + abs(g.x_center_norm - av.x_center_norm)))
                matches.append((av, nearest))
            continue

        row_avatars_sorted = sorted(row_avatars, key=lambda d: d.x_center_norm)
        row_guesses_sorted = sorted(row_guesses, key=lambda d: d.x_center_norm)

        if len(row_avatars_sorted) == len(row_guesses_sorted):
            for av, gu in zip(row_avatars_sorted, row_guesses_sorted):
                matches.append((av, gu))
        else:
            # Greedy nearest matching within the row, without reuse of guesses
            unused = set(range(len(row_guesses_sorted)))
            for av in row_avatars_sorted:
                best_j = None
                best_cost = float("inf")
                for j in unused:
                    gu = row_guesses_sorted[j]
                    cost = abs(gu.y_center_norm - av.y_center_norm) * 2.0 + abs(gu.x_center_norm - av.x_center_norm)
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j
                if best_j is not None:
                    matches.append((av, row_guesses_sorted[best_j]))
                    unused.remove(best_j)

    # Deduplicate in case the fallback caused duplicates
    unique = []
    seen = set()
    for av, gu in matches:
        key = (id(av), id(gu))
        if key in seen:
            continue
        seen.add(key)
        unique.append((av, gu))
    return unique


def draw_matches(
    image: np.ndarray,
    matches: List[Tuple[Detection, Detection]],
) -> np.ndarray:
    h, w = image.shape[:2]
    vis = image.copy()

    # Colors in BGR
    color_avatar = (255, 128, 0)  # blue-ish
    color_guess = (0, 200, 0)     # green
    color_link = (0, 220, 255)    # yellow-ish

    for idx, (av, gu) in enumerate(matches):
        ax1, ay1, ax2, ay2 = av.to_bbox_px(w, h)
        gx1, gy1, gx2, gy2 = gu.to_bbox_px(w, h)

        cv2.rectangle(vis, (ax1, ay1), (ax2, ay2), color_avatar, 2)
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), color_guess, 2)

        acx, acy = av.center_px(w, h)
        gcx, gcy = gu.center_px(w, h)
        cv2.circle(vis, (acx, acy), 3, color_avatar, -1)
        cv2.circle(vis, (gcx, gcy), 3, color_guess, -1)

        cv2.line(vis, (acx, acy), (gcx, gcy), color_link, 2)
        cv2.putText(
            vis,
            f"{idx}",
            (min(acx, gcx) + 5, min(acy, gcy) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def main() -> int:
    # Defaults for this workspace
    default_image = os.path.abspath(os.path.join("images", "horizontal.png"))
    default_labels = os.path.abspath(os.path.join("runs", "detect", "predictions", "labels", "horizontal.txt"))
    default_output = os.path.abspath(os.path.join("outputs", "horizontal_matches.png"))

    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    labels_path = sys.argv[2] if len(sys.argv) > 2 else default_labels
    output_path = sys.argv[3] if len(sys.argv) > 3 else default_output

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1
    if not os.path.exists(labels_path):
        print(f"Labels not found: {labels_path}")
        return 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 1

    detections = load_yolo_labels(labels_path)
    avatars = [d for d in detections if d.class_id == 0]
    guesses = [d for d in detections if d.class_id == 1]

    matches = match_avatars_to_guesses(avatars, guesses)
    vis = draw_matches(image, matches)

    cv2.imwrite(output_path, vis)
    # cv2.imshow("Matches", vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"Saved visualization to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


