from PIL import Image
import numpy as np
import colorsys
import argparse
from PIL import ImageDraw
from PIL import Image
import numpy as np
import colorsys
import cv2

from PIL import Image
import numpy as np
import colorsys
from typing import Tuple, Optional

from PIL import Image
import numpy as np
import colorsys

def extract_wordle_grid_with_empty(image: np.ndarray | Image.Image,
                                   bbox_px: Optional[Tuple[int, int, int, int]] = None,
                                   rows=6, cols=5,
                                   empty_token='E', gray_token='B',
                                   pad: float = 0.06):
    """
    Return a rows×cols grid of {'G','Y','B','E'} from a Wordle screenshot.

    'E' marks truly empty (unplayed) squares, distinct from gray 'B' wrong guesses.
    """

    # Normalize to RGB float32 array in [0,1]
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Assume BGR from OpenCV; convert to RGB
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            rgb = arr.astype(np.float32) / 255.0

    # Optional crop using bbox with padding
    if bbox_px is not None:
        x1, y1, x2, y2 = bbox_px
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        px = int(round(bw * pad))
        py = int(round(bh * pad))
        left = max(0, x1 - px)
        top = max(0, y1 - py)
        right = min(rgb.shape[1] - 1, x2 + px)
        bottom = min(rgb.shape[0] - 1, y2 + py)
        rgb = rgb[top:bottom+1, left:right+1]

    # show the cropped image
    cv2.imshow("cropped", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- helpers ----------
    def _rgb_to_hsv_array(a):
        hsv = np.zeros_like(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                hsv[i, j] = colorsys.rgb_to_hsv(*a[i, j])
        return hsv

    def _sobel_edges(gray):
        gx = np.zeros_like(gray); gy = np.zeros_like(gray)
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
        ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)
        H, W = gray.shape
        for y in range(1, H-1):
            for x in range(1, W-1):
                region = gray[y-1:y+2, x-1:x+2]
                gx[y, x] = np.sum(region * kx)
                gy[y, x] = np.sum(region * ky)
        return np.hypot(gx, gy)

    def _bands_from_profile(profile, n, frac=0.22):
        """Threshold a 1-D profile to bands, coalesce/split to exactly n."""
        if profile.max() == 0:
            step = len(profile) / n
            return [(int(i*step), int((i+1)*step)-1) for i in range(n)]
        thresh = profile.max() * frac
        bands = []; inside = False; start = 0
        for i, v in enumerate(profile):
            if (v >= thresh) and not inside:
                inside = True; start = i
            elif (v < thresh) and inside:
                inside = False; bands.append((start, i-1))
        if inside:
            bands.append((start, len(profile)-1))
        # coalesce/split to n
        bands = [(a, b, (a+b)/2) for a, b in bands]
        bands.sort(key=lambda x: x[2])
        while len(bands) > n:
            dists = [(bands[i+1][2]-bands[i][2], i) for i in range(len(bands)-1)]
            _, i = min(dists, key=lambda t: t[0])
            a1, b1, _ = bands[i]; a2, b2, _ = bands[i+1]
            bands[i] = (min(a1, a2), max(b1, b2), (min(a1, a2)+max(b1, b2))/2)
            del bands[i+1]
        if len(bands) < n:
            step = (bands[-1][1] - bands[0][0] + 1) / n
            base = bands[0][0]
            return [(int(base+i*step), int(base+(i+1)*step)-1) for i in range(n)]
        return [(a, b) for a, b, _ in bands]

    def _classify_cell(h, s, v, v_bg, dv=0.07):
        # empty: near-background & low saturation
        if s < 0.18 and v <= v_bg + dv:
            return empty_token
        # gray: low saturation or dark, but brighter than "empty"
        if s < 0.22 or v < 0.30:
            return gray_token
        # green / yellow bands (tolerant for small screenshots)
        if 80 <= h <= 140:
            return 'G'
        if 35 <= h <= 75:
            return 'Y'
        # fallback by nearest canonical hue
        dg = min(abs(h-100), 360-abs(h-100))
        dy = min(abs(h-55),  360-abs(h-55))
        return 'G' if dg < dy else 'Y'

    # ---------- load & preprocess ----------
    a = rgb
    hsv = _rgb_to_hsv_array(a)
    H = hsv[:, :, 0] * 360.0
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    gray = 0.299*a[:, :, 0] + 0.587*a[:, :, 1] + 0.114*a[:, :, 2]
    edges = _sobel_edges(gray)

    # ---------- find grid bbox ----------
    fg = (V > 0.20).astype(np.uint8)
    ed = (edges > edges.mean()).astype(np.uint8)
    mask = (fg | ed)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return [[empty_token]*cols for _ in range(rows)]  # degenerate

    ymin, ymax = max(0, ys.min()), min(H.shape[0]-1, ys.max())
    xmin, xmax = max(0, xs.min()), min(H.shape[1]-1, xs.max())

    # pad slightly to include outer grid lines
    pad_y = max(1, (ymax - ymin) // 40)
    pad_x = max(1, (xmax - xmin) // 40)
    ymin = max(0, ymin - pad_y); ymax = min(H.shape[0]-1, ymax + pad_y)
    xmin = max(0, xmin - pad_x); xmax = min(H.shape[1]-1, xmax + pad_x)

    # estimate background from a ring just outside the bbox
    # fall back to dark quantile if bbox touches edges
    ring_width = max(2, min(V.shape[0], V.shape[1])//50)
    strips = []
    if ymin - ring_width >= 0: strips.append(V[ymin-ring_width:ymin, max(0,xmin-ring_width):min(V.shape[1],xmax+ring_width)])
    if ymax + 1 + ring_width <= V.shape[0]-1: strips.append(V[ymax+1:ymax+1+ring_width, max(0,xmin-ring_width):min(V.shape[1],xmax+ring_width)])
    if xmin - ring_width >= 0: strips.append(V[max(0,ymin-ring_width):min(V.shape[0],ymax+ring_width), xmin-ring_width:xmin])
    if xmax + 1 + ring_width <= V.shape[1]-1: strips.append(V[max(0,ymin-ring_width):min(V.shape[0],ymax+ring_width), xmax+1:xmax+1+ring_width])
    if strips:
        v_bg = float(np.median(np.concatenate([s.ravel() for s in strips])))
    else:
        v_bg = float(np.quantile(V, 0.10))  # fallback

    # crop to bbox
    Hc = H[ymin:ymax+1, xmin:xmax+1]
    Sc = S[ymin:ymax+1, xmin:xmax+1]
    Vc = V[ymin:ymax+1, xmin:xmax+1]

    # ---------- columns from projection; rows by even split ----------
    col_profile = (Vc > 0.20).sum(axis=0).astype(float)
    # add a little edge signal to better catch column boundaries
    col_edges = edges[ymin:ymax+1, xmin:xmax+1].sum(axis=0)
    if col_edges.max() > 0:
        col_profile += 0.5 * (col_edges / col_edges.max())
    col_bands = _bands_from_profile(col_profile, cols, frac=0.22)

    height = Hc.shape[0]
    step = height / rows
    row_bands = [(int(i*step), int((i+1)*step)-1) for i in range(rows)]

    # ---------- classify each cell ----------
    grid = []
    for ry0, ry1 in row_bands:
        row = []
        for cx0, cx1 in col_bands:
            # erode to avoid grid lines: keep center ~56%
            y0 = int(ry0 + (ry1-ry0)*0.22); y1 = int(ry1 - (ry1-ry0)*0.22)
            x0 = int(cx0 + (cx1-cx0)*0.22); x1 = int(cx1 - (cx1-cx0)*0.22)
            # clip
            y0 = max(0, y0); y1 = min(Hc.shape[0]-1, y1)
            x0 = max(0, x0); x1 = min(Hc.shape[1]-1, x1)
            h = float(np.median(Hc[y0:y1+1, x0:x1+1]))
            s = float(np.median(Sc[y0:y1+1, x0:x1+1]))
            v = float(np.median(Vc[y0:y1+1, x0:x1+1]))
            row.append(_classify_cell(h, s, v, v_bg))
        grid.append(row)
    return grid

def overlay_grid(image_path, grid):
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            draw.text((j*im.width/5, i*im.height/6), cell, fill="red")
    im.show()

def check_num_guesses_and_solved(grid):
    num_guesses = 0
    solved = False
    for row in grid:
        if row.count('E') != 5:
            num_guesses += 1
        if row.count('G') == 5:
            solved = True
            break
    return num_guesses, solved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    grid = extract_wordle_grid_with_empty(img)
    overlay_grid(args.image, grid)

    num_guesses, solved = check_num_guesses_and_solved(grid)

    print(f"num_guesses: {num_guesses}")
    print(f"solved: {solved}")

    # for row in grid:
    #     print(" ".join(row))

if __name__ == "__main__":
    main()