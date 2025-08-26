import argparse
import colorsys
import re
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract



def extract_wordle_grid(image: np.ndarray | Image.Image,
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

    # (no GUI side-effects)

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

def render_grid_overlay(image_path, grid):
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            draw.text((j*im.width/5, i*im.height/6), cell, fill="red")
    im.show()

def summarize_grid(grid):
    num_guesses = 0
    solved = False
    for row in grid:
        if row.count('E') != 5:
            num_guesses += 1
        if row.count('G') == 5:
            solved = True
            break
    return num_guesses, solved

def extract_wordle_number(image: np.ndarray | Image.Image,
                          top_band_ratio: float = 0.18,
                          center_width_ratio: float = 0.62,
                          debug: bool = False) -> Optional[int]:
    """
    Extract the "Wordle No. ####" number from the top banner using Tesseract.

    Heuristics:
    - Crop the top band of the image (default ~18% height) and center it
      horizontally to avoid avatars/columns.
    - Apply several robust preprocessing pipelines and OCR settings.
    - Parse with regex, correct common OCR confusions, and majority-vote
      across passes.

    Returns an integer Wordle number if detected, otherwise None.
    """

    # ---------- normalize to RGB uint8 ----------
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
    else:
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            if arr.ndim == 2:
                rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            else:
                rgb = arr.astype(np.uint8)

    H, W = rgb.shape[:2]
    if H == 0 or W == 0:
        return None

    # ---------- crop top banner (centered horizontally) ----------
    top_h = max(10, int(round(H * top_band_ratio)))
    mid_x = W // 2
    half_w = int(round((W * center_width_ratio) / 2))
    x0 = max(0, mid_x - half_w)
    x1 = min(W, mid_x + half_w)
    banner = rgb[0:top_h, x0:x1]

    # If banner is too dark/empty, slightly expand
    if banner.mean() < 10 and top_h < int(0.30 * H):
        top_h2 = int(0.30 * H)
        banner = rgb[0:top_h2, x0:x1]

    # ---------- build preprocess variants ----------
    def to_gray(a: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    def clahe_enhance(g: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(g)

    def invert_if_dark_bg(g: np.ndarray) -> np.ndarray:
        # If background is dark and text bright, invert to black-text on white
        return cv2.bitwise_not(g) if g.mean() < 128 else g

    def otsu(g: np.ndarray) -> np.ndarray:
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def morph_close(g: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(g, cv2.MORPH_CLOSE, k, iterations=1)

    def upscale(a: np.ndarray, scale: float) -> np.ndarray:
        return cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = to_gray(banner)
    variants: list[np.ndarray] = []
    # Variant A: CLAHE -> invert -> OTSU
    va = otsu(invert_if_dark_bg(clahe_enhance(gray)))
    variants.append(va)
    # Variant B: plain invert + OTSU
    vb = otsu(invert_if_dark_bg(gray))
    variants.append(vb)
    # Variant C: CLAHE only
    vc = clahe_enhance(gray)
    variants.append(vc)
    # Variant D: OTSU then close
    vd = morph_close(va)
    variants.append(vd)

    # Prepare upscaled versions to aid OCR on small text
    base_variants = list(variants)
    for v in base_variants:
        variants.append(upscale(v, 1.6))
        variants.append(upscale(v, 2.2))

    # ---------- OCR passes ----------
    psm_options = [7, 6, 13]  # single line, block, raw line
    configs = [
        lambda psm: f"-l eng --oem 3 --psm {psm} "+
                    "-c tessedit_char_whitelist=0123456789WordleNo.# ",
        lambda psm: f"-l eng --oem 3 --psm {psm}",
    ]

    # Regex patterns capturing the number
    patterns = [
        re.compile(r"(?i)wordle\s*(?:no\.?|#)?\s*(\d{1,5})"),
        re.compile(r"(?i)(?:no\.?|#)\s*(\d{1,5})"),
        re.compile(r"(\d{3,5})"),
    ]

    def normalize_ocr_text(t: str) -> str:
        # Common OCR confusions: O→0, l→1, I→1, S→5, B→8
        t = t.replace("O", "0").replace("o", "0")
        t = t.replace("l", "1").replace("I", "1")
        t = t.replace("S", "5")
        return t

    candidates: list[int] = []
    texts: list[str] = []
    for img_variant in variants:
        for psm in psm_options:
            for cfg_fn in configs:
                cfg = cfg_fn(psm)
                try:
                    txt = pytesseract.image_to_string(img_variant, config=cfg)
                except Exception:
                    continue
                if not txt:
                    continue
                txt_norm = normalize_ocr_text(txt)
                texts.append(txt_norm)
                for pat in patterns:
                    m = pat.search(txt_norm)
                    if m:
                        num_str = m.group(1)
                        try:
                            num = int(num_str)
                        except ValueError:
                            continue
                        # Plausibility filter: Wordle numbers are 1..10000 (generous)
                        if 1 <= num <= 10000:
                            candidates.append(num)
                        break

    if debug:
        # Print last few OCR texts for troubleshooting during development
        print("OCR texts (sample):", texts[-5:])

    if not candidates:
        return None

    # Majority vote by frequency; tiebreaker: highest number of digits then min value
    freq = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1
    best = sorted(freq.items(), key=lambda kv: (-kv[1], -len(str(kv[0])), kv[0]))[0][0]
    return int(best)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    grid = extract_wordle_grid(img)
    render_grid_overlay(args.image, grid)

    num_guesses, solved = summarize_grid(grid)

    print(f"num_guesses: {num_guesses}")
    print(f"solved: {solved}")

    # for row in grid:
    #     print(" ".join(row))

if __name__ == "__main__":
    main()