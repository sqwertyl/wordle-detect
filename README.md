### Setup
- Install dependencies:
  - [Optional] Create and activate python virtual environment
    - `python -m venv <name e.g. ".venv">`
    - `source <name>/bin/activate`
  - Python 3.9+
  - `pip install -r requirements.txt`

### Run detection
Use `detect.py` to run inference on an image or a directory of images.

Example (typical command):
```bash
python detect.py --source './images' --device 0 --weights './yolov9-s-wordle.pt' --name predictions --save-txt --save-crop
```

- **--source**: path to a single image (e.g., `./images/image.png`) or a folder (e.g., `./images`).
- **--device**: GPU index (e.g., `0`) or `cpu`.
- **--weights**: path to the `.pt` weights file.
- **--name**: run name; outputs go to `./runs/detect/<name>/`.
- **--save-txt**: write YOLO-format bounding boxes to `runs/detect/<name>/labels/*.txt`.
- **--save-crop**: save cropped detections to `runs/detect/<name>/crops/`.

Other options can be found in `detect.py`
Outputs are written under `./runs/detect/<name>/`.

### Match and visualize
Use `match.py` to load an original image and its YOLO labels, then save a visualization with matched rows/letters.

How to use:
```bash
python match.py <path to original image> <path to labels .txt file> <path to output visualization>
```
Example
```bash
python match.py ./images/image.png ./runs/detect/predictions/labels/image.txt ./outputs/image.png
```

### Credits
- This project uses components adapted from YOLOv9. See the original repository: https://github.com/WongKinYiu/yolov9