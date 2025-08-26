# assume image already predicted and label exists in same folder as original image

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image

from avatar_matcher import annotate_labels_with_matches
from match import Detection, load_yolo_labels, match_avatars_to_guesses
from wordle_extract import extract_wordle_grid, extract_wordle_number, summarize_grid

@dataclass
class Player:
    user_id: str
    username: str
    user_id_conf: float
    avatar_bbox: List[float]
    avatar_conf: float
    guess_bbox: List[float]
    guess_conf: float
    wordle_id: int
    wordle_grid: str
    wordle_num_guesses: int
    wordle_solved: bool

    def __init__(self, avatar: Detection, guess: Detection, wordle_id: int, wordle_grid: str, num_guesses: int, solved: bool):
        self.user_id = avatar.user_id
        self.username = avatar.username
        self.user_id_conf = avatar.user_id_conf
        self.avatar_bbox = [avatar.x_center_norm, avatar.y_center_norm, avatar.width_norm, avatar.height_norm]
        self.avatar_conf = avatar.confidence
        self.guess_bbox = [guess.x_center_norm, guess.y_center_norm, guess.width_norm, guess.height_norm]
        self.guess_conf = guess.confidence
        self.wordle_id = wordle_id
        self.wordle_grid = wordle_grid
        self.wordle_num_guesses = num_guesses
        self.wordle_solved = solved

    def to_json(self):
        return {
            "player": {
                "username": self.username,
                "conf": self.user_id_conf,
            },
            "predictions": {
                "avatar": {
                    "bbox": self.avatar_bbox,
                    "conf": self.avatar_conf
                },
                "guess": {
                    "bbox": self.guess_bbox,
                    "conf": self.guess_conf
                }
            },
            "wordle": {
                "id": self.wordle_id,
                "grid": self.wordle_grid,
                "num_guesses": self.wordle_num_guesses,
                "solved": self.wordle_solved
            }
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("pfp_db"))
    parser.add_argument("--output", type=Path, default=Path("matches.json"))
    args = parser.parse_args()
    
    # add match to labels
    annotate_labels_with_matches(
        index_dir=args.index_dir,
        image_path=args.image,
        labels_path=args.labels,
    )
    print(f"Done annotating {args.labels} with matches")

    # load labels
    detections = load_yolo_labels(args.labels)
    print(f"Loaded {len(detections)} detections from {args.labels}")
    avatars = [d for d in detections if d.class_id == 0]
    guesses = [d for d in detections if d.class_id == 1]

    # match avatars to guesses
    matches = match_avatars_to_guesses(avatars, guesses)
    print(f"Matched {len(matches)} avatars to guesses")

    # load image
    image = cv2.imread(args.image)
    h, w = image.shape[:2]

    # find players and get wordle info
    players = []
    for avatar, guess in matches:
        x1, y1, x2, y2 = guess.to_bbox_px(w, h)
        wordle_grid = extract_wordle_grid(image, (x1, y1, x2, y2))
        wordle_id = extract_wordle_number(image)
        num_guesses, solved = summarize_grid(wordle_grid)
        print(f"Extracted wordle grid: {wordle_grid} for {avatar.username} with {num_guesses} guesses and solved={solved}")
        player = Player(avatar, guess, wordle_id, wordle_grid, num_guesses, solved)
        players.append(player)

    # write matches to json keyed by user_id for easy access
    players_by_id = {}
    for idx, p in enumerate(players):
        key = p.user_id if p.user_id else (p.username if p.username else f"player_{idx}")
        players_by_id[key] = p.to_json()
    with open(args.output, "w") as f:
        json.dump(players_by_id, f, indent=2)


if __name__ == "__main__":
    main()