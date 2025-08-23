# assume image already predicted and label exists in same folder as original image

import argparse
from dataclasses import dataclass
from pathlib import Path
from avatar_matcher import annotate_labels_with_matches
from match import match_avatars_to_guesses, load_yolo_labels, Detection
from typing import List, Tuple
import json
from wordle_extract import extract_wordle_grid_with_empty, check_num_guesses_and_solved
from PIL import Image
import cv2

@dataclass
class Player:
    user_id: str
    username: str
    user_id_conf: float
    avatar_bbox: List[float]
    avatar_conf: float
    guess_bbox: List[float]
    guess_conf: float
    wordle_grid: str
    wordle_num_guesses: int
    wordle_solved: bool

    def __init__(self, avatar: Detection, guess: Detection, wordle_grid: str, num_guesses: int, solved: bool):
        self.user_id = avatar.user_id
        self.username = avatar.user_id
        self.user_id_conf = avatar.user_id_conf
        self.avatar_bbox = [avatar.x_center_norm, avatar.y_center_norm, avatar.width_norm, avatar.height_norm]
        self.avatar_conf = avatar.confidence
        self.guess_bbox = [guess.x_center_norm, guess.y_center_norm, guess.width_norm, guess.height_norm]
        self.guess_conf = guess.confidence
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

    # load labels
    detections = load_yolo_labels(args.labels)
    avatars = [d for d in detections if d.class_id == 0]
    guesses = [d for d in detections if d.class_id == 1]

    # match avatars to guesses
    matches = match_avatars_to_guesses(avatars, guesses)

    # load image
    image = cv2.imread(args.image)
    h, w = image.shape[:2]

    players = []
    for avatar, guess in matches:
        x1, y1, x2, y2 = guess.to_bbox_px(w, h)
        wordle_grid = extract_wordle_grid_with_empty(image, (x1, y1, x2, y2))
        num_guesses, solved = check_num_guesses_and_solved(wordle_grid)
        player = Player(avatar, guess, wordle_grid, num_guesses, solved)
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