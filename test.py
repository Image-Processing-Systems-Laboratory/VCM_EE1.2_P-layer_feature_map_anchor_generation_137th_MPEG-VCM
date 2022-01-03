import argparse
import csv
import glob
import json
import os

import utils
from eval import DetectEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", default=1, type=int)
    parser.add_argument("-n", "--number", default=5000, type=int)
    parser.add_argument("-m", "--mode")

    args = parser.parse_args()
    set_idx = args.index
    number = args.number
    mode = args.mode

    with open(f"settings/{set_idx}.json", "r") as setting_json:
        settings = json.load(setting_json)

    if settings["model_name"] == "x101":
        methods_eval = DetectEval(settings, set_idx)
        picklist = sorted(glob.glob(os.path.join(os.environ["DETECTRON2_DATASETS"], "*.jpg")))[:number]
        picklist = [utils.simple_filename(x) for x in picklist]
        methods_eval.prepare_part(picklist, data_name="pick")

    if mode == "feature_coding":
        filenames = methods_eval.feature_coding()

    elif mode == "evaluation":
        filenames = glob.glob(f"feature/{set_idx}_rec/*.png")
        methods_eval.evaluation(filenames)
