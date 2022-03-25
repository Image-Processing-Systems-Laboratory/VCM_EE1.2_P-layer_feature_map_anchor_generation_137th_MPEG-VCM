import glob
import json
import os
import shutil
from datetime import datetime

import imagesize
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import numpy as np
import torch
import matplotlib.gridspec as gridspec

def simple_filename(filename_ext):
    filename_base = os.path.basename(filename_ext)
    filename_noext = os.path.splitext(filename_base)[0]
    return filename_noext

def model_loader(settings):
    cfg = get_cfg()
    cfg.merge_from_file(settings["yaml_path"])
    cfg.MODEL.WEIGHTS = settings["pkl_path"]
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    model = DefaultPredictor(cfg).model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, cfg

def pick_coco_exp(name, targetlist):
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.makedirs(name, exist_ok=True)

    coco_path = os.environ["DETECTRON2_DATASETS"]
    anno_path = "./dataset/annotations/instances_OpenImage_v6.json"    

    file_list = glob.glob(os.path.join(coco_path, "*.jpg"))
    file_list = [x for x in file_list if simple_filename(x) in targetlist]

    file_name_list = [os.path.basename(x) for x in file_list]
    with open(anno_path, "r") as anno_file:
        coco_json = json.load(anno_file)
    my_json = {}
    my_json["info"] = coco_json["info"]
    my_json["licenses"] = coco_json["licenses"]
    my_json["images"] = []
    my_json["annotations"] = []
    my_json["categories"] = coco_json["categories"]

    my_json["images"].extend(
        [x for x in coco_json["images"] if x["file_name"] in file_name_list]
    )
    image_id_list = [x["id"] for x in my_json["images"]]
    my_json["annotations"].extend(
        [x for x in coco_json["annotations"] if x["image_id"] in image_id_list]
    )

    for filepath in file_list:
        shutil.copy(filepath, name)
    with open(f"{name}/my_anno.json", "w") as my_file:
        my_file.write(json.dumps(my_json))
    register_coco_instances(name, {}, f"{name}/my_anno.json", name)

def print_settings(settings, index):
    model_name = settings["model_name"]
    VTM_param = settings["VTM"]
    print()
    print("Evaluation of proposed methods for:", model_name.upper())
    print(f"Settings ID: {index}")
    print(f"VTM paramerters      : {VTM_param}")


import cv2
import numpy as np

def save_feature_map(filename, features):
    features_draw = features.copy()
    del features_draw["p6"]
    _save_feature_map(filename, features_draw)

def _save_feature_map(filename, features, debug=False):
    feat = [features["p2"].squeeze(), features["p3"].squeeze(), features["p4"].squeeze(), features["p5"].squeeze()]
    width_list = [16, 32, 64, 128]
    height_list = [16, 8, 4, 2]
    tile_big = np.empty((0, feat[0].shape[2] * width_list[0]))
    for blk, width, height in zip(feat, width_list, height_list):
        big_blk = np.empty((0, blk.shape[2] * width))
        for row in range(height):
            big_blk_col = np.empty((blk.shape[1], 0))
            for col in range(width):
                tile = blk[col + row * width].cpu().numpy()
                if debug:
                    cv2.putText(
                        tile,
                        f"{col + row * width}",
                        (32, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                big_blk_col = np.hstack((big_blk_col, tile))
            big_blk = np.vstack((big_blk, big_blk_col))
        tile_big = np.vstack((tile_big, big_blk))
    tile_big = tile_big.astype(np.uint16)
    cv2.imwrite(filename, tile_big)

def result_in_list(settings, number, result, set_index):
    res = list(result.values())[0]
    ap = res["AP"]
    ap50 = res["AP50"]
    aps = res["APs"]
    apm = res["APm"]
    apl = res["APl"]

    return [
        datetime.now(),
        set_index,
        number,
        f"{ap:.3f}",
        f"{ap50:.3f}",
        f"{aps:.3f}",
        f"{apm:.3f}",
        f"{apl:.3f}",
    ]

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
