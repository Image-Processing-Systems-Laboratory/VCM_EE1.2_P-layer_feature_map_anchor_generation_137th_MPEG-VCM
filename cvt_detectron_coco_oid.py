# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

#
# convert the output of coco format to OpenImages format 
#

import numpy as np
import pandas as pd
import os, json

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--coco_output_file', type=str, default='coco_classes.txt', \
#   help='prediction output file in COCO format')
# parser.add_argument('--oid_output_file', type=str, default='output_oid.txt', \
#   help='prediction output file in OpenImages format')
# parser.add_argument('--selected_classes', type=str, default='selected_classes.txt', \
#   help='A file contain the list of selected classes')

# args = parser.parse_args()

# # prediciton output
# coco_fname = args.coco_output_file
# oid_fname = args.oid_output_file



def conversion(index):
  coco_fname = f"output/{index}_coco.txt"
  oid_fname = f"output/{index}_oi.txt"
  selected_coco_classes_fname = "oi_eval/selected_classes.txt"

  # unify class name by replacing space to underscore
  def unify_name(class_name):
    return class_name.replace(' ', '_')

  # selected coco classes
  with open(selected_coco_classes_fname, 'r') as f:
    selected_classes = [unify_name(x) for x in f.read().splitlines()]


  # load coco output data file
  coco_output_data = pd.read_csv(coco_fname)

  # generate output
  of = open(oid_fname, 'w')

  #write header
  #of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
  of.write(','.join(coco_output_data.columns)+'\n')

  # iterate all input files
  for idx, row in coco_output_data.iterrows():
    fields = row.tolist()
    #coco_id = fields[1]
    coco_id = unify_name(row['LabelName'])
    if coco_id in selected_classes:
      oid_id = coco_id
      row['LabelName'] = oid_id
      o_line = ','.join(map(str,row))
      of.write(o_line + '\n')

  of.close()


