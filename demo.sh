absolute_path_to_dataset="./dataset/val_openimage_v6"
export DETECTRON2_DATASETS=${absolute_path_to_dataset}

for idx in 35 37 39 41 43 45; do
    # Encoding and Decoding
    python test.py -i ${idx} -m feature_coding
    # Inference and Evaluation
    python test.py -i ${idx} -m evaluation
done