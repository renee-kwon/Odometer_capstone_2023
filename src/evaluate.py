#!/usr/bin/env python
# coding: utf-8

import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os


from predict import predict
from utils import remove_first_dirs

# Load model
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("cuda not found, using CPU")
    device = torch.device("cpu")

odo_model_pth = "outputs/odo.pt"
digit_model_pth = "outputs/digit.pt"

odo_model = YOLO(odo_model_pth)
digit_model = YOLO(digit_model_pth)


## ICBC - Odometer Photo Readings


root = "tmp/Test Data/"


# get list of downloaded test images
test_dir = os.path.join(root, "Odometer_Photo_Readings")
paths = []
for path, subdirs, files in os.walk(test_dir):
    for name in files:
        paths.append(os.path.join(path, name))
paths = pd.DataFrame({"PHOTO_FILE_PATH": paths})
paths["PHOTO_FILE_PATH"] = paths["PHOTO_FILE_PATH"].apply(
    lambda x: remove_first_dirs(x, 2)
)


# get list of test image ground truths
gt = pd.read_excel(os.path.join(test_dir, "Odometer_Photo_reading.xlsx"))
gt["PHOTO_FILE_PATH"] = gt["PHOTO_FILE_PATH"].replace(r"\\", "/", regex=True)

# merge downloaded test images with ground truths
test_df = gt.merge(paths, how="inner", on="PHOTO_FILE_PATH")

assert len(test_df) != 0, "No image files were found, check directory and excel file"

# Predict on each image
print("Starting predictions on " + str(test_dir))
print("This takes like 3hrs on a M2 Air")
# TODO: There must be a way of increasing performance through parallel processing
results = []
skipped = []
for file in list(test_df["PHOTO_FILE_PATH"]):
    try:
        results.append(
            predict(odo_model, digit_model, os.path.join(root, file), device)
        )
    except:
        skipped.append(file)
print("Finished predictions on " + str(test_dir))


# merge outputs with ground truths
results_df = pd.DataFrame(results)
results_df["PHOTO_FILE_PATH"] = results_df["PHOTO_FILE_PATH"].apply(
    lambda x: remove_first_dirs(x, 2)
)
results_df["value_conf"] = results_df["digits_conf"].apply(
    lambda x: np.mean(x) if len(x) > 0 else None
)
results_df = pd.merge(test_df, results_df, on="PHOTO_FILE_PATH", how="inner")


# save results
results_df.to_json("outputs/Test_Data_Results.json")
pd.DataFrame(skipped, columns=["skipped paths"]).to_json(
    "outputs/Test_Data_Skipped.json"
)


## Non-ICBC Images - Non-Odometer_Images

root = "tmp/Test Data/"


# get list of downloaded test images
test_dir = os.path.join(root, "Non-Odometer_Images")
paths = []
for path, subdirs, files in os.walk(test_dir):
    for name in files:
        paths.append(os.path.join(path, name))
paths = pd.DataFrame({"PHOTO_FILE_PATH": paths})

paths["PHOTO_FILE_PATH"] = paths["PHOTO_FILE_PATH"].apply(
    lambda x: remove_first_dirs(x, 2)
)
paths = paths[paths["PHOTO_FILE_PATH"].str.endswith((".jpg", ".jpeg"))]

# add ground "truths"
test_df = paths
test_df["VIN"] = None
test_df["VERIFIED_ODOMETER_READING"] = -1

assert len(test_df) != 0, "No image files were found, check directory and excel file"

# Predict on each image
print("Starting predictions on " + str(test_dir))
print("This takes like 3hrs on a M2 Air")
# TODO: There must be a way of increasing performance through parallel processing
results = []
skipped = []
for file in list(test_df["PHOTO_FILE_PATH"]):
    try:
        results.append(
            predict(odo_model, digit_model, os.path.join(root, file), device)
        )
    except:
        skipped.append(file)
print("Finished predictions on " + str(test_dir))

results_df = pd.DataFrame(results)

results_df["PHOTO_FILE_PATH"] = results_df["PHOTO_FILE_PATH"].apply(
    lambda x: remove_first_dirs(x, 2)
)
results_df["value_conf"] = results_df["digits_conf"].apply(
    lambda x: np.mean(x) if len(x) > 0 else None
)
results_df = pd.merge(test_df, results_df, on="PHOTO_FILE_PATH", how="inner")

pd.concat([results_df, pd.read_json("outputs/Test_Data_Results.json")]).reset_index(
    drop=True
).to_json("outputs/Test_Data_Results.json")
pd.concat(
    [
        pd.DataFrame(skipped, columns=["skipped paths"]),
        pd.read_json("outputs/Test_Data_Skipped.json"),
    ]
).reset_index(drop=True).to_json("outputs/Test_Data_Skipped.json")
print("results saved")
