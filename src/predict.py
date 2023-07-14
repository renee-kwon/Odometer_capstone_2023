import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def predict(
    odo_model,
    digit_model,
    img: str or np.ndarray,
    device,
    return_raw_preds=False,
    debug=False,
):
    # predict odo
    odo_result = odo_model.predict(
        source=img, device=device, imgsz=1280, max_det=1, conf=0
    )
    if debug:
        cv2.imshow(odo_result[0].plot())
    if len(odo_result[0].boxes.cls) == 0:
        # odometer not found
        pred = -2
        odo_conf = None
        digits = None
        digits_conf = None
    else:
        # odometer found
        odo_conf = odo_result[0].boxes.conf.cpu().numpy()[0]
        bbox = odo_result[0].boxes.xywh.cpu().numpy()[0]

        # open image and crop it
        cropped_img = crop(img, bbox, 1)
        if debug:
            cv2.imshow(cropped_img)

        # predict digits
        digit_result = digit_model.predict(
            source=cropped_img, device=device, imgsz=320, agnostic_nms=True, conf=0.3
        )
        if debug:
            cv2.imshow(digit_result[0].plot())

        indices = torch.argsort(digit_result[0].boxes.xywh[:, 0])
        digits = digit_result[0].boxes.cls[indices].cpu().numpy()
        digits_conf = digit_result[0].boxes.conf[indices].cpu().numpy()
        if debug:
            print(digits)
        if debug:
            print(digits_conf)

        pred = _list_to_num(digits)

    results = {
        "pred": pred,
        "odo_conf": odo_conf,
        "digits": digits,
        "digits_conf": digits_conf,
    }
    if isinstance(img, str):
        results["PHOTO_FILE_PATH"] = img
        results["raw_image"] = None
    elif isinstance(img, np.ndarray):
        results["PHOTO_FILE_PATH"] = None
        results["raw_image"] = img

    if return_raw_preds:
        results["odo_result"] = odo_result
        results["digit_results"] = digit_result

    return results


def crop(img: str or np.ndarray, xywh: list, pad) -> np.ndarray:
    if isinstance(img, str):
        image = cv2.imread(img)
    else:
        image = img
    x, y, w, h = (
        int(xywh[0]),
        int(xywh[1]),
        int(xywh[2] / 2 * pad),
        int(xywh[3] / 2 * pad),
    )
    return image[y - h : y + h, x - w : x + w]


def _list_to_num(arr: list[int]) -> int:
    if len(arr) != 0:
        if 10 in arr:
            arr = [x for x in arr if x != 10]
        if len(arr) == 7:
            arr = np.delete(arr, len(arr) - 1)
        string_elements = [str(int(x)) for x in arr]
        if len(string_elements) > 0:
            return int("".join(string_elements))
    else:
        return -1
