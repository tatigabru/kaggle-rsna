import numpy as np
from typing import List, Set, Dict, Tuple, Optional


def iou(box1: List[int], box2: List[int]) -> float:
    """
    Helper, calculates Intersection over union
    Args: 
        box1, box2: x, y, w, h of the boxes
    Output:   
        Intersection over union  
    """
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    if w1 * h1 <= 0 or w2 * h2 <= 0:
        return 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def test_iou() -> None:
    """Helper to test iou function"""
    box1 = [100, 100, 200, 200]
    box2 = [100, 100, 300, 200]
    assert abs(iou(box1, box2) - 2 / 3) < 1e-3


def map_iou(boxes_true: np.ndarray, boxes_pred: np.ndarray, scores: np.ndarray, thresholds: Tuple[float]=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)) -> Optional[float]:
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    Args:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate the mean average precision on
    Output:
        map: mean average precision of the image
    """

    # images with no ground truth bboxes are not included in the map score unless
    # there is a false positive detection
    # return None if both are empty, don't count the image in final evaluation
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0.0
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    return map_total / len(thresholds)


def test_map_iou() -> Optional[float]:
    """Helper to test map_iou function"""
    boxes_true = [100, 100, 200, 200]
    boxes_pred = [100, 100, 300, 200]
    boxes_true = np.reshape(boxes_true, (1, 4))
    boxes_pred = np.reshape(boxes_pred, (1, 4))
    result = map_iou(boxes_true, boxes_pred, [1], thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75))
    print("map_iou: ", result)
    return result


if __name__ == "__main__":
    test_iou()
    test_map_iou()
