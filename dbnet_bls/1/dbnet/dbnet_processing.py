import numpy as np
from typing import List
from .dbnet import DBPostProcessor, db_resnet50
from .preprocessor import PreProcessor

det_predictor = db_resnet50(pretrained=False, assume_straight_pages=True)
det_predictor.eval()

# def _det_predictor_jit(image):
#     print('image.shape', image.shape)
#     prob_map = det_predictor_jit(image.float())
#     return prob_map


def _extract_crops(img: np.ndarray, boxes: np.ndarray, channels_last: bool = True) -> List[np.ndarray]:
    """Created cropped images from list of bounding boxes
    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one
    Returns:
        list of cropped images
    """
    cord = []
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    h, w = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != int:
        _boxes[:, [0, 2]] *= w
        _boxes[:, [1, 3]] *= h
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
    if channels_last:
        for box in _boxes:
            cord.append([box[1], box[3], box[0], box[2]])
        #   print([box[1], box[3], box[0], box[2]])
        return cord
    else:
        for box in _boxes:
            cord.append([box[1], box[3], box[0], box[2]])
        #   print([box[1], box[3], box[0], box[2]])
        return cord


def det_preprocess(img):
    '''wow'''
    pages = [img]
    if any(page.ndim != 3 for page in pages):
        raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

    pre_processor = PreProcessor(det_predictor.cfg['input_shape'][1:], batch_size=2,
                                mean=det_predictor.cfg['mean'], std=det_predictor.cfg['std'])
    processed_batches = pre_processor(pages)
    return processed_batches



def det_postprocess(prob_map):
    postprocessor = DBPostProcessor(assume_straight_pages=True)
    preds = [preds[0] for preds in postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy())]
    return preds



def detect_words(img, predicted_batches):
    pages = [img]
    loc_preds = [pred for batch in predicted_batches for pred in batch]

    channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray)
    crops = [
            _extract_crops(page, _boxes[:, :4], channels_last=channels_last)  # type: ignore[operator]
            for page, _boxes in zip(pages, loc_preds)
        ]
    return crops