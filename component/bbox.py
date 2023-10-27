from ultralytics import YOLO
import cv2
import numpy as np

hand_dict = {0: 'Standard', 1: 'Non-Standard'}


def bbox(image, include_standard, mask_expand_ratio):
    model = YOLO('model/yolo.pt')
    solution = model.predict(image)

    # detect bbox
    result = []
    for det in solution[0]:
        x1, y1, x2, y2, conf, cls = list(det.boxes.data[0])
        bbox = [int(x1), int(y1), int(x2), int(y2), int(cls), float(conf), hand_dict[int(cls)]]
        result.append(bbox)

        print('BBox Result:', bbox, image.shape)

    # draw bbox
    for bbox in result:
        COLOR = (0, 255, 0) if bbox[4] == 0 else (255, 0, 0)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR, 2)
        cv2.putText(image, str(bbox[6]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2, cv2.LINE_AA)

    # get mask
    image_mask = np.zeros(image.shape, dtype=np.uint8)
    for bbox in result:
        # only non-standard
        if not include_standard and bbox[4] == 0:
            continue

        # expand
        bbox[0] = max(0, bbox[0] - int((bbox[2] - bbox[0]) * (mask_expand_ratio - 1) * 0.5))
        bbox[1] = max(0, bbox[1] - int((bbox[3] - bbox[1]) * (mask_expand_ratio - 1) * 0.5))
        bbox[2] = min(image.shape[1], bbox[2] + int((bbox[2] - bbox[0]) * (mask_expand_ratio - 1) * 0.5))
        bbox[3] = min(image.shape[0], bbox[3] + int((bbox[3] - bbox[1]) * (mask_expand_ratio - 1) * 0.5))

        image_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    # get bbox info
    bbox_info = f'Number of hands: {len(result)}\n'
    for i, hand in enumerate(result):
        bbox_info += f'Hand {i}: {hand[6]} with confidence {hand[5]:.4f}.\n'

    return bbox_info, image, image_mask