import cv2
import numpy as np
import json


def read_template(name, type):
    path_template = f'templates/{name}.png'
    path_keypoint = f'templates/{name}.txt'

    template = cv2.cvtColor(cv2.imread(path_template), cv2.COLOR_BGR2RGB)
    template_keypoint = np.loadtxt(path_keypoint, dtype=np.int32, delimiter=',')

    if type == 'b':
        template = cv2.flip(template, 1)
        template_keypoint[:, 0] = template.shape[1] - template_keypoint[:, 0]

    return template, template_keypoint


def affine(image, image_keypoint, template, template_keypoint, expand_ratio):
    # pre process
    template_white = np.full(template.shape, 255, dtype=np.uint8)

    # resize
    scale = np.linalg.norm(image_keypoint[0] - image_keypoint[2]) / np.linalg.norm(
        template_keypoint[0] - template_keypoint[2])
    scale *= expand_ratio
    M = np.array([
        [scale, 0, 0],
        [0, scale, 0],
    ], dtype=np.float32)
    size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
    template_white = cv2.warpAffine(template_white, M, size)
    template = cv2.warpAffine(template, M, size)
    template_keypoint = template_keypoint * scale

    # move
    distance = image_keypoint[0] - template_keypoint[0]
    M = np.array([
        [1, 0, distance[0]],
        [0, 1, distance[1]],
    ], dtype=np.float32)
    template_white = cv2.warpAffine(template_white, M, image.shape[1::-1])
    template = cv2.warpAffine(template, M, image.shape[1::-1])

    # rotate
    v1 = template_keypoint[2] - template_keypoint[0]
    v2 = image_keypoint[2] - image_keypoint[0]

    angle = np.rad2deg(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    if np.cross(v1, v2) > 0:
        angle = -angle

    M = cv2.getRotationMatrix2D(image_keypoint[0], angle, 1)
    template_white = cv2.warpAffine(template_white, M, image.shape[1::-1])
    template = cv2.warpAffine(template, M, image.shape[1::-1])

    return template_white, template


def control(image, image_bbox_mask, res_skeleton, expand_ratio, template_name, image_skeleton, force):
    # left hand
    template_white_l, template_l = np.zeros_like(image), np.zeros_like(image)
    image_keypoint = np.array(json.loads(res_skeleton))[0]
    if np.max(image_keypoint) < 1:
        image_keypoint[:, 0], image_keypoint[:, 1] = image_keypoint[:, 1] * image.shape[1], image_keypoint[:, 0] * image.shape[0]
        # check hand be detected by yolo
        be_detected = False
        for keypoint in image_keypoint:
            if np.sum(image_bbox_mask[int(keypoint[1]), int(keypoint[0])]) > 1:
                be_detected = True
                break
        if be_detected or force:

            if np.cross(image_keypoint[2] - image_keypoint[0], image_keypoint[3] - image_keypoint[1]) > 0:
                template, template_keypoint = read_template(template_name, 'a')
            else:
                template, template_keypoint = read_template(template_name, 'b')

            template_white_l, template_l = affine(image, image_keypoint, template, template_keypoint, expand_ratio)

    # right hand
    template_white_r, template_r = np.zeros_like(image), np.zeros_like(image)

    image_keypoint = np.array(json.loads(res_skeleton))[1]
    if np.max(image_keypoint) < 1:
        image_keypoint[:, 0], image_keypoint[:, 1] = image_keypoint[:, 1] * image.shape[1], image_keypoint[:, 0] * image.shape[0]
        # check hand be detected by yolo
        be_detected = False
        for keypoint in image_keypoint:
            if np.sum(image_bbox_mask[int(keypoint[1]), int(keypoint[0])]) > 1:
                be_detected = True
                break
        if be_detected or force:

            if np.cross(image_keypoint[2] - image_keypoint[0], image_keypoint[3] - image_keypoint[1]) > 0:
                template, template_keypoint = read_template(template_name, 'a')
            else:
                template, template_keypoint = read_template(template_name, 'b')

            template_white_r, template_r = affine(image, image_keypoint, template, template_keypoint, expand_ratio)


    # control image
    template = np.clip(template_l.astype(np.int32) + template_r.astype(np.int32), 0, 255).astype(np.uint8)
    # combine image
    image_combine = np.clip(image.astype(np.int32) + template, 0, 255).astype(np.uint8)
    # control mask
    template_white = np.clip(template_white_l.astype(np.int32) + template_white_r.astype(np.int32), 0, 255).astype(np.uint8)
    # union mask
    image_union_mask = np.clip(image_bbox_mask.astype(np.int32) + template_white.astype(np.int32), 0, 255).astype(np.uint8)

    # visualization
    image_visualization = np.clip(
        template.astype(np.int32) * - 1 +
        image_skeleton.astype(np.int32) * 0.6 +
        image_bbox_mask.astype(np.int32) * 0.3 +
        template_white.astype(np.int32) * 0.3
        , 0, 255).astype(np.uint8)

    return image_visualization, image_combine, template, template_white, image_union_mask

