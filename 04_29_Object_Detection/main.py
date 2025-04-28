import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
# import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

# TODO - Exercise 1
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    ### START CODE HERE
    # Step 1: Compute box scores
    ##(≈ 1 line)
    box_scores = box_class_probs * box_confidence

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ##(≈ 2 lines)
    # IMPORTANT: set axis to -1
    box_classes = tf.argmax(box_class_probs, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ## (≈ 1 line)
    filtering_mask = [True if x >= threshold else False for x in box_class_scores]

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ## (≈ 3 lines)
    scores = [box_scores[i] for i in range(len(box_scores)) if filtering_mask[i]]
    boxes = [boxes[i] for i in range(len(boxes)) if filtering_mask[i]]
    classes = [box_classes[i] for i in range(len(box_classes)) if filtering_mask[i]]
    ### END CODE HERE

    return scores, boxes, classes

# BEGIN UNIT TEST
tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)

print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

assert type(scores) == EagerTensor, "Use tensorflow functions"
assert type(boxes) == EagerTensor, "Use tensorflow functions"
assert type(classes) == EagerTensor, "Use tensorflow functions"

assert scores.shape == (1789,), "Wrong shape in scores"
assert boxes.shape == (1789, 4), "Wrong shape in boxes"
assert classes.shape == (1789,), "Wrong shape in classes"

assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
assert classes[2].numpy() == 8, "Values are wrong on classes"

print("\033[92m All tests passed!\033[0m\n\n")
# END UNIT TEST


# # TODO - Exercise 2
# def iou(box1, box2):
#     """Implement the intersection over union (IoU) between box1 and box2
#
#     Arguments:
#     box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
#     box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
#     """
#
#     (box1_x1, box1_y1, box1_x2, box1_y2) = box1
#     (box2_x1, box2_y1, box2_x2, box2_y2) = box2
#
#     ### START CODE HERE
#     # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
#     ##(≈ 7 lines)
#     xi1 =
#     yi1 =
#     xi2 =
#     yi2 =
#     inter_width =
#     inter_height =
#     inter_area =
#
#     # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
#     ## (≈ 3 lines)
#     box1_area =
#     box2_area =
#     union_area =
#
#     # compute the IoU
#     iou =
#     ### END CODE HERE
#
#     return iou
#
# # BEGIN UNIT TEST
# ## Test case 1: boxes intersect
# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4)
#
# print("iou for intersecting boxes = " + str(iou(box1, box2)))
# assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
# assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"
#
# ## Test case 2: boxes do not intersect
# box1 = (1,2,3,4)
# box2 = (5,6,7,8)
# print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
# assert iou(box1, box2) == 0, "Intersection must be 0"
#
# ## Test case 3: boxes intersect at vertices only
# box1 = (1,1,2,2)
# box2 = (2,2,3,3)
# print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
# assert iou(box1, box2) == 0, "Intersection at vertices must be 0"
#
# ## Test case 4: boxes intersect at edge only
# box1 = (1,1,3,3)
# box2 = (2,3,3,4)
# print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
# assert iou(box1, box2) == 0, "Intersection at edges must be 0"
#
# print("\033[92m All tests passed!\033[0m\n\n")
# # END UNIT TEST

#
#
# # TODO - Exercise 3
# def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
#     """
#     Applies Non-max suppression (NMS) to set of boxes
#
#     Arguments:
#     scores -- tensor of shape (None,), output of yolo_filter_boxes()
#     boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
#     classes -- tensor of shape (None,), output of yolo_filter_boxes()
#     max_boxes -- integer, maximum number of predicted boxes you'd like
#     iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
#
#     Returns:
#     scores -- tensor of shape (None, ), predicted score for each box
#     boxes -- tensor of shape (None, 4), predicted box coordinates
#     classes -- tensor of shape (None, ), predicted class for each box
#
#     Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
#     function will transpose the shapes of scores, boxes, classes. This is made for convenience.
#     """
#     boxes = tf.cast(boxes, dtype=tf.float32)
#     scores = tf.cast(scores, dtype=tf.float32)
#
#     nms_indices = []
#     classes_labels = tf.unique(classes)[0]  # Get unique classes
#
#     for label in classes_labels:
#         filtering_mask = classes == label
#
#         #### START CODE HERE
#
#         # Get boxes for this class
#         # Use tf.boolean_mask() with 'boxes' and `filtering_mask`
#         boxes_label =
#
#         # Get scores for this class
#         # Use tf.boolean_mask() with 'scores' and `filtering_mask`
#         scores_label =
#
#         if tf.shape(scores_label)[0] > 0:  # Check if there are any boxes to process
#
#             # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
#             ##(≈ 1 line)
#             nms_indices_label =
#
#             # Get original indices of the selected boxes
#             selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)
#
#             # Append the resulting boxes into the partial result
#             # Use tf.gather() with 'selected_indices' and `nms_indices_label`
#             nms_indices.append(# Your code here )
#
#     # Flatten the list of indices and concatenate
#     # Use tf.concat() with 'nms_indices' and `axis=0`
#     nms_indices =
#
#     # Use tf.gather() to select only nms_indices from scores, boxes and classes
#     ##(≈ 3 lines)
#     scores =
#     boxes =
#     classes =
#
#     ### END CODE HERE
#
#     # Sort by scores and return the top max_boxes
#     sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
#     scores = tf.gather(scores, sort_order[0:max_boxes])
#     boxes = tf.gather(boxes, sort_order[0:max_boxes])
#     classes = tf.gather(classes, sort_order[0:max_boxes])
#
#     return scores, boxes, classes
#
# # BEGIN UNIT TEST
# # This example mimics the case shown in the slides where a car overlaps with a person
# # As both boxes are of different classes, they are never suppressed, despite the iou_threshold
# scores = np.array([0.855, 0.828])
# boxes = np.array([[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]])
# classes = np.array([0, 1])
#
# print(f"iou:    \t{iou(boxes[0], boxes[1])}")
#
# scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.9)
#
# assert np.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
# assert np.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes {boxes2.numpy()}"
# assert np.array_equal(classes2.numpy(), [0, 1]), f"Wrong value on classes {classes2.numpy()}"
#
# scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.1)
#
# assert np.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
# assert np.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes {boxes2.numpy()}"
# assert np.array_equal(classes2.numpy(), [0, 1]), f"Wrong value on classes {classes2.numpy()}"
#
# classes = np.array([0, 0])
#
# # If both boxes belongs to the same class, one box is suppressed if iou is under the iou_threshold
# scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.15)
#
# assert np.allclose(scores2.numpy(), [0.855]), f"Wrong value on scores {scores2.numpy()}"
# assert np.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6]]), f"Wrong value on boxes {boxes2.numpy()}"
# assert np.array_equal(classes2.numpy(), [0]), f"Wrong value on classes {classes2.numpy()}"
#
# # It must return both boxes, as they belong to different classes
# print(f"scores:  \t{scores2.numpy()}")
# print(f"boxes:  \t{boxes2.numpy()}")
# print(f"classes:\t{classes2.numpy()}")
#
# # If both boxes belongs to the same class, one box is suppressed if iou is under the iou_threshold
# scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, [0, 0], iou_threshold = 0.9)
#
# assert np.allclose(scores2.numpy(), [0.855, 0.828]), f"Wrong value on scores {scores2.numpy()}"
# assert np.allclose(boxes2.numpy(), [[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]]), f"Wrong value on boxes {boxes2.numpy()}"
# assert np.array_equal(classes2.numpy(), [0, 0]), f"Wrong value on classes {classes2.numpy()}"
#
# from unit_tests import test_yolo_non_max_suppression
#
# test_yolo_non_max_suppression(yolo_non_max_suppression)
# # END UNIT TEST
#
#
#
# def yolo_boxes_to_corners(box_xy, box_wh):
#     """Convert YOLO box predictions to bounding box corners."""
#     box_mins = box_xy - (box_wh / 2.)
#     box_maxes = box_xy + (box_wh / 2.)
#
#     return tf.keras.backend.concatenate([
#         box_mins[..., 1:2],  # y_min
#         box_mins[..., 0:1],  # x_min
#         box_maxes[..., 1:2],  # y_max
#         box_maxes[..., 0:1]  # x_max
#     ])
#
# # TODO - Exercise 4
# def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
#     """
#     Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
#
#     Arguments:
#     yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
#                     box_xy: tensor of shape (None, 19, 19, 5, 2)
#                     box_wh: tensor of shape (None, 19, 19, 5, 2)
#                     box_confidence: tensor of shape (None, 19, 19, 5, 1)
#                     box_class_probs: tensor of shape (None, 19, 19, 5, 80)
#     image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
#     max_boxes -- integer, maximum number of predicted boxes you'd like
#     score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
#     iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
#
#     Returns:
#     scores -- tensor of shape (None, ), predicted score for each box
#     boxes -- tensor of shape (None, 4), predicted box coordinates
#     classes -- tensor of shape (None,), predicted class for each box
#     """
#
#     ### START CODE HERE
#
#     # Retrieve outputs of the YOLO model (≈1 line)
#     box_xy, box_wh, box_confidence, box_class_probs =
#
#     # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
#     boxes =
#
#     # Use the function `yolo_filter_boxes` you've implemented to perform Score-filtering with a threshold of score_threshold
#     scores, boxes, classes = yolo_filter_boxes(,  # Use boxes
#                                                ,  # Use box confidence
#                                                ,  # Use box class probability
#                                                  # Use threshold=score_threshold
#                                                )
#
#     # Scale boxes back to original image shape.
#     boxes = scale_boxes(boxes, image_shape)
#
#     # Use the function `yolo_non_max_suppression` you've implemented to perform Non-max suppression with
#     # maximum number of boxes set to max_boxes and a threshold of iou_threshold
#     scores, boxes, classes = yolo_non_max_suppression(,  # Use scores
#                                                       ,  # Use boxes
#                                                       ,  # Use classes
#                                                       ,  # Use max boxes
#                                                        # Use iou_threshold=iou_threshold
#                                                       )
#
#     ### END CODE HERE
#
#     return scores, boxes, classes
#
#
# # BEGIN UNIT TEST
# tf.random.set_seed(10)
# yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#                 tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#                 tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
#                 tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
# scores, boxes, classes = yolo_eval(yolo_outputs)
# print("scores[2] = " + str(scores[2].numpy()))
# print("boxes[2] = " + str(boxes[2].numpy()))
# print("classes[2] = " + str(classes[2].numpy()))
# print("scores.shape = " + str(scores.numpy().shape))
# print("boxes.shape = " + str(boxes.numpy().shape))
# print("classes.shape = " + str(classes.numpy().shape))
#
# assert type(scores) == EagerTensor, "Use tensoflow functions"
# assert type(boxes) == EagerTensor, "Use tensoflow functions"
# assert type(classes) == EagerTensor, "Use tensoflow functions"
#
# assert scores.shape == (10,), "Wrong shape"
# assert boxes.shape == (10, 4), "Wrong shape"
# assert classes.shape == (10,), "Wrong shape"
#
# assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
# assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
# assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"
#
# print("\033[92m All tests passed!\033[0m\n\n")
# # END UNIT TEST
#
