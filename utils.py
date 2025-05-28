import torch
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# Non Max Supression
def nms():
    pass

# Intersection Over Union
def iou():
    pass

def mAP():
    pass

# Encode the Target to YOLO-Style Output
def encode_target(image, target, s, b, c):
    pass

# Corner Coordinates to Center Coordinates
def corner_to_center(xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) /2
    width = xmax - xmin
    height = ymax - ymin

    return ([x_center, y_center, width, height])

# Visualize Bounding Boxes
def visualize_bounding_boxes(image: torch.tensor, target, class_dict):
    image = image.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
    boxes = target["boxes"]
    labels = target["labels"]
    
    fig, ax = plt.subplots(1, figsize=(8,8))

# matplotlib.patches.Rectangle uses Top-left corner + Width and Height
# Matplotlib uses a Cartesian coordinate system, so the origin is at the bottom-left by default
# But the origin (0, 0) is at the top-left of the image in image coordinate systems 
