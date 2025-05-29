import torch
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# Non Max Supression
def nms(predictions, iou_threshold, conf_threshold):
    # predictions is [[], [], ....] list of boxes
    # each box is [xmin, ymin, xmax, ymax, conf, class_id]
    
    # If confidence is less than certain threshold, remove the box
    predictions = [box for box in predictions if box[4] > conf_threshold]
    
    # Sort the list in descending order based on the 5th element of each item 
    predictions.sort(key=lambda x: x[4], reverse=True) 

    final_boxes = []
    
    while predictions:
        chosen = predictions.pop(0)
        final_boxes.append(chosen)
        
        predictions = [box for box in predictions
                       if chosen[5] != box[5]                           # Keep the box if the class is not same
                       or iou(chosen[:4], box[:4]) < iou_threshold]     # Keep if iou is less than threshold; iou() expects dimensions (N, 4)
        
    return final_boxes
    
    
# Intersection Over Union
def iou(boxes_preds, boxes_labels):
    # The boxes_pred and boxes_labels are of dimensions (N, 4), N is number of bounding boxes
    # box1[..., 0] gives you a tensor with shape reduced by 1 dimension (e.g., from [N, 4] -> [N]).
    # box1[..., 0:1] gives you a slice, which keeps the last dimension (e.g., from [N, 4] -> [N, 1]).
    # The ... is a flexible way to handle multi-dimensional tensors, meaning "all preceding dimensions".
    
    b1x1 = boxes_preds[..., 0:1]
    b1y1 = boxes_preds[..., 1:2]
    b1x2 = boxes_preds[..., 2:3]
    b1y2 = boxes_preds[..., 3:4]
    b2x1 = boxes_labels[..., 0:1]
    b2y1 = boxes_labels[..., 1:2]
    b2x2 = boxes_labels[..., 2:3]
    b2y2 = boxes_labels[..., 3:4]
    
    inter_x1 = torch.max(b1x1, b2x1)
    inter_y1 = torch.max(b1y1, b2y1)
    inter_x2 = torch.min(b1x2, b2x2)
    inter_y2 = torch.min(b1y2, b2y2)
    
    # Add clamp(0) in case they dont intersect at all
    intersection = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Use abs to get rid of negative area values
    # Add 1e-6 as epsilon smoothing
    union = abs((b1x2 - b1x1) * (b1y2 - b1y1)) + abs((b2x2 - b2x1) * (b2y2 - b2y1)) - intersection + 1e-6
    
    return intersection / union
    
def mAP():
    pass

# Encode the Target to YOLO-Style Output
def encode_target(image, target, S, B, C):
    pass

def decode_prediction(model_output, S, B, C):
    pass

# Corner Coordinates to Center Coordinates
def corner_to_center(box):
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) /2
    width = xmax - xmin
    height = ymax - ymin

    return ([x_center, y_center, width, height])

# Visualize Bounding Boxes
def visualize_bounding_boxes(image: torch.tensor, target, class_names):
    image = image.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
    boxes = target["boxes"]
    labels = target["labels"]
    
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(image)
    
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.tolist()
        bbox = patches.Rectangle(xy=(xmin, ymin), width=(xmax-xmin), height=(ymax-ymin), linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(bbox)
        
        label = class_names[labels[i].item()]
        ax.text(xmin, ymin-4, label, color="white", fontsize=10, backgroundcolor="red")
    
    plt.axis('off')
    plt.show()

# matplotlib.patches.Rectangle uses Top-left corner + Width and Height
# Matplotlib uses a Cartesian coordinate system, so the origin is at the bottom-left by default
# But the origin (0, 0) is at the top-left of the image in image coordinate systems 
