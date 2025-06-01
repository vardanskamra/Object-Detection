import torch
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# Non Max Supression; only used in inference, not in training
def nms(predictions, iou_threshold):
    # predictions is [[], [], ....] list of boxes
    # each box is [xmin, ymin, xmax, ymax, conf, class_id]
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
def target_to_yolo(target, S=7, B=2, C=20, image_size=(224, 224)): #(image_size is w, h)
    # YOLO output for one grid cell looks like:
    # [C0.......C19, Conf_B1, X_B1, Y_B1, W_B1, H_B1, Conf_B2, X_B2, Y_B2, W_B2, H_B2]
    
    # Only one of the two bounding box predictors (B=2) in each grid cell is filled during target preparation.
    # This is because at train-time, only 1 predictor should be responsible for an object.
    # We fill only the first box slot (index range C:C+5), and leave the second empty.
    
    # The responsibility assignment is handled during training by the loss function:
    # During loss computation (in YOLOv1 loss), for each grid cell with an object,
    # the box predictor whose prediction has the highest IoU with the ground truth box
    # is assigned responsibility for that object.
    
    # The loss function compares both predicted boxes against the ground truth,
    # and assigns the one with the best match to be trained for that object.
    # The other predictor (box 2) is either ignored or penalized based on objectness loss (depending on its confidence).
    
    # This dynamic assignment ensures that both box predictors can learn over time,
    # even though only one is used during labeling.
    
    yolo_target = torch.zeros((S, S, C+B*5))
    boxes = target["boxes"]
    labels = target["labels"]
    img_w, img_h = image_size
    
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        class_idx = labels[i]
        
        # Convert to centre format
        x, y, w, h = corner_to_center(xmin, ymin, xmax, ymax)
        
        # Normalize between 0 and 1; remember that dataset sclaes it to 224, but here we need normalization
        x /= img_w
        y /= img_h
        w /= img_w
        h /= img_h
        
        # Determine the grid cell containing the centre points
        # if x=0.4 -> grid_x = int(0.4 * 7) = 2 (starts from 0, goes till S-1)
        grid_x = int(x * S)
        grid_y = int(y * S)
        
        # If the normalized x_center or y_center is exactly 1.0 (i.e. on the far-right or bottom edge) then:
        # grid_x = int(1.0 * S) = S
        # This is out of bounds because valid indices are from 0 to S-1.
        grid_x = min(grid_x, S-1)
        grid_y = min(grid_y, S-1)
        
        # x and y are normalized coordinates of the object's center w.r.t. entire image
        # If x = 0.65 and S = 7, then 0.65 * 7 = 4.55, which means the object center lies in grid cell 4 (indexing from 0), and is 0.55 units into that cell
        # Now we need position of the object's center inside the grid cell, expressed as a value between 0 and 1
        cell_x = x * S - grid_x
        cell_y = y * S - grid_y
        
        # Skip if the grid cell is taken already (only one object per cell in YOLOv1)
        if yolo_target[grid_y, grid_x, C] == 1:
            continue
        
        yolo_target[grid_y, grid_x, class_idx] = 1                                      # one-hot class
        yolo_target[grid_y, grid_x, C:C+5] = torch.tensor([cell_x, cell_y, w, h, 1.0])  # box + objectness
        # In YOLOv1, each predicted bounding box has a confidence score, which is defined as:
        # Confidence Pr(Obj) X IoU(pred, truth) 
        # Objectness = 1 means yes, there is an object here
        
        # We use yolo_target[grid_y, grid_x, ...] because tensors follow [row, column] = [height, width] convention
        # grid_y is the row (vertical axis -> height), grid_x is the column (horizontal axis -> width)
        # This matches how image tensors are stored in PyTorch: (height, width, channels)
        
    return yolo_target

# Encode the Linear Output to YOLO-Style Output
def linear_to_yolo(model_output, S, B, C):
    """ 
    Takes a model output prediction that is (N, (S*S*(C+B*5))) and converts it to YOLO output (N, S, S, C+B*5)
    """
    # .view() is a reshape operation and does not detach the computation graph
    # -1 means PyTorch infers the batch size automatically
    return model_output.view(-1, S, S, C+B*5)

# Make Boxes from YOLO-Style Output; only used for inference, not for training
def yolo_to_boxes(yolo_pred, S=7, B=2, C=20, confidence_threshold=0.5):
    boxes = []
    for i in range(S):
        for j in range(S):
            cell = yolo_pred[i, j]
            class_prob = cell[:C]
            class_id = torch.argmax(class_prob).item()
            class_score = class_prob[class_id].item()
            
            for b in range(B):
                idx = C + 5 * B
                x, y, w, h, obj = b[idx : idx+5]
                conf = (obj * class_score).item() # confidence = objectness * class probability
                
                if conf < confidence_threshold:
                    continue
                
                xmin, ymin, xmax, ymax = center_to_corner(x, y, w, h)
                boxes.append([xmin, ymin, xmax, ymax, conf, class_id])
                
    return boxes

# Corner Coordinates to Center Coordinates
def corner_to_center(xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    return (x_center, y_center, width, height)

# Center Coordinates to Corner Coordinates
def center_to_corner(x_center, y_center, width, height):
    xmin = x_center - width/2
    ymin = y_center - height/2
    xmax = x_center + width/2
    ymax = y_center + height/2
    
    return (xmin, ymin, xmax, ymax)

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
