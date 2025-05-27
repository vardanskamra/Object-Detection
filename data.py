import os
import torch
import xml.etree.ElementTree as ET 
import torch.utils
import torchvision.transforms as T 
import torchvision.transforms.functional as F 

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train'):
        self.root = root_dir # Path to VOC2012
        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.ann_dir = os.path.join(self.root, "Annotations")
        image_set_file = os.path.join(self.root, "ImageSets", "Main", f"{image_set}.txt")
        
        with open (image_set_file, 'r') as f: 
            self.image_ids = f.readlines()
            self.image_ids = [i.strip() for i in self.image_ids]
        
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']
        
        # MAP CLASS TO A NUMBER
        self.class_dict = {name: idx for name, idx in enumerate(self.class_names)}
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")
        
        # OPEN IMAGE
        image = Image.open(image_path).convert('RGB')
        
        # PARSE XML TREE
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
         
        for obj in root.findall('object'):
            label = obj.find('name').text.lower().strip()
    
            if label not in self.class_names:
                continue
            
            labels.append(self.class_dict[label])
            
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            boxes.append([xmin, ymin, xmax, ymax])
            
        # CONVERT TO TORCH TENSORS    
        boxes = torch.tensor(boxes, dtype=torch.float32) # Shape: [num_objects, 4]
        labels = torch.tensor(labels, dtype=torch.int64) # Shape: [num_objects]
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index])
        }
        
        image = self.transform(image)
            
        return image, target
    
# Now we need a collate function because DataLoader does this (for batch_size=2):
#   batch = [
#       dataset[0],   # (image0, target0)
#       dataset[1]    # (image1, target1)
#   ]

# Because collate_fn = default_collate does this:
# 
#   images = torch.stack([image0, image1])  # works
#   targets = torch.stack([target0, target1]) # breaks

# That breaks because:
#   target0["boxes"] might be shape [2, 4]
#   target1["boxes"] might be [5, 4]
# PyTorch can't stack tensors of different shapes!

# Default DataLoader uses default_collate() -> tries torch.stack() -> fails when shapes vary (like boxes).
# Custom collate_fn avoids stacking -> just returns tuples -> works with detection tasks.
# So now we get two tuples, one of images and one of targets
#   zip((img0, target0), (img1, target1)) => (img0, img1), (target0, target1)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(batch_size=32, train_size=0.8, shuffle=True):
    
    assert train_size < 1 and train_size > 0, "train_size must be between 0 and 1"
    dataset = VOCDataset(r"C:\Users\VARDAN S KAMRA\Desktop\Projects\Object Detection\VOC2012")
    
    train_size = int(train_size * len(dataset))
    test_size = int(len(dataset) - train_size)
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    
    return train_dataloader, test_dataloader

