import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import cv2
from torchvision import datasets,transforms
from glob import glob
import os
from PIL import Image
from matplotlib import patches
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import albumentations as al
from albumentations.pytorch.transforms import ToTensorV2



class Wheatdatasets(Dataset):
    
    def __init__(self,dataframe,image_dir,transforms = None):
        
        super().__init__()
    
        self.image = [image]
        self.img_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self,index):
       
        image = cv2.cvtColor(np.asarray(self.image[index]),cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image/255.0
       
        if self.transforms:
            sample ={
                'image':image
            }
            sample =self.transforms(**sample)
            image = sample['image']
         
        return np.asarray(image)
    
    def __len__(self) -> int:
        return len(self.image)
    
# Albumentations
def get_test_transform():
    return al.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__=="__main__":
    st.header("did it worked ?, did it worked ?")
    st.subheader("checking subheader")
    uploaded_file = st.file_uploader("CHoose an image___",type="jpg")
    button = st.button("Conform")
    weights_file="fasterrcnn.pth"
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False,pretrained_backbone=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = 2 #wheats and background

    in_feature = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature,num_class) #changin the pretrained head with a new one

    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    
    
    detection_threshold = 0.5
    results = []
    outputs = None
    images = None
    
    if button and uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        test_dataset = Wheatdatasets(image, get_test_transform())
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=collate_fn
        )

        for images in test_data_loader:

            
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()

                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                scores = scores[scores >= detection_threshold]

                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                for j in zip(boxes, scores):
                    result = {
                        'Detected Boxes': "{} {} {} {}".format(j[0][0], j[0][1], j[0][2], j[0][3]),
                        'Confidence%': j[1]
                    }
                    results.append(result)

    if len(results) != 0:
        # print out results
        sample = images[0].permute(1, 2, 0).cpu().numpy()
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        fig, ax = plt.subplots(1, 1, figsize=(32, 16))
        for box in boxes:
            x1, y1, x2, y2 = box
            sample = cv2.rectangle(img=sample,
                                   pt1=(x1, y1),
                                   pt2=(x2, y2),
                                   color=(0, 0, 255), thickness=3)
        ax.set_axis_off()
        st.image(cv2.UMat.get(sample), clamp=True)
        st.write("# Results")
        st.dataframe(pd.DataFrame(results))
    else:
        st.write("")
        st.write("""
        No wheat heads detected in the image!
        """)

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
