import random
from torch import IntTensor
from transformers import SegformerFeatureExtractor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
import os
import cv2
import torchvision.transforms.functional as TF

from torchvision import transforms as tfs

class KITTIDataset(Dataset):
    """KITTI semantic segmentation dataset."""

    def __init__(self, root_dir:str, transforms=None, split:str = 'training'):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
        self.transforms = transforms

        self.img_dir = os.path.join(self.root_dir, split, "image_2")
        self.ann_dir = os.path.join(self.root_dir, split, "semantic_rgb")
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = image_file_names
        #self.images = sorted(image_file_names) #They are already sorted
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = annotation_file_names
        # self.annotations = sorted(annotation_file_names) #They are already sorted

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        #print(self.img_dir)
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
    
        #image = Image.open()
        #segmentation_map = Image.open()

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

ds = KITTIDataset("/home/a.lombardi/KITTI_Dataset", transforms=None, split='training')

prova = ds[0]

print(prova["pixel_values"].shape)
print(prova["labels"].shape)

import matplotlib.pyplot as plt
plt.imshow(prova["labels"].numpy())
plt.savefig("prova.png")