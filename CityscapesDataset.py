import random
from torch import IntTensor
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF

from torchvision import transforms as tfs

from collections import namedtuple

class CityscapesDataset(Dataset):
    def __init__(self, path: str, feature_extractor, split: str, transforms: bool=False, mode='fine', target_type='semantic'):
        """
        Args:
            root_dir (string): Root directory of the dataset.
            split: Whether to load "training", "validation" or "test" set.
            mode: 'fine' or 'coarse"
            target_type: for the label type, that can be 'instance', 'semantic' or 'panoptic'
        """
        self.split = split
        self.dataset = Cityscapes(path, split=split, mode=mode, target_type=target_type)
        self.feature_extractor = feature_extractor
        self.transforms = transforms


        Label = namedtuple( 'Label' , [
            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
            'id'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.
            'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
            'category'    , # The name of the category that this label belongs to
            'categoryId'  , # The ID of this category. Used to create ground truth images on category level.
            'hasInstances', # Whether this label distinguishes between single instances or not
            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored during evaluations or not
            'color'       , # The color of this label
            ] )

        #--------------------------------------------------------------------------------
        # A list of all labels
        #--------------------------------------------------------------------------------

        self.labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]


    #--------------------------------------------------------------------------------
    # Create dictionaries for a fast lookup
    #--------------------------------------------------------------------------------
    def get_id2label(self):
        # return id to label object
        id2label        = { label.id      : label for label in self.labels           }
        return  id2label

    def get_label2id(self):
        # return name to label object
        name2label      = { label.name    : label for label in self.labels           }
        return name2label

    def get_label2color(self):
        # return label to color code dictionary
        label2color = {label.color : label for label in self.labels}
        return label2color
    #--------------------------------------------------------------------------------

    def getNumClasses(self):
        return len(self.dataset.classes)

    def __len__(self):
        return len(self.dataset)

    def __transform__(self, image, mask):
        
        # Resize
        #resize = tfs.Resize(size=(1024, 512))
        #image = resize(image)
        #segmentation_map = resize(segmentation_map)

        # Random crop
        i, j, h, w = tfs.RandomCrop.get_params(
            image, output_size=(1024,1024))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Transform to tensor
        #image = TF.to_tensor(image)
        #mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, idx):
        
        image, segmentation_map = self.dataset[idx]

        if self.transforms:
            
            #image = self.cvt_to_tensor(image).numpy()
            #segmentation_map = self.cvt_to_tensor(segmentation_map).numpy()
            #augmented = self.transforms(image=image, mask=segmentation_map)

            image, segmentation_map = self.__transform__(image, segmentation_map)

            encoded_inputs = self.feature_extractor(images=image, segmentation_maps=segmentation_map, return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
'''
ds = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', split='test', transforms=False)

prova = ds[60]

print(prova["pixel_values"].shape)
print(prova["labels"].shape)

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(prova["labels"].numpy())
plt.savefig("prova.png")

p = prova["pixel_values"].numpy()
p = np.swapaxes(p, 0, 2)
p = np.swapaxes(p, 0, 1)
plt.imshow(p)
plt.savefig("prova2.png")
'''