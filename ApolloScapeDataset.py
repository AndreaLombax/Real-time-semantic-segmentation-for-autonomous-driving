import random

from collections import namedtuple

from transformers import SegformerFeatureExtractor
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms as tfs
import numpy as np

class ApolloScapeDataset(Dataset):
    """KITTI semantic segmentation dataset."""

    def __init__(self, root_dir:str, split:str='train', transforms:bool=False):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            split: the split of the dataset (train, test or val)
        """
        assert split=='train' or split=='test' or split=='val', "The split of the dataset must be one between 'train', 'test' or 'val'"

        self.root_dir = root_dir
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
        self.transforms = transforms

        self.img_dir = os.path.join(self.root_dir, "ColorImage", split)
        self.ann_dir = os.path.join(self.root_dir, "Label", split)
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            for f in files:
                complete_path = os.path.join(root, f)
                #print(complete_path)
                image_file_names.append(complete_path)
        
        self.images = sorted(image_file_names) 
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            for f in files:
                complete_path = os.path.join(root, f)
                annotation_file_names.append(complete_path)
        
        self.annotations = sorted(annotation_file_names) 
        
        
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

        # a label and all meta information
        Label = namedtuple('Label', [
            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
            'clsId'       ,
            'id'          , # An integer ID that is associated with this label.
            'trainId'     , 
            'category'    , # The name of the category that this label belongs to
            'categoryId'  , # The ID of this category. Used to create ground truth images on category level.
            'hasInstances', # Whether this label distinguishes between single instances or not
            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored during evaluations or not
            'color'       , # The color of this label
            ])
        #--------------------------------------------------------------------------------
        # A list of all labels
        #--------------------------------------------------------------------------------

        self.labels = [
            #     name                    clsId    id   trainId   category  catId  hasInstanceignoreInEval   color
            Label('others'              ,    0 ,    0,   0  , 'others'        ,   0  ,False , True  ,    (0, 0, 0)       ),
            Label('rover'               , 0x01 ,    1,   1  , 'others'        ,   0  ,False , True  ,    (0, 0, 0)       ),
            Label('sky'                 , 0x11 ,   17,   2  , 'sky'           ,   1  ,False , False ,    (70, 130, 180)  ),
            Label('car'                 , 0x21 ,   33,   3  , 'movable object',   2  ,True  , False ,    (0, 0, 142)     ),
            Label('car_groups'          , 0xA1 ,  161,   4  , 'movable object',   2  ,True  , False ,    (0, 0, 142)     ),  
            Label('motorbicycle'        , 0x22 ,   34,   5  , 'movable object',   2  ,True  , False ,    (0, 0, 230)     ),
            Label('motorbicycle_group'  , 0xA2 ,  162,   6  , 'movable object',   2  ,True  , False ,    (0, 0, 230)     ),
            Label('bicycle'             , 0x23 ,   35,   7  , 'movable object',   2  ,True  , False ,    (119, 11, 32)   ),
            Label('bicycle_group'       , 0xA3 ,  163,   8  , 'movable object',   2  ,True  , False ,    (119, 11, 32)   ),
            Label('person'              , 0x24 ,   36,   9  , 'movable object',   2  ,True  , False ,    (0, 128, 192)   ),
            Label('person_group'        , 0xA4 ,  164,  10  , 'movable object',   2  ,True  , False ,    (0, 128, 192)   ),
            Label('rider'               , 0x25 ,   37,  11  , 'movable object',   2  ,True  , False ,    (128, 64, 128)  ),
            Label('rider_group'         , 0xA5 ,  165,  12  , 'movable object',   2  ,True  , False ,    (128, 64, 128)  ),
            Label('truck'               , 0x26 ,   38,  13  , 'movable object',   2  ,True  , False ,    (128, 0, 192)   ),
            Label('truck_group'         , 0xA6 ,  166,  14  , 'movable object',   2  ,True  , False ,    (128, 0, 192)   ), 
            Label('bus'                 , 0x27 ,   39,  15  , 'movable object',   2  ,True  , False ,    (192, 0, 64)    ),
            Label('bus_group'           , 0xA7 ,  167,  16  , 'movable object',   2  ,True  , False ,    (192, 0, 64)    ),
            Label('tricycle'            , 0x28 ,   40,  17  , 'movable object',   2  ,True  , False ,    (128, 128, 192) ),
            Label('tricycle_group'      , 0xA8 ,  168,  18  , 'movable object',   2  ,True  , False ,    (128, 128, 192) ),
            Label('road'                , 0x31 ,   49,  19  , 'flat'          ,   3  ,False , False ,    (192, 128, 192) ),
            Label('siderwalk'           , 0x32 ,   50,  20  , 'flat'          ,   3  ,False , False ,    (192, 128, 64)  ),
            Label('traffic_cone'        , 0x41 ,   65,  21  , 'road obstacles',   4  ,False , False ,    (0, 0, 64)      ),
            Label('road_pile'           , 0x42 ,   66,  22  , 'road obstacles',   4  ,False , False ,    (0, 0, 192)     ),
            Label('fence'               , 0x43 ,   67,  23  , 'road obstacles',   4  ,False , False ,    (64, 64, 128)   ),
            Label('traffic_light'       , 0x51 ,   81,  24  , 'Roadside objects',   5  ,False , False ,  (192, 64, 128)  ),
            Label('pole'                , 0x52 ,   82,  25  , 'Roadside objects',   5  ,False , False ,  (192, 128, 128) ),
            Label('traffic_sign'        , 0x53 ,   83,  26  , 'Roadside objects',   5  ,False , False ,  (0, 64, 64)     ),
            Label('wall'                , 0x54 ,   84,  27  , 'Roadside objects',   5  ,False , False ,  (192, 192, 128) ),
            Label('dustbin'             , 0x55 ,   85,  28  , 'Roadside objects',   5  ,False , False ,  (64, 0, 192)    ),
            Label('billboard'           , 0x56 ,   86,  29  , 'Roadside objects',   5  ,False , False ,  (192, 0, 192)   ),
            Label('building'            , 0x61 ,   97,  30  , 'building'        ,   6  ,False , False ,  (192, 0, 128)   ),
            Label('bridge'              , 0x62 ,   98,  31  , 'building'        ,   6  ,False , True  ,  (128, 128, 0)   ),
            Label('tunnel'              , 0x63 ,   99,  32  , 'building'        ,   6  ,False , True  ,  (128, 0, 0)     ),
            Label('overpass'            , 0x64 ,  100,  33  , 'building'        ,   6  ,False , True  ,  (64, 128, 64)   ),
            Label('vegatation'          , 0x71 ,  113,  34  , 'natural'         ,   7  ,False , False ,  (128, 128, 64)  ),
            Label('unlabeled'           , 0xFF ,  -1 ,  -1  , 'unlabeled'       ,   8  ,False , True  ,  (255, 255, 255) ),
        ]

    #--------------------------------------------------------------------------------
    # Create dictionaries for a fast lookup
    #--------------------------------------------------------------------------------
    def get_id2label(self):
        # return id to label object
        id2label        = { label.id      : label for label in self.labels }
        return  id2label

    def get_label2id(self):
        # return name to label object
        name2label      = { label.name    : label for label in self.labels }
        return name2label

    def get_trainId2label(self):
        # trainId to label object. This is used as a id2label.
        trainId2label   = {label.trainId: label for label in self.labels}
        return trainId2label

    def get_label2color(self):
        # return label to color code dictionary
        label2color = {label.color : label for label in self.labels}
        return label2color
    #--------------------------------------------------------------------------------

    def __len__(self):
        return len(self.images)

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

        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        
        # Use train ids for a possible cross validation
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx])).convert("L")
        for l in self.labels:
            segmentation_map = np.where(segmentation_map!=l.id, segmentation_map, l.trainId).astype(np.uint8)

        if self.transforms:
            # Return to PIL Image to apply transformations
            segmentation_map = Image.fromarray(segmentation_map).convert("L")
        
            image, segmentation_map = self.__transform__(image, segmentation_map)
            encoded_inputs = self.feature_extractor(images=image, segmentation_maps=segmentation_map, return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
'''
ds = ApolloScapeDataset("/home/a.lombardi/ApolloScape_Dataset", split='test', transforms=True)

prova = ds[55]

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