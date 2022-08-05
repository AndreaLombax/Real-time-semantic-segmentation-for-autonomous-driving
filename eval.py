from transformers import SegformerFeatureExtractor
from modelv2.Segformer_model import SegformerForSemanticSegmentation
from CityscapesDataset import CityscapesDataset
from ApolloScapeDataset import ApolloScapeDataset
from torch.utils.data import DataLoader
from configparser import ConfigParser
import torch
from utils import bcolors
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# assign gpu devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluateOnImage(model: torch.nn.Module, image_path:str, label2color:dict):

    image = Image.open(image_path)
    image = image.convert("RGB")
    # prepare the image for the model (aligned resize)
    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

    pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)
    
    model.eval()
    outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()
    # First, rescale logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(logits,
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3\

    palette = label2color

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(img)
    axs[1].imshow(color_seg)
    plt.savefig("prova.png")

if __name__ == "__main__":
    torch.cuda.empty_cache()

    ###############################################
    ####### Getting configuration settings ########
    config = ConfigParser()
    config.read('/home/a.lombardi/my_segformer/configuration.ini')
    BATCH_SIZE = config.getint('TRAINING', 'batch_size')
    PRETRAINED_WEIGHTS = config.get('MODEL', 'model_to_test')
    ###############################################

    ###############################################
    ############## Preparing the model ############
    model = SegformerForSemanticSegmentation.from_pretrained(PRETRAINED_WEIGHTS, # Encoder pretrained weights
                                                        ignore_mismatched_sizes=True,
                                                         #num_labels=len(test_set.labels), 
                                                         #id2label=test_set.get_id2label(), 
                                                         #label2id=test_set.get_label2id(),
                                                         reshape_last_stage=True)

    if torch.cuda.is_available():
        print("Loading the model on GPU: ", torch.cuda.get_device_name(0))
        model = model.cuda()
    else:
        print("Using the model on CPU\n")
    ###############################################

    image_path = "/home/a.lombardi/CityScapes_Dataset/leftImg8bit/val/munster/munster_000069_000019_leftImg8bit.png"
    label2color = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', split='test').get_label2color()
    #label2color = ApolloScapeDataset("/home/a.lombardi/ApolloScape_Dataset", split='test', transforms=None).get_label2color()
    evaluateOnImage(model=model, image_path=image_path, label2color=label2color)