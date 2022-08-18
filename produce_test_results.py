from codecs import ignore_errors
from modelv2.Segformer_model import SegformerForSemanticSegmentation
from transformers import SegformerFeatureExtractor
from CityscapesDataset import CityscapesDataset
from ApolloScapeDataset import ApolloScapeDataset
import os
import argparse
import torch
import tqdm
import numpy as np

from PIL import Image

# assign gpu devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def produce_test_results(model: torch.nn.Module, model_name:str, image_file_names):
    
    # Create the directory in which to store the test images
    dir_path = os.path.join("/home/a.lombardi/CityScapes_Dataset/leftImg8bit/tested", os.path.basename(os.path.normpath(model_name)))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


    print("-> Testing started:")
    with torch.no_grad():
        model.eval()

        for f in tqdm.tqdm(image_file_names):

            image = Image.open(f)
            image = image.convert("RGB")
            
            # prepare the image for the model (aligned resize)
            feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

            pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)
            
            outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
            logits = outputs.logits.cpu()
            
            # First, rescale logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(logits,
                    size=[1024, 2048], # (height, width)
                    mode='bilinear',
                    align_corners=False)
            
            # Second, apply argmax on the class dimension and convert to numpy
            predicted = upsampled_logits.argmax(dim=1)
            predicted = np.squeeze(predicted.numpy().astype(np.uint8))
            
            # Save the image with its original filename
            output_file_path = os.path.join(dir_path, os.path.basename(f))
            #print(output_file_path)
            
            im_pred = Image.fromarray(predicted).convert("L")
            im_pred.save(output_file_path)
            
                
if __name__ == "__main__":
    torch.cuda.empty_cache()

    ###############################################
    ####### Getting configuration settings ########
    parser = argparse.ArgumentParser()

    parser.add_argument("-pw", "--pretrained_weights", type=str, 
                    help="path or str of the pretrained model weights")
    parser.add_argument("-fe", "--feature_extractor", type=str, default=0,
                    choices=["0","1","2","3","4","5"],
                    help="type of nvidia/mit-bX pretrained weights of the feature extractor")
    args = parser.parse_args()

    MODEL = args.pretrained_weights
    ###############################################

    ###############################################
    ############ Preparing the dataset ############
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b"+args.feature_extractor)
    test_set = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', feature_extractor=feature_extractor, split='test', transforms=False)

    ###############################################

    
    image_file_names = []
    test_dir = "/home/a.lombardi/CityScapes_Dataset/leftImg8bit/test"
    for root, dirs, files in os.walk(test_dir):
            for f in files:
                complete_path = os.path.join(root, f)
                #print(complete_path)
                image_file_names.append(complete_path)
        
    
    image_file_names = sorted(image_file_names) 

    ###############################################
    ############## Preparing the model ############
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, # Encoder pretrained weights
                                                        ignore_mismatched_sizes=True,
                                                        num_labels=len(test_set.get_id2label()), 
                                                        id2label=test_set.get_id2label(), 
                                                        label2id=test_set.get_label2id(),
                                                        reshape_last_stage=True)

    if torch.cuda.is_available():
        print("Loading the model on GPU: ", torch.cuda.get_device_name(0))
        model = model.cuda()
    else:
        print("Using the model on CPU\n")
    ###############################################

    print(f'Testing the following model:\n{MODEL}\n\n')
    produce_test_results(model, MODEL, image_file_names)