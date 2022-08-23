from transformers import SegformerFeatureExtractor
from modelv2.Segformer_model import SegformerForSemanticSegmentation
from CityscapesDataset import CityscapesDataset
from ApolloScapeDataset import ApolloScapeDataset
import torch
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# assign gpu devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluateOnImage(model: torch.nn.Module, image_path:str, label2color:dict):


    custom_lines = [Line2D([0], [0], color=(key[0]/255,key[1]/255,key[2]/255), lw=4) for key,_ in label2color.items()]
    #print(custom_lines)

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

    for color, L in label2color.items():
        label_id = L._asdict()["id"]
        color_seg[seg == label_id, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(img)
    axs[1].imshow(color_seg)
    axs[1].legend(custom_lines, [item._asdict()["name"] for _,item in label2color.items()], loc='upper right', bbox_to_anchor=(1.25, 1.2))

    plt.savefig("prova2.png")

if __name__ == "__main__":
    torch.cuda.empty_cache()

    ###############################################
    ####### Getting configuration settings ########
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-pw', '--pretrained_weights', type=str, 
                    help="path or str of the pretrained model weights")
    parser.add_argument('-i', '--image_path', type=str, 
                    help="path of image")
    parser.add_argument("-fe", "--feature_extractor", type=str, default=0,
                    choices=["0","1","2","3","4","5"],
                    help="type of nvidia/mit-bX pretrained weights of the feature extractor")
    args = parser.parse_args()
    ###############################################

    ###############################################
    ############## Preparing the model ############
    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_weights, # Encoder pretrained weights
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
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b"+args.feature_extractor)
    label2color = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', feature_extractor=feature_extractor,split='test').get_label2color()

    #label2color = ApolloScapeDataset("/home/a.lombardi/ApolloScape_Dataset", split='test', transforms=None).get_label2color()
    evaluateOnImage(model=model, image_path=args.image_path, label2color=label2color)