from codecs import ignore_errors
from modelv2.Segformer_model import SegformerForSemanticSegmentation
from transformers import SegformerFeatureExtractor
from CityscapesDataset import CityscapesDataset
from ApolloScapeDataset import ApolloScapeDataset
from torch.utils.data import DataLoader
from configparser import ConfigParser
import torch
from torchmetrics import JaccardIndex
import tqdm
import numpy as np
import argparse

from datasets import load_metric
# assign gpu devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model: torch.nn.Module, test_loader, num_labels):

    jaccards = []
    jaccard = JaccardIndex(num_classes=num_labels, average="weighted", ignore_index=255)
    metric = load_metric("mean_iou")

    print("-> Testing started:")
    with torch.no_grad():
        model.eval()

        for batch in tqdm.tqdm(test_loader):
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            if torch.cuda.is_available():
                pixel_values, labels = pixel_values.cuda(), labels.cuda()

            # evaluate
            outputs = model(pixel_values=pixel_values)
            # First, rescale logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            # Second, apply argmax on the class dimension
            predicted = upsampled_logits.argmax(dim=1)

            #mask = (labels != 255) # we don't include the background class in the accuracy calculation
            pred_labels = predicted.detach().cpu()
            true_labels = labels.detach().cpu()
            
            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=pred_labels.numpy(), references=true_labels.numpy())
            
            jaccards.append(jaccard(pred_labels, true_labels))
            
        meanIoU = np.mean(jaccards)
        metrics = metric.compute(num_labels=num_labels, ignore_index=255,
                                reduce_labels=False)# we've already reduced the labels before

        print(metrics)
        print("\n\nMean_iou: ", metrics["mean_iou"])
        print("Mean accuracy: ", metrics["mean_accuracy"])
        print("Jaccard index (mIoU): ", meanIoU)

                
if __name__ == "__main__":
    torch.cuda.empty_cache()

    ###############################################
    ####### Getting configuration settings ########
    config = ConfigParser()
    parser = argparse.ArgumentParser()
    
    config.read('/home/a.lombardi/my_segformer/configuration.ini')
    BATCH_SIZE = config.getint('TRAINING', 'batch_size')

    parser.add_argument('-pw', '--pretrained_weights', type=str, 
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

    test_set = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', feature_extractor=feature_extractor, split='val', transforms=False)
    #test_set = ApolloScapeDataset("/home/a.lombardi/ApolloScape_Dataset", split='test', transforms=None)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ###############################################

    num_labels = len(test_set.get_label2id())

    ###############################################
    ############## Preparing the model ############
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, # Encoder pretrained weights
                                                        ignore_mismatched_sizes=True,
                                                        num_labels=num_labels, 
                                                        id2label=test_set.get_id2label(), 
                                                        label2id=test_set.get_label2id(),
                                                        reshape_last_stage=True)

    if torch.cuda.is_available():
        print("Loading the model on GPU: ", torch.cuda.get_device_name(0))
        model = model.cuda()
    else:
        print("Using the model on CPU\n")
    ###############################################

    print(f'\nTesting the following model:\n{MODEL}\n')
    test(model,
        test_loader=test_loader, 
        num_labels=num_labels)