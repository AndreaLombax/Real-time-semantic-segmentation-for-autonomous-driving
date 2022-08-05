#from transformers import SegformerForSemanticSegmentation
from modelv2.Segformer_model import SegformerForSemanticSegmentation
from CityscapesDataset import CityscapesDataset
from ApolloScapeDataset import ApolloScapeDataset
from torch.utils.data import DataLoader
from configparser import ConfigParser
from math import ceil, inf
import torch
from sklearn.metrics import accuracy_score
from utils import bcolors
from torchvision import transforms as tfs
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.flush()
# assign gpu devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(model: torch.nn.Module, model_type:str, train_loader, val_loader, criterion, optimizer, epochs=1, print_step=1):
    """ Training function

    Args:
        model (torch.nn.Module): model to train
        model_type (str): name of the model type
        device (_type_): GPU or CPU device
        train_loader (_type_): train data laoder
        val_loader (_type_): validation data loader
        criterion (_type_): already initialized loss function
        optimizer (_type_): 
        num_classes (_type_): number of the classes to predict
        epochs (int, optional): Training epochs. Defaults to 1.
        print_step (int, optional): Defaults to 1.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5, patience=5, 
                                                             min_lr=0.0000001, verbose=True)

    if model_type is None:
        model_type = "model"

    # utils data
    min_valid_loss = inf
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []

    print("-> Training started:")
    # cycle through epochs
    for e in range(epochs):
        
        model.train() # turn on train mode

        ###############################################
        ############### TRAINING STEP #################
        for batch in train_loader:
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            if torch.cuda.is_available():
                pixel_values, labels = pixel_values.cuda(), labels.cuda()

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)

            # First, rescale logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            # Second, apply argmax on the class dimension
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels != 255) # we don't include the background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            loss = outputs.loss
            accuracies.append(accuracy)
            losses.append(loss.item())

            writer.add_scalar("Train loss", loss.item(), e)

            # clear the gradients
            optimizer.zero_grad()
            # calculate gradients
            loss.backward()
            # update weights
            optimizer.step()
        ###############################################

        ###############################################
        ################## VAL STEP ###################
        with torch.no_grad():   # disable gradient calculation
            model.eval()

            for batch in val_loader:
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]

                if torch.cuda.is_available():
                    pixel_values, labels = pixel_values.cuda(), labels.cuda()
                
                # evaluate
                outputs = model(pixel_values=pixel_values, labels=labels)
                # First, rescale logits to original image size
                upsampled_logits = torch.nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                mask = (labels != 255) # we don't include the background class in the accuracy calculation
                pred_labels = predicted[mask].detach().cpu().numpy()
                true_labels = labels[mask].detach().cpu().numpy()
                accuracy = accuracy_score(pred_labels, true_labels)
                val_loss = outputs.loss
                val_accuracies.append(accuracy)
                val_losses.append(val_loss.item())

                writer.add_scalar("Val loss", val_loss.item(), e)
        ###############################################

        scheduler.step(sum(val_losses)/len(val_losses))

        # Print the loss each print_step epochs
        if e%print_step==0:
            print(f"| Epoch {e}")
            print(f"| Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}\
                Train Loss: {sum(losses)/len(losses)}\
                Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}\
                Val Loss: {sum(val_losses)/len(val_losses)}\n")

        # Save best model weights 
        if min_valid_loss > (sum(val_losses)/len(val_losses)):
            
            train_loss = sum(losses)/len(losses)
            min_valid_loss = sum(val_losses)/len(val_losses)

            # Saving the model
            print(f"| Epoch {e} - Saving the model with train loss={train_loss:.6f} and val_loss={min_valid_loss:.6f}")
            model_name = str(f'{model_type}_TRAIN_{train_loss:.6f}_VAL_{min_valid_loss:.6f}')
            model.save_pretrained(str('/home/a.lombardi/my_segformer/models/' + model_name + "/"))



if __name__ == "__main__":
    torch.cuda.empty_cache()

    ###############################################
    ####### Getting configuration settings ########
    config = ConfigParser()
    config.read('/home/a.lombardi/my_segformer/configuration.ini')
    BATCH_SIZE = config.getint('TRAINING', 'batch_size')
    PRETRAINED_WEIGHTS = config.get('MODEL', 'pretrained_type')
    ###############################################

    ###############################################
    ############ Preparing the dataset ############

    transforms = tfs.Compose([
        tfs.RandomHorizontalFlip(p=0.5),
        #aug.CenterCrop(1024,1024, always_apply=False, p=0.5),
        tfs.RandomCrop(1024),
        ])
    
    #train_set = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', split='train', transforms=True)
    #val_set = CityscapesDataset(path='/home/a.lombardi/CityScapes_Dataset', split='val', transforms=False)
    
    train_set = ApolloScapeDataset("/home/a.lombardi/ApolloScape_Dataset", split='train', transforms=None)
    print(f"trainset len {len(train_set)}")
    val_set = ApolloScapeDataset("/home/a.lombardi/ApolloScape_Dataset", split='val', transforms=None)
    print(f"valset len {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ###############################################
    
    num_labels = len(train_set.get_label2id())

    ###############################################
    ############## Preparing the model ############
    model = SegformerForSemanticSegmentation.from_pretrained(PRETRAINED_WEIGHTS, # Encoder pretrained weights
                                                        ignore_mismatched_sizes=True,
                                                         num_labels=num_labels, 
                                                         id2label=train_set.get_id2label(), 
                                                         label2id=train_set.get_label2id(),
                                                         reshape_last_stage=True)

    if torch.cuda.is_available():
        print("Loading the model on GPU: ", torch.cuda.get_device_name(0))
        model = model.cuda()
    else:
        print("Using the model on CPU\n")
    ###############################################

    # loss function and optimizer
    # The loss is already been used in the model decode head as the output is given
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = 
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.00006) # Adam, AdamW or RMSprop

    training(model=model, model_type="b1ApolloMODIFIED", 
            train_loader=train_loader, val_loader=val_loader,
            criterion=None, optimizer=optimizer, epochs=250, print_step=1)

    