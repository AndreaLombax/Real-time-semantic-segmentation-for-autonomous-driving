from requests import patch
import torch
from math import inf
from utils import bcolors
from model.model import SegFormer

from Dataset import Cityscapes_Dataset
from torch.utils.data import DataLoader, distributed
import torch.distributed as dist

from torchsummary import summary
# mean IoU
from configparser import ConfigParser
import json
import os

# assign gpu devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
nodes = 4
gpus = 4
nr = 0
world_size = nodes*gpus
rank = nr * gpus + 0

os.environ['MASTER_ADDR'] = '193.205.230.3'
os.environ['MASTER_PORT'] = '8889'  
'''

def training(model: torch.nn.Module, device, train_loader, val_loader, criterion, optimizer, num_classes, epochs=1, print_step=10):

    print("-> Run training")
    '''
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=world_size,                              
    	rank=rank                                               
    )

    # Wrap the model ##########
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[device])
    ###########################
    '''

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=0.0000001, verbose=True)
    min_valid_loss = inf
    
    # cycle through epochs
    for e in range(epochs):
        
        model.train() # turn on train mode

        # TRAINING STEP ---------------------|
        train_loss = 0.0

        for data, labels in train_loader: # cycle thorugh batches
            
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            #print("Data shape: ",data.shape)
            #print("Labels shape: ",labels.shape)
            # forward pass 
            output = model(data)
            #print("Output shape: ",output.shape)
            # find the loss
            loss = criterion(output, labels)
            # clear the gradients
            optimizer.zero_grad()
            # calculate gradients
            loss.backward()
            # update weights
            optimizer.step()
            # update the loss
            train_loss += loss.item()
        # ------------------------------------|

        # VALIDATION STEP --------------------|
        valid_loss = 0.0
        with torch.no_grad():   # disable gradient calculation
            model.eval()

            for data, labels in val_loader:

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # forward pass
                output = model(data)
                # find the loss
                loss = criterion(output, labels)
                # calculate loss
                valid_loss += loss.item()
        # ------------------------------------|
        
        scheduler.step(valid_loss)

        # the loss is printed each print_step epochs
        if e%print_step==0:
            print(f'| {bcolors.BOLD}Epoch {e+1}{bcolors.ENDC} | Training Loss: {train_loss/len(train_loader):.5f} | Validation Loss: {valid_loss/len(val_loader):.5f}' )
        if min_valid_loss > valid_loss:
            if e%print_step==0:
                print(f'{bcolors.OKGREEN}| Val loss decreased ({min_valid_loss:.5f}--->{valid_loss:.5f}) - Saving model.{bcolors.ENDC}')
            min_valid_loss = valid_loss

            # Saving the model
            model_name = str(f'model_TRAIN_{train_loss/len(train_loader):.6f}_VAL_{valid_loss/len(val_loader):.6f}.pth')
            torch.save(model.state_dict(), '/home/a.lombardi/my_segformer/models/' + model_name)

    input()
if __name__ == "__main__":
    torch.cuda.empty_cache()

    ################ Getting configuration settings ###############
    config = ConfigParser()
    config.read('/home/a.lombardi/my_segformer/configuration.ini')
    BATCH_SIZE = config.getint('TRAINING', 'batch_size')
    ################################################################

    # load data
    train_set = Cityscapes_Dataset(path='/home/a.lombardi/CityScapes_Dataset', 
                                    batch_size=BATCH_SIZE, 
                                    image_size=config.getint('MODEL','img_train_size'),
                                    split='train')
    val_set = Cityscapes_Dataset(path='/home/a.lombardi/CityScapes_Dataset',
                                    batch_size=BATCH_SIZE, 
                                    image_size=config.getint('MODEL','img_train_size'),
                                    split='train')
    
    '''
    train_sampler = distributed.DistributedSampler(train_set,num_replicas=world_size,rank=rank)
    val_sampler = distributed.DistributedSampler(val_set,num_replicas=world_size,rank=rank)
    '''
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # define and load the model
    model_type = 'MODEL'
    model = SegFormer(in_channels=config.getint(model_type,'in_channels'), #image channels
                        widths=json.loads(config.get(model_type,'widths')),
                        depths=json.loads(config.get(model_type,'depths')),
                        all_num_heads=json.loads(config.get(model_type,'all_num_heads')),
                        patch_sizes=json.loads(config.get(model_type,'patch_sizes')),
                        overlap_sizes=json.loads(config.get(model_type,'overlap_sizes')),
                        reduction_ratios=json.loads(config.get(model_type,'reduction_ratios')),
                        mlp_expansions=json.loads(config.get(model_type,'mlp_expansions')),
                        decoder_channels=config.getint(model_type,'decoder_channels'),
                        scale_factors=json.loads(config.get(model_type,'scale_factors')),
                        num_classes=train_set.getNumClasses(),
                        ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    if torch.cuda.is_available():
        print("Loading the model on GPU: ", torch.cuda.get_device_name(0))
        model = model.cuda()
    else:
        print("Using the model on CPU\n")

    # loss function and optimizer
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = 
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.00008) # Adam, AdamW or RMSprop
    
    training(model, device, train_loader, val_loader, criterion, optimizer, num_classes=train_set.getNumClasses(), epochs=500)