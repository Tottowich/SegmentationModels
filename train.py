from utils.training import Trainer
import torch as T   
from data.loaders import get_dataloader, ADE20K, ADE20KSingleExample, split_dataset
from models.UNets import UNet
from torch import optim
from loss.loss_functions import UNetLossFunction

# Create model
def create_model(config_file,in_channels=3):
    model = UNet(in_channels=in_channels,config=config_file,device="cuda" if T.cuda.is_available() else "cpu")
    return model

# Create optimizer
def create_optimizer(model,hyps:dict):
    optimizer = optim.Adam(model.parameters(),lr=hyps["lr"],weight_decay=hyps["weight_decay"])
    return optimizer

# Create loss function
def create_loss_function():
    loss_function = UNetLossFunction()
    return loss_function

# Create dataloader
def create_dataset(img_size,cache,fraction,transform=None):
    dataset = ADE20K(cache=cache,img_size=img_size,fraction=fraction,transform=transform,categorical=True)
    return dataset

# Create dataloader
def create_dataloader(dataset,batch_size,num_workers):
    dataloader = get_dataloader(dataset,batch_size=batch_size,num_workers=num_workers)
    return dataloader

# Create trainer
def create_trainer(
                model,
                optimizer,
                loss_function,
                dataset,
                batch_size,
                num_workers,
                device,
                save_path,
                hyper_parameters,
                wandb_run,
                log_interval,
                save_interval,
                save_best,
                save_last,
                ):
                
    test_set, val_set, train_set = split_dataset(dataset)
    train_loader = create_dataloader(train_set,batch_size,num_workers)
    val_loader = create_dataloader(val_set,batch_size,num_workers)
    test_loader = create_dataloader(test_set,batch_size,num_workers)
    trainer = Trainer(model,optimizer,loss_function,train_loader,val_loader,test_loader,device,save_path,hyper_parameters,wandb_run,log_interval,save_interval,save_best,save_last)
    return trainer

if __name__=="__main__":
    config_file = "./models/model_configs/UNet.yaml"
    model = create_model(config_file)
    hyps = {"lr":1e-3,"weight_decay":1e-5}
    optimizer = create_optimizer(model,hyps)
    loss_function = create_loss_function()
    dataset = create_dataset(img_size=512,cache=True,fraction=0.001)
    trainer = create_trainer(model,optimizer,loss_function,dataset,batch_size=4,num_workers=4,device="cuda",save_path="models/saved_models",hyper_parameters=hyps,wandb_run=None,log_interval=10,save_interval=100,save_best=True,save_last=True)























#  config_file = "../models/model_configs/UNet.yaml"
        # unet = UNet(config=config_file, in_channels=self.in_channels,verbose=True).to(self.device)


