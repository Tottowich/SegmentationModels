from types import new_class
from utils.training import ProgBar, Trainer
from utils.metrics import AccuracyMeter, IoUMeter,F1Meter, MetricList
import torch as T   
from data.loaders import get_dataloader, ADE20K, ADE20KSingleExample, split_dataset
from models.UNets import UNet
from torch import optim
from loss.loss_functions import UNetLossFunction
from typing import Type, Union, List, Tuple, Dict, Optional
import warnings
import argparse
# Interupt on user warning
warnings.simplefilter("error", UserWarning)
# Create model
def create_model(config_file:str=None,checkpoint:str=None)->Type[UNet]:
    model = UNet(config=config_file,device="cuda" if T.cuda.is_available() else "cpu",checkpoint=checkpoint)
    return model

# Create optimizer
def create_optimizer(opt:str="SGD",model=None,hyps:dict=None)->Type[optim.Adam]:
    optimizers = {
        "SGD":optim.SGD,
        "Adam":optim.Adam,
        "AdamW":optim.AdamW,
        "Adadelta":optim.Adadelta,
        "Adagrad":optim.Adagrad,
        "Adamax":optim.Adamax,
        "ASGD":optim.ASGD,
        "LBFGS":optim.LBFGS,
        "RMSprop":optim.RMSprop,
        "Rprop":optim.Rprop,
        "SparseAdam":optim.SparseAdam,
    }
    assert opt in optimizers.keys(), f"Optimizer {opt} not supported only supports {optimizers.keys()}"
    if model is None:
        return optimizers[opt]
    else:
        return optimizers[opt](model.parameters(),**hyps)

# Create loss function
def create_criterion()->Type[UNetLossFunction]:
    """
    Create loss function
    """
    loss_function = UNetLossFunction()
    return loss_function

def create_ADE20K_dataset(img_size,cache,fraction,transform=None,single_example=False,index=0)->Type[ADE20K]:
    """
    Create ADE20K dataset
    INPUT:
        img_size: tuple of ints - size of image to use
        cache: bool - cache dataset
        fraction: float - fraction of dataset to use
    """
    if single_example:
        dataset = ADE20KSingleExample(img_size=img_size,transform=transform,fraction=fraction,categorical=True,index=index)
    else:
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
                criterion,
                dataset,
                batch_size,
                num_workers,
                device,
                save_path,
                hyper_parameters,
                wandb_run,
                epochs,
                log_interval,
                save_interval,
                save_best,
                save_last,
                val_interval,
                checkpoint,
                resume,
                verbose,
                metric_list,
                pbar,
                )->Type[Trainer]:
    if len(dataset) > 1:
        train_set, val_set, test_set = split_dataset(dataset)
        train_loader = create_dataloader(train_set,batch_size,num_workers)
        val_loader = create_dataloader(val_set,batch_size,num_workers)
        test_loader = create_dataloader(test_set,batch_size,num_workers)
    else:
        train_loader = create_dataloader(dataset,batch_size,num_workers)
        val_loader = train_loader
        test_loader = train_loader
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_path=save_path,
        hyper_parameters=hyper_parameters,
        wandb_run=wandb_run,
        epochs=epochs,
        log_interval=log_interval,
        save_interval=save_interval,
        save_best=save_best,
        save_last=save_last,
        val_interval=val_interval,
        checkpoint=checkpoint,
        resume=resume,
        verbose=verbose,
        metric_list=metric_list,
        pbar=pbar,
    )
    return trainer
def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",action="store_true",help="Use GUI to select config file")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--log_interval",type=int,default=10)
    parser.add_argument("--save_interval",type=int,default=1)
    parser.add_argument("--val_interval",type=int,default=1)
    parser.add_argument("--save_best",type=bool,default=True)
    parser.add_argument("--save_last",type=bool,default=True)
    parser.add_argument("--resume",type=bool,default=False)
    parser.add_argument("--verbose",type=bool,default=True)
    parser.add_argument("--config_file",type=str,default="config.json")
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument("--save_path",type=str,default="checkpoints")
    parser.add_argument("--cache",type=str,default="cache")
    parser.add_argument("--img_size",type=int,default=256)
    parser.add_argument("--fraction",type=float,default=1.0)
    parser.add_argument("--single_example",type=bool,default=False)
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--wandb",type=bool,default=False)
    parser.add_argument("--wandb_project",type=str,default="UNet")
    parser.add_argument("--wandb_entity",type=str,default="joseph")
    parser.add_argument("--wandb_run",type=str,default=None)
    parser.add_argument("--wandb_tags",type=str,default=None)
    parser.add_argument("--wandb_notes",type=str,default=None)
    parser.add_argument("--wandb_name",type=str,default=None)
    parser.add_argument("--wandb_group",type=str,default=None)
    parser.add_argument("--wandb_job_type",type=str,default=None)
    parser.add_argument("--wandb_save_code",type=bool,default=False)
    parser.add_argument("--wandb_resume",type=bool,default=False)
    parser.add_argument("--wandb_id",type=str,default=None)
    parser.add_argument("--wandb_offline",type=bool,default=False)
    parser.add_argument("--wandb_config",type=str,default=None)
    parser.add_argument("--wandb_config_exclude_keys",type=str,default=None)
    parser.add_argument("--wandb_config_include_keys",type=str,default=None)
    parser.add_argument("--wandb_config_patch",type=str,default=None)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = input_arguments()
    if args.gui:
        from utils.train_gui import TrainGui
        gui = TrainGui((1280,720))
        gui.run()
    else:

        config_file = None #"./models/model_configs/UNet.yaml"
        checkpoint = "runs/UNet_2022-10-14_21-53-51/checkpoints/epoch_80.pt"
        # print(config_file)
        model = create_model(config_file,checkpoint)
        # model = UNet
        hyps = {"lr":1e-4,"weight_decay":1e-5}
        optimizer = create_optimizer(model,hyps)
        optimizer = optim.Adam
        # optimizer = None
        dataset = create_ADE20K_dataset(img_size=model.img_size[0],cache=True,fraction=1.0,single_example=True)
        # dataset = create_dataset(img_size=256,cache=True,fraction=0.0001,single_example=True)
        criterion = create_criterion()
        n_classes = dataset.num_classes
        trainer = create_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=dataset,
            batch_size=10,
            num_workers=4,
            device="cuda" if T.cuda.is_available() else "cpu",
            save_path="./runs/UNet",
            hyper_parameters=hyps,
            wandb_run=None,
            epochs=100,
            log_interval=5,
            save_interval=20,
            save_best=True,
            save_last=True,
            val_interval=5,
            checkpoint=checkpoint,
            resume=False,
            verbose=True,
            metric_list=MetricList(metrics=[AccuracyMeter(n_classes=n_classes),IoUMeter(num_classes=n_classes),F1Meter(n_classes=n_classes)]),
            pbar=True,
        )
        trainer.train()
        # model = UNet
    # hyps = {"lr":1e-4,"weight_decay":1e-5}
    # optimizer = optim.Adam
    # dataset = create_dataset(img_size=256,cache=True,fraction=0.05,single_example=True)
    # criterion = create_criterion()
    # n_classes = dataset.num_classes
    # trainer = create_trainer(
    #     model=model,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     dataset=dataset,
    #     batch_size=10,
    #     num_workers=4,
    #     device="cuda" if T.cuda.is_available() else "cpu",
    #     save_path="./runs/UNet",
    #     hyper_parameters=hyps,
    #     wandb_run=None,
    #     epochs=1,
    #     log_interval=5,
    #     save_interval=10,
    #     save_best=True,
    #     save_last=True,
    #     val_interval=5,
    #     checkpoint="runs/UNet_2022-10-14_20-44-36/checkpoints/best_model.pt",
    #     resume=True,
    #     verbose=True,
    #     metric_list=MetricList(metrics=[AccuracyMeter(n_classes=n_classes),IoUMeter(n_classes=n_classes),F1Meter(n_classes=n_classes)]),
    #     pbar=True,
    # )
    # trainer.train()























#  config_file = "../models/model_configs/UNet.yaml"
        # unet = UNet(config=config_file, in_channels=self.in_channels,verbose=True).to(self.device)


