# Gui used to train the model.
# Author: Theodor Jonsson (thjo0148@student.umu.se)
# Date: 
# Version: 1.0
from ctypes import alignment
import os
import sys
from pydantic import NoneIsAllowedError
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch as T
from utils.training import Trainer, create_ADE20K_dataset, create_dataloader, create_trainer, create_model,create_criterion,create_optimizer,random_split
from utils.metrics import Metric,MetricList, IoUMeter, F1Meter, AccuracyMeter
import PySimpleGUI as sg
from pathlib import Path
from typing import Type, List, Tuple, Dict, Union, Optional


class TrainGui:
    """
    Training Gui.
    Includes:
        - Ability to select fraction of complete dataset to use.
        - Ability to use only single image dataset to check ability to converge.
        - Ability to cache dataset for faster image loading. Takes time to preprocess data.
        - Selection of:
            - Optimizer.
            - Model config file: (Optional[str])
            - Model checkpoint: (Optional[str])
            - Hyper parameters.
                - Hyper parameters are:
                    - Batch size: (int), bs
                    - Learning Rate: (float), lr
                    - Weight Decay: (float), weight_decay
                    - Momentum: (float), momentum
                    - Epochs: (int), epochs
                    - Clip: (float), clip
            - Metrics: (Optional[list[Metric]]) Selection of all modules that are subclasses of Metric.
            - Number of workers: (int) Number of workers used to load data. 
            - Save interval: (int) Number of epochs between each automatic save.
            - Val interval: (int) Number of epochs between each pass of the validation dataset.
        - Ability to start training.
        TODO: - Sample interface, input image -> output segmentation
            
    """
    def __init__(self,window_size:tuple[int]=(800,600),theme:str="DarkGray8",font:str="Helvetica 14",font_size:int=14):
        """
        Args:
            window_size (tuple[int]): Size of the window.
        """
        sg.theme(theme)
        self.window_size = window_size
        self.font = font
        self.font_size = font_size
        self.metrics = [metric.__name__ for metric in Metric.__subclasses__()]
        self.optimizers = [class_name.__name__ for class_name in T.optim.Optimizer.__subclasses__()]
        print(self.metrics)
        self.gui = self.create_gui()
        # Open window
        while True:
            event, values = self.gui.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == "Start training":
                self.init_trainer(values)
    def create_gui(self):
        """
        Creates the gui.
        """
        # Create layout
        layout = []
        # Column 1 should contain all the options as two columns with labels and inputs.
        parent_path = Path(__file__).parent.parent
        devices = [f"cuda:{i}" for i in range(T.cuda.device_count())]+["cpu"]
        col1 = [
                [sg.Text("Training Gui",font=(self.font, 25))],
                # Align slider with text
                [sg.Text("Fraction of dataset to use",font=(self.font, self.font_size)),sg.Slider(range=(0,1),default_value=1,orientation="horizontal",resolution=0.01,size=(10,10),font=(self.font, self.font_size//2),key="Fraction")],
                [sg.Text("Use single example dataset",font=(self.font, self.font_size)),sg.Checkbox("",key="single_example")],
                [sg.Text("Cache dataset",font=(self.font, self.font_size)),sg.Checkbox("",key="cache")],
                [sg.Text("Optimizer",font=(self.font, self.font_size)),sg.InputCombo(self.optimizers,default_value=self.optimizers[0],key="optimizer")],
                # Choose config file and checkpoint file for model using file browser.
                [sg.Text("Model config file",font=(self.font, self.font_size)),sg.InputText(key="model_config_file",size=(20,1),default_text="../models/model_configs/UNet.yaml"),sg.FileBrowse(file_types=(("*.yaml"),),initial_folder=parent_path)],
                [sg.Text("Model checkpoint",font=(self.font, self.font_size)),sg.InputText(key="model_checkpoint",size=(20,1)),sg.FileBrowse(file_types=(("Checkpoint Files",("*.pt","*.pth"),),),initial_folder=parent_path)],
                [sg.Text("Save directory",font=(self.font, self.font_size)),sg.InputText(default_text="./runs/UNet",key="save_dir",size=(30,1)),sg.FileBrowse(file_types=(("Directory","*"),),initial_folder=parent_path)],
                # [sg.Text("Model config file",font=(self.font, self.font_size)),sg.InputText(default_text="",key="config_file",size=(30,1))],
                # [sg.Text("Model checkpoint",font=(self.font, self.font_size)),sg.InputText(default_text="",key="checkpoint",size=(30,1))],
                [sg.Text("Hyper parameters",font=(self.font, self.font_size))],
                [sg.Text("Batch size",font=(self.font, self.font_size)),sg.InputText(default_text="1",key="bs",size=(10,1))],
                [sg.Text("Learning rate",font=(self.font, self.font_size)),sg.InputText(default_text="0.001",key="lr",size=(10,1))],
                [sg.Text("Weight decay",font=(self.font, self.font_size)),sg.InputText(default_text="0.0005",key="weight_decay",size=(10,1))],
                [sg.Text("Momentum",font=(self.font, self.font_size)),sg.InputText(default_text="0.9",key="momentum",size=(10,1))],
                [sg.Text("Alpha",font=(self.font, self.font_size)),sg.Slider(range=(0,4),default_value=1,orientation="horizontal",resolution=0.02,size=(10,10),font=(self.font, self.font_size//2),key="Alpha")],
                [sg.Text("Beta",font=(self.font, self.font_size)),sg.Slider(range=(0,4),default_value=1,orientation="horizontal",resolution=0.02,size=(10,10),font=(self.font, self.font_size//2),key="Beta")],
                [sg.Text("Epochs",font=(self.font, self.font_size)),sg.InputText(default_text="100",key="epochs",size=(10,1))],
                [sg.Text("Clip",font=(self.font, self.font_size)),sg.InputText(default_text="1.0",key="clip",size=(10,1))],
                # [sg.Listbox(values=self.metrics, size=(40, 20), key="-LIST-"),sg.Checkbox("All",key="-ALL-",enable_events=True)],
                [sg.Text("Number of workers",font=(self.font, self.font_size)),sg.InputText(default_text="0",key="num_workers",size=(10,1))],
                [sg.Text("Save interval",font=(self.font, self.font_size)),sg.InputText(default_text="1",key="save_interval",size=(10,1))],
                [sg.Text("Val interval",font=(self.font, self.font_size)),sg.InputText(default_text="1",key="val_interval",size=(10,1))],
                [sg.Text("Device",font=(self.font, self.font_size)),sg.Listbox(values=devices, size=(10, 2), key="device",default_values=devices[0])],
                [sg.Text("Resume",font=(self.font, self.font_size)),sg.Checkbox("",key="resume")],
                [sg.Text("Save best",font=(self.font, self.font_size)),sg.Checkbox("",key="save_best")],
                [sg.Text("Save last",font=(self.font, self.font_size)),sg.Checkbox("",key="save_last")],
                [sg.Text("Verbose",font=(self.font, self.font_size)),sg.Checkbox("",key="verbose")],
                [sg.Text("Progress bar",font=(self.font, self.font_size)),sg.Checkbox("",key="pbar")],

        ]
        col2 = [[sg.Text("Metrics",font=(self.font, self.font_size))]] + [[sg.Checkbox(metric,font=(self.font, int(self.font_size/1.5)),key=metric)] for metric in self.metrics]
        col1 = [[sg.Column(col1,vertical_alignment="top")]]
        # Center column 2 vertically
        col2 = [[sg.Column(col2,vertical_alignment="bottom")]]
        # layout.append([sg.Button("Start training")])
        layout = [
            [sg.Button("Start training")],
            [sg.Column(col1),sg.Column(col2,scrollable=True,vertical_scroll_only=True,size=(self.window_size[0]/2,self.window_size[1]/2))],
            # [sg.Button("Start training")]
        ]
        layout = [[sg.Column(layout,scrollable=True,size=self.window_size,vertical_scroll_only=True)]]
        window = sg.Window("Training Gui",layout,size=self.window_size)
        return window
    def init_trainer(self,values):
        num_workers =               int(values["num_workers"])
        weight_decay =              float(values["weight_decay"])
        momentum =                  float(values["momentum"])
        epochs =                    int(values["epochs"])
        clip =                      float(values["clip"])
        save_interval =             int(values["save_interval"])
        val_interval =              int(values["val_interval"])
        lr =                        float(values["lr"])
        bs =                        int(values["bs"])
        opt =                       values["optimizer"]
        device =                    values["device"][0]
        alpha =                     float(values["Alpha"])
        beta =                      float(values["Beta"])
        save_path =                 values["save_dir"]
        # Get metrics
        metrics =                   [metric for metric in self.metrics if values[metric]]
        model_config_file =         values["model_config_file"]
        model_checkpoint =          values["model_checkpoint"]
        cache =                     values["cache"]
        single_example =            values["single_example"]
        fraction =                  values["Fraction"]
        save_best =                 values["save_best"]
        save_last =                 values["save_last"]
        verbose =                   values["verbose"]
        pbar =                      values["pbar"]
        resume =                    values["resume"]

        metric_list = MetricList(metrics=metrics,n_classes=151)
        print("Model checkpoint",model_checkpoint)
        model_checkpoint = model_checkpoint if len(model_checkpoint) else None
        print("Model checkpoint",model_checkpoint)
        print("Model config file",model_config_file)
        model_config_file = model_config_file if len(model_config_file) and model_checkpoint is None else None
        print("Model config file",model_config_file)
        model = create_model(model_config_file,model_checkpoint)
        criterion = create_criterion(alpha=alpha,beta=beta)
        train_dataset = create_ADE20K_dataset(img_size=model.img_size,fraction=fraction,cache=cache,single_example=single_example,train=True)
        train_loader = create_dataloader(train_dataset,bs,num_workers)
        # train_loader = create_dataloader(train_dataset,bs,num_workers)
        if not single_example:
            val_dataset = create_ADE20K_dataset(img_size=model.img_size,fraction=fraction,cache=cache,single_example=single_example,train=False)
            val_dataset, test_dataset = random_split(val_dataset,[int(len(val_dataset)*0.5),int(len(val_dataset)*0.5)])
            val_loader = create_dataloader(val_dataset,bs,num_workers)
            test_loader = create_dataloader(test_dataset,bs,num_workers)
        else:
            val_loader = None
            test_loader = None
        hyps = {"lr":lr,"weight_decay":weight_decay,"momentum":momentum,"clip":clip}
        assert opt in self.optimizers, f"{opt} Optimizer is not supported. Choose from {self.optimizers}."
        optimizer = create_optimizer(opt=opt,model=model,hyps=hyps)
        # Get optimizer parameters
        optimizer_params = optimizer.param_groups[0]
        # trainer = create_trainer(model=model,optimizer=optimizer,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,metrics=metrics,epochs=epochs,save_interval=save_interval,val_interval=val_interval)
        self.trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_path=save_path,
            hyper_parameters=optimizer_params,
            wandb_run=None, # TODO: Add wandb support
            epochs=epochs,
            log_interval=save_interval,
            save_interval=save_interval,
            save_best=save_best,
            save_last=save_last,
            val_interval=val_interval,
            checkpoint=model_checkpoint,
            resume=resume,
            verbose=verbose,
            metric_list=metric_list,
            pbar=pbar,
        )

if __name__=="__main__":
    # sg.theme_previewer()
    TrainGui()