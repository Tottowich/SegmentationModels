from data.loaders import get_dataloader, ADE20K, ADE20KSingleExample, split_dataset
from typing import Type, List, Tuple, Dict, Optional, Union, Any
from data.DataEvaluation import DatasetEvaluator
# Create dataloader
def create_ADE20K_dataset(img_size,cache,fraction,transform=None,single_example=False,index=0,categorical=True)->Type[ADE20K]:
    if single_example:
        dataset = ADE20KSingleExample(img_size=img_size,transform=transform,fraction=fraction,categorical=categorical,index=index)
    else:
        dataset = ADE20K(cache=cache,img_size=img_size,fraction=fraction,transform=transform,categorical=categorical)
    return dataset

if __name__=="__main__":
    # Create dataset
    # dataset = create_ADE20K_dataset(img_size=(224,224),cache=True,fraction=0.01,single_example=True,index=0)
    dataset = create_ADE20K_dataset(img_size=224,cache=True,fraction=0.1)
    dataloader = get_dataloader(dataset,batch_size=len(dataset)//10,shuffle=False)
    print(f"Dataset length: {len(dataset)}")
    # Create dataloader
    evaluator = DatasetEvaluator(dataset_name="ADE20K_train_0_01",dataset=dataloader,results_dir="results_data",class_names=dataset.class_names)
    evaluator.evaluate()