from data.DataEvaluation import DatasetEvaluator
from data.loaders import get_dataloader, create_ADE20K_dataset
import argparse # For parsing arguments
from typing import Type, List, Tuple, Dict, Optional, Union, Any

if __name__=="__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Evaluate dataset")
    # parser.add_argument("--cache",type=bool,default=True,help="Cache dataset")
    parser.add_argument("--fraction",type=float,default=1,help="Fraction of dataset to use")
    parser.add_argument("--img_size",type=int,default=224,help="Size of image to use")
    # Batch size
    parser.add_argument("--batch_size",type=int,default=50,help="Batch size")
    # parser.add_argument("--single_example",type=bool,default=False,help="Use single example")
    parser.add_argument("--index",type=int,default=0,help="Index of single example")
    # Results path and name
    parser.add_argument("--results_dir",type=str,default="data/results/",help="Path to results directory")
    parser.add_argument("--dataset_name",type=str,default="ADE20K_train_fraction",help="Name of dataset")
    parser.add_argument("--cache",dest="cache",action="store_true")
    parser.add_argument("--single_example",dest="single_example",action="store_true")
    parser.set_defaults(cache=False)
    parser.set_defaults(single_example=False)

    # Parse arguments
    args = parser.parse_args()
    # Create dataset
    dataset = create_ADE20K_dataset(img_size=args.img_size,cache=args.cache,fraction=args.fraction,single_example=args.single_example,index=args.index)
    dataloader = get_dataloader(dataset,batch_size=args.batch_size,shuffle=False)

    # Create dataloader
    evaluator = DatasetEvaluator(dataset_name=args.dataset_name,dataset=dataset,dataloader=dataloader,results_dir=args.results_dir,class_names=dataset.class_names)
    results = evaluator.evaluate()
    figs = evaluator.plot()
    evaluator.interactive_plot()
