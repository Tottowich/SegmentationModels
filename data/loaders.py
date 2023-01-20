import os, sys
from regex import P
# Add the parent directory to the path
if __name__=="__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision as TV
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from data.loaders_utils import plot_polygon
from utils.helper import to_2tuple
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Callable
class ADE20K_Dataset(Dataset):
    def __init__(self, root, img_size:tuple=(256,256), split=None, transform=None,target_transform=None,cache=False):
        self.num_classes = 150
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cached = cache
        self.img_size = to_2tuple(img_size)
        self.reshaper = transforms.Resize((self.img_size))
        self.reshaper_mask = transforms.Resize((self.img_size), interpolation=Image.NEAREST)
        self.files = self.get_files()
        self.data_pairs = []
        if self.cached:
            self.data = self.load_full_dataset()
            self.check_dataset()
        self.length = len(self.files)
    def get_files(self):
        return [os.path.join(self.root, name) for name in os.listdir(self.root) if name.endswith('.jpg') and name.startswith("ADE")]
    def load_full_dataset(self):
        data = []
        for file in tqdm(self.files, desc='Caching dataset'):
            data = self.loadAde20K(file)
            img = data['img']
            seg = data['segmentation'] # Should have values between 0 and 149
            # instance_mask = data['instance_mask']
            data_pair = self.preprocesss(img, seg)
            self.data_pairs.append(data_pair)
            # exit()
            # Resize the image and masks to the desired size
            # data.append(self.loadAde20K(file))

        return data
    def preprocesss(self, img, seg):
        # Resize the image and masks to the desired size
        # print("Image shape: ", img.shape)
        # print("Segmentation shape: ", seg.shape)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.Tensor(img)/255 # Normalize the image
        img = self.reshaper(img)[None]
        # R = seg[0,:,:] 
        # G = seg[1,:,:] 
        # B = seg[2,:,:]
        R = seg[:,:,0]
        G = seg[:,:,1]
        B = seg[:,:,2]
        print("G max: ", G.max(), "G min: ", G.min(), "R max: ", R.max(), "R min: ", R.min())
        # print("G max: ", G.max())
        # print("B max: ", B.max())

        ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32)) 
        # ObjectClassMasks = self.to_categorical(ObjectClassMasks)
        ObjectClassMasks = np.expand_dims(ObjectClassMasks, axis=-1)
        # print(ObjectClassMasks.shape)
        # print(np.sum(ObjectClassMasks[:,:,116]))
        # seg = F.one_hot(ObjectClassMasks, self.num_classes)
        class_mask = T.Tensor(ObjectClassMasks.transpose(2,0,1))[None]
        class_mask = self.reshaper_mask(class_mask).to(T.int32)
        # print("Class mask shape: ", class_mask.shape)
        # plt.imshow(class_mask[0,0])
        # plt.show()
        # print("Class mask shape and type: ", class_mask.shape, class_mask.dtype)
        # print("Image shape and type: ", img.shape, img.dtype)
        return img, class_mask
    # def load_single(self, path):
    #     return self.loadAde20K(path)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        if self.cached:
            data = self.data[index]
        else:
            data = self.loadAde20K(self.files[index])
        img = data['img']
        seg = data['segmentation']
        R = seg[:,:,0] 
        # G = seg[:,:,1] 
        # B = seg[:,:,2] 
        # ObjectClassMasks = (R/150).astype(np.int32)*256+(G.astype(np.int32)) 
        # Transform the image and masks
        if self.transform is not None:
            pass
        return data
    def to_categorical(self,y):
        """ 1-hot encodes a tensor """
        if hasattr(self, "_to_categorical"):
            self._to_categorical = np.eye(self.num_classes, dtype='uint8')
        return self._to_categorical[y]
    def random_example(self):
        index = np.random.randint(0, len(self))
        if self.cached:
            d = self.data[index]
        else:
            d = self.loadAde20K(self.files[index])
        return d, index
    def show_example(self, index:int=None):
        if index is None:
            data,_ = self.random_example()
        else:
            data = self[index]
        img = data['img'] # Torch tensor
        class_mask = data['class_mask'] # Numpy array 
        instance_mask = data['instance_mask'] # Numpy array
        img_np = img.numpy().transpose(1,2,0).astype(np.uint8)
        Image.fromarray(img_np).show("Original Image")
        # Image.fromarray(class_mask/255).show("Class Mask")
        Image.open(data['segm_name']).show("Segmentation Mask")
    def print_example(self, index:int=None):
        if index is None:
            data,_ = self.random_example()
        else:
            data = self[index]
        img = data['img'] # Numpy array
        class_mask = data['class_mask'] # Numpy array
        instance_mask = data['instance_mask'] # Numpy array
    def loadAde20K(self,file):
        # with Image.open(file).convert("RGB") as f:
        with Image.open(file).convert("RGB") as f:
            img = np.array(f)
        fileseg = file.replace('.jpg', '_seg.png') 
        with Image.open(fileseg) as io:
            seg = np.array(io)
        return {'img':img, 'segmentation':seg, 'segm_name':fileseg}
        # Obtain the segmentation mask, bult from the RGB channels of the _seg file
        # R = seg[:,:,0] 
        # G = seg[:,:,1] 
        # B = seg[:,:,2] 
        # ObjectClassMasks = (R/150).astype(np.int32)*256+(G.astype(np.int32)) 


        # Obtain the instance mask from the blue channel of the _seg file
        # Minstances_hat = np.unique(B, return_inverse=True)[1]
        # Minstances_hat = np.reshape(Minstances_hat, B.shape)
        # ObjectInstanceMasks = Minstances_hat
        # return {'img': T.Tensor(img).permute(2,0,1), 'img_name': file, 'segm_name': fileseg,
        #         'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks}
    def check_dataset(self):
        max_id = 0
        for file in tqdm(self.files, desc='Checking dataset'):
            data = self.loadAde20K(file)
            img = data['img']
            seg = data['segmentation']
            R = seg[:,:,0] 
            G = seg[:,:,1] 
            B = seg[:,:,2]
            max_id = max(np.max(B), max_id)
        print("Max id: ", max_id)


# Create single image dataset from images located in ./datasets/drone_seg/dataset/semantic_drone_dataset/original_images and ./datasets/drone_seg/dataset/semantic_drone_dataset/label_images_semantic
class DroneSegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.original_images = ImageFolder(root_dir + '/original_images', transform=transform)
        self.label_images = ImageFolder(root_dir + '/label_images_semantic', transform=transform)
        self.num_samples = len(self.original_images)
    def __getitem__(self, index):
        original_image = self.original_images[index][0]
        label_image = self.label_images[index][0]
        return original_image, label_image
    def __len__(self):
        return self.num_samples

class CoCoSegDataset(Dataset):
    """
    Using torchvision.datasets.CoCoDetection:
    """
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.dataset = TV.datasets.CocoDetection(root, annFile)
        self.num_samples = len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return self.num_samples

class ADE20K(Dataset):
    def __init__(self,train:bool=True,cache:bool=False,img_size:int=520,transform:transforms=None,categorical:bool=False,fraction:float=None) -> None:
        from data.ADE20KSegmentation import ADE20KSegmentation
        super().__init__()
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.dataset = ADE20KSegmentation(split='train' if train else 'val', transform=transform,crop_size=img_size,categorical=categorical)
        self.fraction = fraction if fraction is not None else 1.0
        self.class_names = [name.split(",")[0] for name in ADE20KSegmentation.CLASSES]
            
        self._num_classes = self.dataset.num_class
        self.cached = cache
        if cache:
            self._cache_data()
    def _cache_data(self):
        """ Cache the data in memory """
        print(f"Caching {len(self)} images in memory")
        self.data = [self.dataset[i] for i in tqdm(range(int(len(self.dataset)*self.fraction)), desc='Caching data')]
    def __len__(self) -> int:
        return int(len(self.dataset)*self.fraction)
    @property
    def num_classes(self) -> int:
        return self._num_classes
    @property
    def num_channels(self) -> int:
        return 3
    @property
    def img_size(self) -> int:
        return self.dataset.crop_size
    def __getitem__(self, index: int) -> tuple[T.Tensor,T.Tensor]:
        if self.cached:
            d = self.data[index]
        else:
            d = self.dataset[index]
        return d
    def print_labels(self,index,show:bool=True,display_special:int=None) -> None:
        """ Print the labels of a label image """
        img, label = self[index]
        if show:
            plt.imshow(label)
            plt.title('Label image')
            plt.show()
            if display_special is not None:
                plt.imshow((label==display_special)*255, cmap='gray')
                plt.title(self.class_names[display_special])
                plt.show("Special label image")
            plt.imshow(img.permute(1,2,0))
            plt.title('Image')
            plt.show()
        classes = np.unique(label)

        for c in classes:
            print(f'Class {c}:{self.class_names[c]} has {np.sum(label.numpy()==c)} pixels')

class ADE20KSingleExample(Dataset):
    def __init__(self,train:bool=True,img_size:int=520,transform:transforms=None,fraction:float=1.0,categorical:bool=False,index:int=0) -> None:
        from data.ADE20KSegmentation import ADE20KSegmentation
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Check if tuple or list
        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]
        self.dataset = ADE20KSegmentation(split='train' if train else 'val', transform=transform,crop_size=img_size,categorical=categorical)
        self._num_classes = self.dataset.num_class
        self.example = self.dataset[index]
        self.dataset = [self.example for _ in range(int(1000*fraction))]
        self.class_names = [name.split(",")[0] for name in ADE20KSegmentation.CLASSES]
    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def num_classes(self) -> int:
        return self._num_classes
    def __getitem__(self, index: int) -> tuple[T.Tensor,T.Tensor]:
        return self.dataset[index]
    def __iter__(self):
        return iter(self.example)
    def __next__(self):
        return next(self.example)
    def print_labels(self,index,show:bool=True,display_special:int=None) -> None:
        """ Print the labels of a label image """
        img, label = self[index]
        if show:
            plt.imshow(label)
            plt.title('Label image')
            plt.show()
            if display_special is not None:
                plt.imshow((label==display_special)*255, cmap='gray')
                plt.title(self.class_names[display_special])
                plt.show("Special label image")
            plt.imshow(img.permute(1,2,0))
            plt.title('Image')
            plt.show()
        classes = np.unique(label)
        for c in classes:
            print(f'Class {c}:{self.class_names[c]} has {np.sum(label.numpy()==c)} pixels')

def get_dataloader(dataset:Dataset,batch_size:int=1,shuffle:bool=False,num_workers:int=0,drop_last:bool=False) -> DataLoader:
    
    dloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)
    return dloader

def split_dataset(dataset:Dataset,split:list[float,float,float]=[0.9,0.05,0.05]) -> tuple[Dataset,Dataset,Dataset]:
    """ Split a dataset into train, val and test """
    assert sum(split) == 1.0, 'Split must sum to 1'
    train_len = int(len(dataset)*split[0])
    val_len = int(len(dataset)*split[1])
    test_len = len(dataset) - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])





if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
        # transforms.RandomAdjustSharpness(0.5),
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = DroneSegDataset(root_dir='./datasets/drone_seg/dataset/semantic_drone_dataset', transform=transform)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    # for i, data in enumerate(dataloader):
    #     original_images, label_images = data
    #     print(original_images.shape)
    #     print(label_images.shape)
        # break
    # dataset = ADE20K_Dataset(root='.\datasets\ADE20K\ADE20K_2017_05_30_consistency\images\consistencyanalysis\original_ade20k', transform=transform,target_transform=target_transform,cache=True)
    # ex,index = dataset.random_example()
    # print(ex['img'].shape)
    # print(type(ex['img']))
    # print("Max: ", ex['img'].max())
    # print(ex['class_mask'].shape)
    # print(type(ex['class_mask']))
    # print("Max: ", ex['class_mask'].max())
    # print("Min: ", ex['class_mask'].min())
    # print(np.unique(ex['class_mask']))
    # print(ex['instance_mask'].shape)
    # dataset.show_example(index)
    dataset = ADE20K(cache=False,transform=transform)
    ex,index = dataset[0]
    # dataset = CoCoSegDataset(root='./datasets/coco_seg/val2017', transform=transform)