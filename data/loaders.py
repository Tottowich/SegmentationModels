import os, sys
# Add the parent directory to the path
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
from loaders_utils import plot_polygon
from utils.helper import to_2tuple
from sklearn.utils import class_weight
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
            # Resize the image and masks to the desired size
            self.data_pairs.append(data_pair)
            # data.append(self.loadAde20K(file))
            break

        return data
    def preprocesss(self, img, seg):
        # Resize the image and masks to the desired size
        print("Image shape: ", img.shape)
        print("Segmentation shape: ", seg.shape)
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
        ObjectClassMasks = (R/150).astype(np.int32)*256+(G.astype(np.int32)) 
        ObjectClassMasks = self.to_categorical(ObjectClassMasks)
        # print(ObjectClassMasks.shape)
        # print(np.sum(ObjectClassMasks[:,:,116]))
        # seg = F.one_hot(ObjectClassMasks, self.num_classes)
        class_mask = T.Tensor(ObjectClassMasks.transpose(2,0,1))[None]
        class_mask = self.reshaper_mask(class_mask).to(T.int8)
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
        return np.eye(self.num_classes, dtype='uint8')[y]
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
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
        transforms.RandomAdjustSharpness(0.5),
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
    dataset = ADE20K_Dataset(root='.\datasets\ADE20K\ADE20K_2017_05_30_consistency\images\consistencyanalysis\original_ade20k', transform=transform,target_transform=target_transform,cache=True)
    ex,index = dataset.random_example()
    print(ex['img'].shape)
    print(type(ex['img']))
    print("Max: ", ex['img'].max())
    print(ex['class_mask'].shape)
    print(type(ex['class_mask']))
    print("Max: ", ex['class_mask'].max())
    print("Min: ", ex['class_mask'].min())
    print(np.unique(ex['class_mask']))
    print(ex['instance_mask'].shape)
    dataset.show_example(index)

    # dataset = CoCoSegDataset(root='./datasets/coco_seg/val2017', transform=transform)