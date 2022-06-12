import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class ImgDataset(Dataset):
    def __init__(self,img_path,map_path,img_transforms,map_transforms):
        self.img_transform = img_transforms
        self.map_transforms = map_transforms

        self.img_files = sorted(glob.glob(img_path + "/*.*"))
        self.map_files = sorted(glob.glob(map_path+'/*.*'))

    def __getitem__(self,index):
        img = Image.open(self.img_files[index % len(self.img_files)])
        if img.mode =='L':# there are some gray images in the train dataset(dirty data)
            img = F.to_grayscale(img,num_output_channels=3)
        img = self.img_transform(img)
        
        map = Image.open(self.map_files[index % len(self.map_files)])
        map = self.map_transforms(map)
        return {'img':img,'map':map}
    
    def __len__(self):
        return len(self.img_files)