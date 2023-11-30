import torch.utils.data as data
import cv2 as cv
import os
from PIL import Image

class CustomDataset(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
    
    @staticmethod
    def _load_input_image(path):
        with open(path, 'rb') as f:
            # img = Image.open(f)
            img = Image.open(f).convert('RGB')
            # return img.convert('L')
            # img = cv.cvtColor(cv.imread(path, 3), cv.COLOR_BGR2RGB)
            # img = transforms.ToTensor()(img)
            return img

    def __init__(self, images_dir, df, transforms=None):
        self.images_dir = images_dir
        self.images_paths = sorted(img_path for img_path in os.listdir(images_dir) if self._isimage(img_path, self.IMG_EXTENSIONS))
        self.labels = list(df['label'].values)
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img = self._load_input_image(os.path.join(self.images_dir, self.images_paths[idx]))
        lbl = self.labels[idx]
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, lbl

    def __len__(self):
        return len(self.images_paths)