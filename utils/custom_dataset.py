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
            img = Image.open(f).convert('RGB')
            return img

    def __init__(self, images_dir, df, transforms=None, is_test=False, with_path=False):
        self.images_dir = images_dir
        self.is_test = is_test
        if self.is_test == False:
            self.labels = []
        self.images_paths = []
        original_images_paths = [img_path for img_path in os.listdir(images_dir) if self._isimage(img_path, self.IMG_EXTENSIONS)]
        for path in original_images_paths:
            if is_test:
                self.images_paths = [img_path for img_path in os.listdir(images_dir) if self._isimage(img_path, self.IMG_EXTENSIONS)]
                pass
            else:
                label = df[df['id'] == f'train/{path}']['label'].item()
                self.labels.append(label)
                self.images_paths.append(f'{path}')
                    
        self.is_test = is_test
        self.transforms = transforms
        self.with_path = with_path
        
    def __getitem__(self, idx):
        img = self._load_input_image(os.path.join(self.images_dir, self.images_paths[idx]))
        if self.is_test == False:
            lbl = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        if self.is_test == False:
            if self.with_path == False:
                return img, lbl
            else:
                return img, lbl, os.path.join(self.images_dir, self.images_paths[idx])
        else:
            # print(os.path.join(self.images_dir, self.images_paths[idx]))
            return img

    def __len__(self):
        return len(self.images_paths)