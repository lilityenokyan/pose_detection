import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

FOLDER_PATH = '../dataset/yoga_poses/'
FOLDER_PATH_JPEG = '../dataset/yoga_poses_jpeg/'


class YogaPoseDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.data = pd.read_csv(split_file, header=None, delimiter=',')
        self.transform = transform
        self._filter_missing_images()
        print(self.__len__())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_82 = self.data.iloc[idx, 3]

        img = Image.open(FOLDER_PATH + img_path)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label_82

    def _filter_missing_images(self):
        missing_indices = []
        # Iterate over all image and remove those that do not load
        for idx, img_path in enumerate(self.data.iloc[:, 0]):
            img_path = FOLDER_PATH + img_path
            if not os.path.exists(img_path):
                missing_indices.append(idx)
        self.data = self.data.drop(missing_indices)
        self.data.reset_index(drop=True, inplace=True)


class YogaPoseDatasetTransformer(Dataset):
    def __init__(self, split_file, transform=None):
        self.data = []
        with open(split_file, 'r') as f:
            for line in f:
                # The images for this class are in .jpeg format but their references in txt file are in .jpg
                image_path, label1, label2, label3 = line.strip().split(',')
                label = int(label1)
                image_path = image_path.replace(".jpg", ".jpeg")  # replace .jpg with .jpeg
                self.data.append((image_path, label))
        self.transform = transform
        self._filter_missing_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = read_image(FOLDER_PATH_JPEG + image_path)  # torchvision to read
        # Convert the image to RGB
        image = torch.cat([image, image, image], dim=0) if image.shape[0] == 1 else image

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _filter_missing_images(self):
        missing_indices = []
        # Iterate over all image and remove those that do not load
        for idx, (img_path, _) in enumerate(self.data):
            img_path = FOLDER_PATH_JPEG + img_path
            if not os.path.exists(img_path):
                missing_indices.append(idx)
        for index in reversed(missing_indices):
            self.data.pop(index)
