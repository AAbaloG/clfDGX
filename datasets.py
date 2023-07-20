import csv
import cv2
import numpy as np
from skimage.transform import resize
import torch
from torch.utils.data import Dataset


class FilteredDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.__data = []
        self.__labels = []
        self.transform = transform
        self.__n_clases = 5

        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                label = int(row[1])
                data_point = row[2]

                self.__labels.append(label)
                self.__data.append(data_point)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.to_list()

        img_name = self.__data[item]
        image = cv2.imread(img_name)

        new_size: tuple = (300, 225, 3)

        x = resize(image, new_size, order=1, preserve_range=True)

        tensor: torch.tensor = torch.from_numpy(x)
        tensor = tensor.permute(2, 0, 1)

        y_aux = self.__labels[item]

        label = [0]*self.__n_clases
        label[y_aux] = 1

        label = torch.from_numpy(np.array(label))

        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

        if self.transform:
            tensor = self.transform(tensor)

        sample = {'x': tensor, 'y': label}

        return sample

    def get_labels(self):
        return self.__labels

    def __len__(self):
        return len(self.__labels)


