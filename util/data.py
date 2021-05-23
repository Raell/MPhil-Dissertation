import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=True):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        return x if self.labels else x[0]

    def __len__(self):
        return len(self.data)


class DataGenerator:
    def __init__(
        self,
        source_domain,
        target_domain,
        input_shape=(224, 224),
        seed=None
    ):

        self.seed = seed

        self.src_data = self.__prepare_data__(
            source_domain,
            input_shape,
            True
        )

        self.tar_data = self.__prepare_data__(
            target_domain,
            input_shape,
            False
        )

    def __prepare_data__(self, folder, input_shape, src=True, test_split=0):

        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor()
        ])

        label = 0 if src else 1

        data = torchvision.datasets.ImageFolder(folder, transform=transform)
        data.target_transform = lambda id: torch.Tensor((label, id)).long()

        self.classes = data.classes

        return data

    def get_TCV(self, test_split=0.2):
        src_train, src_test = torch.utils.data.random_split(
            self.src_data,
            [
                round(len(self.src_data) * (1 - test_split) - 1e-5),
                round(len(self.src_data) * test_split + 1e-5)
            ],
            generator=None if self.seed is None else torch.Generator().manual_seed(self.seed)
        )

        return src_train, src_test, Dataset(self.tar_data, labels=False)

    def get_datasets(self, target_labels, test_split=0.2):

        tar_train, tar_test = torch.utils.data.random_split(
            self.tar_data,
            [
                round(len(self.tar_data) * (1 - test_split) - 1e-5),
                round(len(self.tar_data) * test_split + 1e-5)
            ],
            generator=None if self.seed is None else torch.Generator().manual_seed(self.seed)
        )

        tar_label, train_nlabel = torch.utils.data.random_split(
            tar_train,
            [
                round(len(tar_train) * target_labels - 1e-5),
                round(len(tar_train) * (1 - target_labels) + 1e-5)
            ],
            generator=None if self.seed is None else torch.Generator().manual_seed(self.seed)
        )

        train_label = torch.utils.data.ConcatDataset([self.src_data, tar_label])

        return train_label, Dataset(train_nlabel, labels=False), tar_test
