import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    # Basic dataset with labels optional
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

        self.domains = len(source_domain) + len(target_domain)

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

    def __prepare_data__(self, folders, input_shape, src=True):
        # Loads data from folders and prepares dataset
        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor()
        ])

        data_list = []

        for i, folder in enumerate(folders):
            label = i if src else self.domains - 1
            data = torchvision.datasets.ImageFolder(folder, transform=transform)
            data.target_transform = lambda id: torch.Tensor((label, id)).long()
            data_list.append(data)

            self.classes = data.classes

        dataset = torch.utils.data.ConcatDataset(data_list)

        return dataset

    def get_datasets(self, test_split=0.2, target_labels=0):
        # Returns datasets for training and testing

        # Prepares source domain data
        src_train, src_test = torch.utils.data.random_split(
            self.src_data,
            [
                round(len(self.src_data) * (1 - test_split) - 1e-5),
                round(len(self.src_data) * test_split + 1e-5)
            ],
            generator=None if self.seed is None else torch.Generator().manual_seed(self.seed)
        )

        # Prepares target domain data
        tar_data = self.tar_data

        tar_nlabel, tar_label = torch.utils.data.random_split(
            tar_data,
            [
                round(len(tar_data) * (1 - target_labels) - 1e-5),
                round(len(tar_data) * target_labels + 1e-5)
            ],
            generator=None if self.seed is None else torch.Generator().manual_seed(self.seed)
        )

        return src_train, src_test, tar_label, Dataset(tar_nlabel, labels=False), tar_nlabel
