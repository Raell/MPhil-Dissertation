import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

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
        val_split=0.2,
        test_split=0.2,
        input_shape=(224, 224),
        target_labels=0.1,
        val_from_labelled=False
    ):
        self.val_from_labelled = val_from_labelled

        src_train, src_val = self.__prepare_data(
            source_domain,
            input_shape,
            True,
            val_split,
            test_split,
            target_labels
        )

        tar_train_label, tar_train_nlabel, tar_val, test = self.__prepare_data(
            target_domain,
            input_shape,
            False,
            val_split,
            test_split,
            target_labels
        )

        if len(tar_val) <= 10:
            val = torch.utils.data.ConcatDataset([src_val, tar_val])
        else:
            val = tar_val

        train_label = torch.utils.data.ConcatDataset([src_train, tar_train_label])
        self.train_label_loader = DataLoader(dataset=train_label, batch_size=64, shuffle=True, num_workers=0)
        if target_labels < 1:
            self.train_nlabel_loader = DataLoader(dataset=tar_train_nlabel, batch_size=64, shuffle=True, num_workers=0)
        else:
            self.train_nlabel_loader = None

        # self.train_src_loader = DataLoader(dataset=src_train, batch_size=32, shuffle=True, num_workers=0)
        #
        # if target_labels > 0:
        #     self.train_tar_label_loader = DataLoader(dataset=tar_train_label, batch_size=32, shuffle=True, num_workers=0)
        # else:
        #     self.train_tar_label_loader = None
        #
        # if target_labels < 1:
        #     self.train_tar_nlabel_loader = DataLoader(dataset=tar_train_nlabel, batch_size=32, shuffle=True, num_workers=0)
        # else:
        #     self.train_tar_nlabel_loader = None

        self.val_loader = DataLoader(dataset=val, batch_size=64, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(dataset=test, batch_size=64, shuffle=True, num_workers=0)

    def __prepare_data(self, folder, input_shape, src=True, val_split=0, test_split=0, target_labels=0.1):

        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor()
        ])

        label = 0 if src else 1

        data = torchvision.datasets.ImageFolder(folder, transform=transform)
        data.target_transform = lambda id: torch.Tensor((label, id))

        self.classes = data.classes

        if src:

            train, val = torch.utils.data.random_split(
                data,
                [round(len(data) * (1 - val_split) - 1e-5), round(len(data) * val_split + 1e-5)]
            )

            return train, val

        else:
            data, test = torch.utils.data.random_split(
                data,
                [round(len(data) * (1 - test_split) - 1e-5), round(len(data) * test_split + 1e-5)]
            )

            if self.val_from_labelled:
                train, train_nlabel = torch.utils.data.random_split(
                    data,
                    [round(len(data) * target_labels - 1e-5), round(len(data) * (1 - target_labels) + 1e-5)]
                )

                train_label, val = torch.utils.data.random_split(
                    train,
                    [round(len(train) * (1 - val_split) - 1e-5), round(len(train) * val_split + 1e-5)]
                )
            else:
                train, val = torch.utils.data.random_split(
                    data,
                    [round(len(data) * (1 - val_split) - 1e-5), round(len(data) * val_split + 1e-5)]
                )

                train_label, train_nlabel = torch.utils.data.random_split(
                    train,
                    [round(len(train) * target_labels - 1e-5), round(len(train) * (1 - target_labels) + 1e-5)]
                )

            return train_label, Dataset(train_nlabel, False), val, test

    def train_data(self):
        # return self.train_src_loader, self.train_tar_label_loader, self.train_tar_nlabel_loader
        return self.train_label_loader, self.train_nlabel_loader

    def val_data(self):
        return self.val_loader

    def test_data(self):
        return self.test_loader
