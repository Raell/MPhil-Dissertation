import torch
import torch.nn as nn


def build_encoder():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34')
    modules = list(model.children())[:-1] + [torch.nn.Flatten()]
    model = nn.Sequential(*modules)
    return model


def build_classifier(input_shape, classes, domains):

    classifier = nn.Sequential(
        nn.Linear(input_shape, 1280),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(1280, 1280),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(1280, domains * classes),
        nn.LogSoftmax(dim=1)
    )
    return classifier


def build_discriminator(input_shape, domains):
    discriminator = nn.Sequential(
        nn.Linear(input_shape, 1280),
        nn.ReLU(True),
        nn.Linear(1280, 1280),
        nn.ReLU(True),
        nn.Linear(1280, domains)
    )
    return discriminator