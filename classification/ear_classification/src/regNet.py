import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((5, 5))  # Allows to use input images of variable size
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(16 * 5 * 5, 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10)
        # )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 5),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(12, 26, 5),
            nn.BatchNorm2d(26),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((5, 5))  # Allows to use input images of variable size
        )
        self.classifier = nn.Sequential(
            nn.Linear(26 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Net_Sure(nn.Module):
    """
    Neural Network specialized on regression tasks.
    Introduces a sureness factor which indicates how strongly the net believes in its prediction.
    """
    def __init__(self):
        super(Net_Sure, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((5, 5))  # Allows to use input images of variable size
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_pred = self.classifier(x)[:, 0]
        x_sure = nn.functional.sigmoid(self.classifier(x)[:, 1])
        return (x_pred, x_sure)

class Net2_Sure(nn.Module):
    """
    Neural Network specialized on regression tasks.
    Introduces a sureness factor which indicates how strongly the net believes in its prediction.
    """
    def __init__(self):
        super(Net2_Sure, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 5),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(12, 26, 5),
            nn.BatchNorm2d(26),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((5, 5))  # Allows to use input images of variable size
        )
        self.classifier = nn.Sequential(
            nn.Linear(26 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_pred = self.classifier(x)[:, 0]
        x_sure = nn.functional.sigmoid(self.classifier(x)[:, 1])
        return (x_pred, x_sure)