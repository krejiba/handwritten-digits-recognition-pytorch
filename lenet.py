import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh()  # Original network uses a scaled hyperbolic tangent
        )
        self.S2 = nn.AvgPool2d(kernel_size=2)   # Original network uses a more complex version of average pooling
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.S4 = nn.AvgPool2d(kernel_size=2)
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.F6 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )
        self.output = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        feature_extractor = nn.Sequential(self.C1, self.S2, self.C3, self.S4, self.C5)
        classifier = nn.Sequential(self.F6, self.output)
        y = feature_extractor(x)
        y = torch.flatten(y, 1)
        logits = classifier(y)
        probs = F.softmax(logits, dim=1)  # Original network uses RBF output
        return logits, probs
