import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        ## my code:
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=6, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )
        self.rnn = nn.GRU(input_size=256 * 4 * 43, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 86, 200),
            nn.Dropout(0.25),
            nn.Linear(200, num_classes),
            nn.Softmax(dim=1)
        )

        ## old code:
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        # self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## my code:
        x = x.to('cuda')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), x.size(2), -1)  # Flatten the feature maps for RNN
        x, _ = self.rnn(x)

        x = self.fc(x)
        return x

        ## old code:
        # x = self.conv1(x)
        # x = self.maxpool(x)
        # x = self.layer0(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #
        # return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
