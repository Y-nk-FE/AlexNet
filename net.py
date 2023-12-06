import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 激活函数
        self.ReLU = nn.ReLU()
        # 1.227,227,3
        # 2.227,227,3 -> 55,55,96
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4)
        # 3.55,55,96 -> 27,27,96
        self.s2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        # 4.27,27,96 -> 27,27,256
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2)
        # 5.27,27,256 -> 13,13,256
        self.s4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        # 6.13,13,256 -> 13,13,384
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)
        # 7.13,13,384 -> 13,13,384
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)
        # 7.13,13,384 -> 13,13,256
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        # 8.13,13,384 -> 6,6,256
        self.s8 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        # 9.6,6,256 -> 9216
        self.flatten = nn.Flatten()
        # 10.9216 -> 4096
        self.f1 = nn.Linear(9216, 4096)
        # 11.4096 -> 4096
        self.f2 = nn.Linear(4096, 4096)
        # 12.4096 -> 2
        self.f3 = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.s8(x)
        # -----------------------
        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = F.dropout(input=x, p=0.5)
        x = self.ReLU(self.f2(x))
        x = F.dropout(input=x, p=0.5)
        x = self.f3(x)
        return x


if __name__ == "__main__":
    # 测试main函数，用来查看网络结构
    # 输入数据格式(深度，宽度，高度)-(3,227,227)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    check_model = AlexNet().to(device)
    print(summary(check_model, input_size=(3, 227, 227)))

