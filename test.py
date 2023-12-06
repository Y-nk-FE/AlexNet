import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from net import AlexNet
import os


def Get_Image_Size(in_size):
    """
    获取图像尺寸，并且辨别是int还是tuple
    :param in_size:
    :return:
    """
    # get Input-Imagee size
    width, high = 224, 224
    if isinstance(in_size, int):
        width = in_size
        high = in_size
    elif isinstance(in_size, tuple):
        width = in_size[0]
        high = in_size[1]
    return width, high


def Data_Loading(root, img_size, batch):
    """
    加载数据
    :param root:数据地址
    :param img_size:需要的图像大小
    :param batch: batch-number
    :return:
    """
    width, high = Get_Image_Size(img_size)
    test_transform = transforms.Compose([transforms.Resize((width, high)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.ImageFolder(root=root, transform=test_transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=True, num_workers=0)
    return test_dataloader


def Test_Model(model, test_dataloader):
    # gpu/cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cup')
    model = model.to(device)
    # initialize parameter
    test_acc = 0.0
    test_num = 0
    test_correct = 0
    # 梯度值为零
    with torch.no_grad():
        for feature, label in test_dataloader:
            feature = feature.to(device)
            label = label.to(device)
            model.eval()
            output = model(feature)
            predict = torch.argmax(input=output, dim=1)
            test_correct += torch.sum(predict == label.data)
            test_num += feature.size(0)
            print(f"predict:{predict}---label:{label}, result:{'1' if predict == label else '0'}")
    test_acc = test_correct.double().item() / test_num
    print('test acc:', test_acc)


if __name__ == "__main__":
    model = AlexNet()
    root_path = os.getcwd()
    model.load_state_dict(torch.load(f=os.path.join(root_path, 'pth_save', 'AlexModel_V11.pth')))
    test_dataloader = Data_Loading(root=os.path.join(root_path, 'data_enhance', 'test'),
                                   img_size=227,
                                   batch=1)
    Test_Model(model=model, test_dataloader=test_dataloader)
