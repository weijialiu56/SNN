import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

from lenet5 import Lenet5

import matplotlib.pyplot as plt

def main():

    batch_size = 128
    data_path = '/data/cifar10'
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    data, targets = next(iter(train_loader))
    print('data:', data.shape, 'targets:', targets.shape)

    model = Lenet5().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    train_loss_hist = []
    test_acc_hist = []

    for epoch in range(10):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            logits = model(data)   # logits:[batch_size, 10], targets:[batch_size]
            loss_val = loss(logits, targets)   # loss: tensor scaler, 长度为零的标量

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            train_loss_hist.append(loss_val.item())
        print(f"Epoch {epoch}, Train Loss: {loss_val.item():.2f}")
        # 这里的loss为最后一个batch的loss，不能代表整个epoch训练过程，acc更能体现性能
        # Loss是一个标量，我们用item()将它转化为numpy，打印出来

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)

                logits = model(data)  # [batch_size, 10]
                pred = logits.argmax(dim=1)  #Return the index of the maximum value of input tensor
                correct = torch.eq(pred, targets).float().sum().item()
                total_correct += correct
                total_num += data.shape[0]

            test_acc = total_correct / total_num
            test_acc_hist.append(test_acc)
            print(f"Epoch: {epoch}, Test Acc: {test_acc * 100: .2f}%\n")

    # Plot test accuracy
    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("Accuracy Curves")
    plt.legend(["Test Acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == '__main__':
    main()

