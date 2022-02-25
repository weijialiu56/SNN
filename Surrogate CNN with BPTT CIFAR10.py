import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np


batch_size = 64
data_path = '/data/cifar10'
# 60000 32*32, 50000 in training and 10000 in test
# 0: airplane 1: automobile 2: bird 3: cat 4: deer 5: dog 6: frog 7: horse 8: ship 9: truck
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# Define the Network: 12C5-MP2-64C5-MP2-1024FC10
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 12

net = nn.Sequential(nn.Conv2d(3, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*5*5, 10),
                    # snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                    # nn.Linear(512, 128),
                    # snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                    # nn.Linear(128, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

# tmp = torch.randn(128, 3, 32, 32)
# out = net(tmp)
# print('net out:', out.shape) # flatten:[128, 1600]--> 10

# Forward pass of num_steps
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)   # resets hidden states for all LIF neuron in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)
print(f"the shape of spk_rec is: {spk_rec.size()}")   # [50, batch_size, 10]

loss_fn = SF.ce_rate_loss()
acc = SF.accuracy_rate(spk_rec, targets)

# the accuracy on the entire DataLoader
def batch_accuracy(train_loader, net, num_steps):
    #net.train()
    total = 0
    acc = 0

    train_loader = iter(train_loader)
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, _ = forward_pass(net, num_steps, data)

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)  # acc 是一个常数
        total += spk_rec.size(1)

        # print(SF.accuracy_rate(spk_rec, targets))
        # print(acc)
        # print(total)

    return acc/total

train_acc = batch_accuracy(train_loader, net, num_steps)
print(f"The total accuracy on the train set is: {train_acc * 100:.2f}%")

# BPTT
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 10
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

for epoch in range(num_epochs):
    net.train()
    train_avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
                             num_steps=num_steps, time_var=False, device=device)

    train_loss_hist.append(train_avg_loss.item())
    print(f"Epoch {epoch}, Train Loss: {train_avg_loss.item():.2f}")

    train_acc = batch_accuracy(train_loader, net, num_steps)
    train_acc_hist.append(train_acc)
    print(f"Epoch {epoch}, Train Acc: {train_acc * 100:.2f}%")

    # Test set
    with torch.no_grad():
        net.eval()
        test_data, test_targets = next(iter(test_loader))
        test_data = test_data.to(device)
        test_targets = test_targets.to(device)

        test_spk_rec, test_mem_rec = forward_pass(net, num_steps, test_data)

        test_loss = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            test_loss = loss_fn(test_spk_rec, test_targets)
        test_loss_hist.append(test_loss.item())
        print(f"Epoch {epoch}, Test Loss: {test_loss.item():.2f}")

        test_acc = batch_accuracy(test_loader, net, num_steps)
        test_acc_hist.append(test_acc)
        print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")


# Plot test accuracy
fig = plt.figure(facecolor="w")
plt.plot(train_acc_hist)
plt.plot(test_acc_hist)
plt.title("Accuracy Curves")
plt.legend(["Train Acc", "Test Acc"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Spike counter of idx-Sample
spk_rec, mem_rec = forward_pass(net, num_steps, data)

from IPython.display import HTML

idx = 2

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
plt.rcParams['animation.ffmpeg_path'] = 'D:\\ffmpeg\\bin\\ffmpeg.exe'

anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
                        animate=True, interpolate=4)
# spk_rec[:, 2]: 3-th [50, 10]
HTML(anim.to_html5_video())
anim.save("spike_bar(Surrogate CNN with BPTT).mp4")

print(f"The target label is: {targets[idx]}")
