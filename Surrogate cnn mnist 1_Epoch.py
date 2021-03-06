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
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools


batch_size = 128
data_path = '/data/mnist'
subset = 10

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Data transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
#print({len(mnist_train)})  # 60000
utils.data_subset(mnist_train, subset)
utils.data_subset(mnist_test, subset)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# Define the Network: 12C5-MP2-64C5-MP2-1024FC10
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

net = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    # snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                    # nn.Linear(512, 128),
                    # snn.Leaky(beta=beta, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                    # nn.Linear(128, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

# tmp = torch.randn(128, 1, 28, 28)
# out = net(tmp)
# print('net out:', out.shape) # flatten:[128, 1024]--> 10

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

#The losses are accumulated over time steps to give the final loss.
loss_fn = SF.ce_rate_loss()
#train_loss_val = loss_fn(spk_rec, targets)

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

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

        # print(SF.accuracy_rate(spk_rec, targets))
        # print(acc)      # acc:  13,  23,  41,  49,  58, 64, 73
        # print(total)    # total:128, 256, 384, 512, 640,768,896

    return acc/total

train_acc = batch_accuracy(train_loader, net, num_steps)
print(f"The total accuracy on the train set is: {train_acc * 100:.2f}%")

# BPTT
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 1
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

with torch.autograd.profiler.profile(use_cpu=True, record_shapes=True, profile_memory=False) as prof:

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

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
