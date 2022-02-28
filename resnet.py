import  torch
from    torch import  nn
from    torch.nn import functional as F

class ResBlk(nn.Module):
    # Resnet block

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)  # 使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] = > [b, ch_out, h, w]
            self.extra =nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                                      nn.BatchNorm2d(ch_out)
                                      )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        out = self.extra(x) + out
        out = F.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
                                   nn.BatchNorm2d(64)
                                   )
        # followed 4 blocks
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)    # 根据经验，max为512，但也可以设置成更大

        self.outlayer = nn.Linear(512, 10)   # full connection

    def forward(self, x):

        x = F.relu(self.conv1(x))   # activit
        # [b, 64, h, w] --> [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        #print('after conv:', x.shape)   # [2, 3, 32, 32] --> [2, 512, 2, 2]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        #print('after pool:', x.shape)     # [b, 512, h, w] => [b, 512, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():

    blk = ResBlk(64, 128, stride=4)    #加上stride，可以实现图片长和宽的减少
    x = torch.randn(2, 64, 32, 32)
    out = blk(x)
    print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet:',out.shape)

if __name__ == '__main__':
    main()









