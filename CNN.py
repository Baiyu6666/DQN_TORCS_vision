import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class CNN(nn.Module):
    def __init__(self, n_kernel, size_filter, stride, linear_n, IMAGE_NUM, N_STATE_LOW, N_ACTIONS):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape
            nn.Conv2d(in_channels=IMAGE_NUM,  # input height
                      out_channels=n_kernel[0],  # n_filter
                      kernel_size=size_filter[0],  # filter size
                      stride=stride[0],  # filter step
                      padding=2  # con2d出来的图片大小不变
                      ),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)  # 2x2采样

        )
        self.conv2 = nn.Sequential(nn.Conv2d(n_kernel[0], n_kernel[1], size_filter[1], stride[1], 2),
                                   nn.ReLU()
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(n_kernel[1], n_kernel[2], size_filter[2], stride[2], 2),
                                   nn.ReLU()
                                   )
        self.adv1 = nn.Sequential(nn.Linear(linear_n[0] + N_STATE_LOW, linear_n[1]),
                                  nn.ReLU())
        self.adv2 = nn.Sequential(nn.Linear(linear_n[1], linear_n[2]),
                   nn.ReLU(),
                   nn.Linear(linear_n[2], N_ACTIONS))

        self.val1 = nn.Sequential(nn.Linear(linear_n[0] + N_STATE_LOW, linear_n[1]),
                                  nn.ReLU())
        self.val2 = nn.Sequential(nn.Linear(linear_n[1], linear_n[2]),
                   nn.ReLU(),
                   nn.Linear(linear_n[2], 1))

        self.apply(weights_init)


        print('Model initialized')

    def forward(self, x_image, x_low):
        x = self.conv1(x_image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_low), 1)

        adv = self.adv1(x)
        adv = self.adv2(adv)

        val = self.val1(x)
        val = self.val2(val)
        return val + adv - adv.mean()
