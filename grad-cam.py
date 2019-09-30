import cv2
from CNN import CNN
from Utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

gradcam = False
feature = []
featureGrad = []

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

    def save_grad(self, input):
        featureGrad.append(input)

    def forward(self, x_image, x_low):
        # x = x_image
        # x.register_hook(self.save_grad)
        # feature.append(x)
        x = self.conv1(x_image)
        x = self.conv2(x)
        x = self.conv3(x)

        # for j in range(55):
        #     p = x.cpu().detach().numpy()[0][j]
        #     p = p / np.max(p)
        #     p = np.uint8(255 * p)
        #     p = cv2.applyColorMap(p, cv2.COLORMAP_JET)

            # cv2.imshow(str(j) + 'feature', p)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_low), 1)

        adv = self.adv1(x)
        adv = self.adv2(adv)

        val = self.val1(x)
        val = self.val2(val)
        Q = val + adv - adv.mean()
        q = torch.mean(Q)
        q.backward(retain_graph=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_kernel = (32, 64, 64)
size_filter = (8, 4, 3)
stride = (4, 2, 1)
linear_n = (7744, 128, 32)

IMAGE_NUM = 3
N_ACTIONS = 6
N_STATE_IMG = IMAGE_NUM*64*64
N_STATE_LOW = 5
memory_MAX = 5000
LEARN_FREQUENCY = 500
SAVE_FREQUENCY = 1000 #save model and reward list
BATCH_SIZE = 1
method = 'Data_generate'
model = CNN(n_kernel, size_filter, stride, linear_n, IMAGE_NUM, N_STATE_LOW, N_ACTIONS).to(device)
model.load_state_dict(torch.load
                              ('data/' + method + '/eval_net_CNN.pkl'))

memory = np.loadtxt('data/memory/cnn.csv',  delimiter=',')

img1 = cv2.imread('figure/cnn/310.png')
img2 = cv2.imread('figure/cnn/20.png')
img3 = cv2.imread('figure/cnn/353.png')
img4 = cv2.imread('figure/cnn/435.png')
step = [308, 353,351,355,430]
png_list = [img1, img3, img3,img3, img4]


for i, png in zip(step, png_list):
    b = memory[[i], :]
    b_s_img = torch.FloatTensor(b[:, :N_STATE_IMG]).reshape(BATCH_SIZE, IMAGE_NUM, 64, 64).to(device)
    b_s_low = torch.FloatTensor(b[:, N_STATE_IMG:N_STATE_IMG + N_STATE_LOW]).to(device)
    b_s_img.requires_grad = True
    model.zero_grad()
    model.forward(b_s_img, b_s_low)

    if gradcam:
        w = featureGrad[0][0].cpu().numpy().mean(1).mean(1)
        layer = feature[0][0].cpu().detach().numpy()
        feature.pop()
        featureGrad.pop()
        heat = np.zeros((layer.shape[1], layer.shape[1]))
        for j in range(w.shape[0]):
            heat += w[j]*layer[j]
    else:
        heat = np.abs(b_s_img.grad[0, 0].cpu().detach().numpy())\
               +np.abs(b_s_img.grad[0, 1].cpu().detach().numpy())\
                +np.abs(b_s_img.grad[0, 2].cpu().detach().numpy())


    img = b_s_img[0, 0, :, :].cpu().detach().numpy()
    # #heat[heat<0.04]=0
    # # plt.figure()
    # # plt.imshow(heat)
    # # plt.figure()
    # # plt.imshow(img,cmap=plt.cm.gray)
    # plt.figure(i)
    #
    # plt.imshow(img+cv2.resize(heat, (64, 64))*10)

    heat /= np.max(heat)
    # cv2.imshow(str(i) + 'heat_o', np.uint8(255 * heat))
    heat = cv2.resize(heat, (png.shape[1], png.shape[0]))
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    superimposed_img = heat * 0.3 + png
    superimposed_img /= np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)


    cv2.imshow(str(i), superimposed_img)
    cv2.imwrite('figure/cnn/str!val'+str(i)+'.png', superimposed_img)
plt.show()
cv2.waitKey(0)
