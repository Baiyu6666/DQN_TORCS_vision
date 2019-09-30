import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math
#假设y服从伯努利分布，P(x|y=0)和P(x|y=1)服从二维高斯分布
#解决中文显示问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]#黑体显示图中中文
mpl.rcParams["axes.unicode_minus"] = False#解决负号显示异常问题
#设定正负样本数分别为100，50
neg_num,pos_num=50,100
#生成样本数据
y1_x1_mu,y1_x1_sigma = 5,1
y1_x1 = np.random.normal(y1_x1_mu,y1_x1_sigma,pos_num)#y=1时的x1
y1_x2_mu,y1_x2_sigma = 8,1
y1_x2 = np.random.normal(y1_x2_mu,y1_x2_sigma,pos_num)#y=1时的x2
y0_x1_mu,y0_x1_sigma = 2,0.5
y0_x1 = np.random.normal(y0_x1_mu,y0_x1_sigma,neg_num)#y=0时的x1
y0_x2_mu,y0_x2_sigma = 3,0.2
y0_x2 = np.random.normal(y0_x2_mu,y0_x2_sigma,neg_num)#y=0时的x2
#整合训练集 train_set第二维上分别为特征x1，特征x2和标签y
train_set = np.zeros((pos_num+neg_num,3))
for i in range(0,neg_num):
    train_set[i,0] = y0_x1[i]
    train_set[i,1] = y0_x2[i]
    train_set[i,2] = 0
for i in range(neg_num,(neg_num+pos_num)):
    train_set[i,0] = y1_x1[i-neg_num]#注意y1_x1的下标~~
    train_set[i,1] = y1_x2[i-neg_num]
    train_set[i,2] = 1
#计算参数phi
phi = pos_num/(pos_num+neg_num)
#计算参数mu0
mu0 = np.zeros((2,1))
for i in range(0,neg_num):
    mu0[0] += train_set[i,0]
    mu0[1] += train_set[i,1]
mu0[0] /= neg_num
mu0[1] /= neg_num
#计算参数mu1
mu1 = np.zeros((2,1))
for i in range(neg_num,neg_num+pos_num):
    mu1[0] += train_set[i,0]
    mu1[1] += train_set[i,1]
mu1[0] /= pos_num
mu1[1] /= pos_num
#计算参数C
C = np.zeros((2,2))
for i in range(0,neg_num+pos_num):
    if train_set[i,2] == 0:
        C[0,0] += (train_set[i,0]-mu0[0])**2
        C[0,1] += (train_set[i,0]-mu0[0])*(train_set[i,1]-mu0[1])
        C[1,0] += (train_set[i,1]-mu0[1])*(train_set[i,0]-mu0[0])
        C[1,1] += (train_set[i,1]-mu0[1])**2
    if train_set[i,2] == 1:
        C[0,0] += (train_set[i,0]-mu1[0])**2
        C[0,1] += (train_set[i,0]-mu1[0])*(train_set[i,1]-mu1[1])
        C[1,0] += (train_set[i,1]-mu1[1])*(train_set[i,0]-mu1[0])
        C[1,1] += (train_set[i,1]-mu1[1])**2
C[0,0] /= (neg_num+pos_num)
C[0,1] /= (neg_num+pos_num)
C[1,0] /= (neg_num+pos_num)
C[1,1] /= (neg_num+pos_num)
det_C = C[0,0]*C[1,1]-C[0,1]*C[1,0]#C的行列式
C_inverse = np.zeros((2,2))#C的逆矩阵
C_inverse[0,0] = -C[1,1]/(C[0,1]**2-C[0,0]*C[1,1])
C_inverse[0,1] = C[0,1]/(C[0,1]**2-C[0,0]*C[1,1])
C_inverse[1,0] = C[0,1]/(C[0,1]**2-C[0,0]*C[1,1])
C_inverse[1,1] = -C[0,0]/(C[0,1]**2-C[0,0]*C[1,1])
#预测函数
def predict(inX1,inX2):
    const_parameter = 1/(2*np.pi*np.sqrt(det_C))#二项高斯分布的常系数
    x_y0_temp1 = (inX1-mu0[0])*C_inverse[0,0]+(inX2-mu0[1])*C_inverse[1,0]#(x-mu0)^T*C_inverse的中间变量
    x_y0_temp2 = (inX1-mu0[0])*C_inverse[0,1]+(inX2-mu0[1])*C_inverse[1,1]#(x-mu0)^T*C_inverse的中间变量
    x_y0_index = x_y0_temp1*(inX1-mu0[0])+x_y0_temp2*(inX2-mu0[1])
    Pro_x_y0 = const_parameter*(math.pow(math.e,(-0.5*x_y0_index)))#P(x|y=0)
    x_y1_temp1 = (inX1-mu1[0])*C_inverse[0,0]+(inX2-mu1[1])*C_inverse[1,0]#(x-mu1)^T*C_inverse的中间变量
    x_y1_temp2 = (inX1-mu1[0])*C_inverse[0,1]+(inX2-mu1[1])*C_inverse[1,1]#(x-mu1)^T*C_inverse的中间变量
    x_y1_index = x_y0_temp1*(inX1-mu1[0])+x_y0_temp2*(inX2-mu1[1])
    Pro_x_y1 = const_parameter*(math.pow(math.e,(-0.5*x_y1_index)))#P(x|y=1)
    Pro_x = Pro_x_y0*(1-phi)+Pro_x_y1*phi#P(x)
    Pro_y1_x = Pro_x_y1*phi/Pro_x
    Pro_y0_x = Pro_x_y0*(1-phi)/Pro_x
    predict_num = 0
    if Pro_y1_x > Pro_y0_x:
        predict_num = 1
    return predict_num
#画出分类图，直观地显示分类边界
xx,yy = np.meshgrid(np.arange(-1,12,0.05),np.arange(-1,12,0.05))
pre_data = np.c_[xx.ravel(),yy.ravel()]
pre_data_axis1= pre_data.shape[0]
z = np.zeros((pre_data_axis1,1))
for i in range(0,pre_data_axis1):
    z[i] = predict(pre_data[i,0],pre_data[i,1])
z = z.reshape(xx.shape)
print(z)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
plt.scatter(train_set[0:neg_num,0],train_set[0:neg_num,1],c=train_set[0:neg_num,2],edgecolors='k',label='负样本')
plt.scatter(train_set[neg_num+1:neg_num+pos_num-1,0],train_set[neg_num+1:neg_num+pos_num-1,1],c=train_set[neg_num+1:neg_num+pos_num-1,2],edgecolors='k',marker='x',label='正样本')
plt.xlabel('特征x1')
plt.ylabel('特征x2')
plt.legend(loc='best',title='0-1分布的样本')
plt.title('二维特征，0-1分布样本的高斯判别分类')
plt.savefig('GDA_classfication.png',format='png')
print('end')
---------------------
作者：ShiZhanfei
来源：CSDN
原文：https://blog.csdn.net/ShiZhanfei/article/details/84928736
版权声明：本文为博主原创文章，转载请附上博文链接！