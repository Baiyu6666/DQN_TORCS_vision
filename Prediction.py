import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
import torch
from torch.nn import BCEWithLogitsLoss
import time
import visdom

#import matplotlib.pyplot as plt

#vis = visdom.Visdom(env='Logistic Regression')

LR = 0.5
maxCycle = 500
testFre = 25


def preprocess(df):
	#丢弃一些数据缺失较多的指标、日期和地点、RISK_MM
	df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date','RISK_MM'],axis=1)
	df = df.dropna()
	#去掉一些异常值
	z = np.abs(stats.zscore(df._get_numeric_data()))
	df = df[(z < 3).all(axis=1)]
	#将数据中的yes，no换为数字
	df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
	df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)
	#将部分非数值的变量分类
	class_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
	df = pd.get_dummies(df, columns=class_columns)
	#将所有数据归一到0~1区间
	scaler = preprocessing.MinMaxScaler()
	scaler.fit(df)
	df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

	X = df.loc[:, df.columns != 'RainTomorrow']
	y = df[['RainTomorrow']]
	return X, y

def GDA(y, X):
	#根据理论计算出多元高斯分布的各个参数
	theta1 = y.mean()
	theta0 = 1-y.mean()
	X_1 = X[(y == 1).squeeze()]
	X_0 = X[(y == 0).squeeze()]
	mu1 = X_1.mean(0)
	mu0 = X_0.mean(0)
	sigma = torch.zeros(X.shape[1], X.shape[1])
	for j in range(len(X_1)):
		delta = (X_1[j] - mu1).reshape(-1, 1)
		sigma += torch.mm(delta, delta.t())
	for j in range(len(X_0)):
		delta = (X_0[j] - mu0).reshape(-1, 1)
		sigma += torch.mm(delta, delta.t())
	sigma /= len(y)
	lnx1x0 = torch.log(theta1/theta0)
	return theta1.numpy(), theta0.numpy(), mu1.numpy(), mu0.numpy(), sigma.numpy(), lnx1x0

def df2tensor(df):
	return torch.Tensor(np.array(df))

def test(w, b, X_test, y_test):
	#给定一组参数，在测试集内测试模型表现，返回正确率
	y_pred = torch.sigmoid(torch.mm(X_test, w.detach()) + b.detach())
	mask = y_pred.ge(0.5).float()
	acc = int((mask == y_test).sum()) / y_test.shape[0]
	return acc


#加载数据并处理
data = pd.read_csv('weatherAUS.csv')
X, y = preprocess(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_test, y_train, y_test = df2tensor(X_train), df2tensor(X_test), df2tensor(y_train), df2tensor(y_test)

#参数初始化
w = torch.randn(X.shape[1], 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=LR)
loss_f = BCEWithLogitsLoss()

#使用GPU加速
# if torch.cuda.is_available():
# 	X_train = X_train.cuda()
# 	X_test = X_test.cuda()
# 	y_train = y_train.cuda()
# 	y_test = y_test.cuda()
# 	w = w.cuda()
# 	b = b.cuda()

##不使用先验信息的逻辑回归
t0 = time.time()
print('Training without prior starts!')
for i in range(maxCycle):
	#计算梯度并进行梯度下降
	y_pred = torch.mm(X_train, w) + b
	loss = loss_f(y_pred, y_train)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	#vis.line(X=torch.Tensor([i]), Y=torch.Tensor([loss]), win='Loss', update='append', opts={'title': 'Loss'})

	#间隔一定时间测试模型的表现
	if (i + 1) % testFre == 0:
		TRacc = test(w, b, X_train, y_train)
		TEacc = test(w, b, X_test, y_test)
		print('epoch: {}, Loss: {:.5f}, TrainAcc: {:.5f}, TestAcc: {:.5f}'.format(i + 1, loss.data.float(), TRacc, TEacc))
		# vis.line(X=torch.Tensor([i]), Y=torch.Tensor([TRacc, TEacc]).reshape(1, 2), win='Acc',
		# 		 update='append' , opts={'title': 'Acc', 'legend': ['TrainAcc', 'TestAcc']})
print('Time taken :', time.time() - t0)
print('Without prior| TrainAcc: {:.5f}, TestAcc: {:.5f}'.format(TRacc, TEacc))


##使用先验信息的逻辑回归(采用生成模型)
t0 = time.time()
print('\n\nTraining with prior starts!')
#利用数据计算生成模型参数
theta1, theta0, mu1, mu0, sigma, ln = GDA(y_train, X_train)
sigma_l = np.linalg.inv(sigma)
w = np.dot(sigma_l, (mu1 - mu0))
b = -0.5*np.dot(np.dot(mu1.T, sigma_l), mu1) + 0.5*np.dot(np.dot(mu0.T, sigma_l), mu0) + ln
w = torch.Tensor(w).reshape(-1, 1)

#测试模型性能
TEacc = test(w, b, X_test, y_test)
TRacc = test(w, b, X_train, y_train)
print('Time taken :', time.time() - t0)
print('With prior| TrainAcc: {:.5f}, TestAcc: {:.5f}'.format(TRacc, TEacc))