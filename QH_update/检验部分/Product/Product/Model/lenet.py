# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:01:06 2019

@author: GJW
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch import optim 
#from sklearn.preprocessing import StandardScaler
import xarray as xr
import pickle



class LeNet(nn.Module):
 
	def __init__(self):
		#Net继承nn.Module类，这里初始化调用Module中的一些方法和属性
		nn.Module.__init__(self)
		# BATCH_SIZE = 50
		# LR = 0.001
		#定义特征工程网络层，用于从输入数据中进行抽象提取特征
		#bn = torch.nn.BatchNorm2d(512)
		self.feature_engineering = nn.Sequential(
			nn.Conv2d(in_channels=3,
					  out_channels=6,
					  kernel_size=5,
					  padding=2),
			
			nn.BatchNorm2d(6),
			nn.ReLU(),
 
			#kernel_size=2, stride=2，正好可以将图片长宽尺寸缩小为原来的一半

 
			nn.Conv2d(in_channels=6,
					  out_channels=8,
					  kernel_size=5,
					  padding=2),
			
			nn.BatchNorm2d(8),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=8,
					  out_channels=12,
					  kernel_size=5,
					  padding=2),
			
			nn.BatchNorm2d(12),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=12,
					  out_channels=6,
					  kernel_size=5,
					  padding=2),
			
			nn.BatchNorm2d(6),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=6,
					  out_channels=2,
					  kernel_size=5,
					  padding=2),
			#nn.BatchNorm2d(2),
			#nn.ReLU()
		)
 

 
 
	def forward(self, x):
		#在Net中改写nn.Module中的forward方法。
		#这里定义的forward不是调用，我们可以理解成数据流的方向，给net输入数据inpput会按照forward提示的流程进行处理和操作并输出数据
		x = x.float()
		x = self.feature_engineering(x)
		print(11111111111111111111)
		print(x.shape)
		print(11111111111111111111)
		# x = x.view(-1, 16*801*1381)
		# x = self.classifier(x)
		return x
if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# 加载标签
	datalabel = np.load('labeldata.npy', allow_pickle='True')
	nt = torch.from_numpy(datalabel)
	print(nt.dtype)
	# 加载输入
	datainput = np.load('inputdata.npy', allow_pickle='True')
	uvg = torch.from_numpy(datainput)
	print(uvg.dtype)


	num_epochs = 15
	batch_size = 1
	torch.manual_seed(1)
	torch_dataset = Data.TensorDataset(uvg, nt)
	#####################
	loader = Data.DataLoader(
		dataset=torch_dataset,		# torch TensorDataset format
		batch_size=batch_size,		# mini batch size
		shuffle=True,				# 要不要打乱数据 (打乱比较好)
		num_workers=0			   # 多线程来读数据
	) 
	net = LeNet().to(device)
	#df = pd.read_csv('RPE_2019123113_.csv', sep='\t')

	# print(nt.shape)
	# outputs = net(nt)

	mse = nn.MSELoss()
	optimizer = optim.SGD(params = net.parameters(),
						  lr = 0.01)

	epochs = 50
	 
	average_loss_series = []

	total_step = len(loader)
	for epochs in range(num_epochs):
		for i, (inputs, labels) in enumerate(loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = net(inputs.float())
			print(outputs.dtype)
			loss =mse(outputs, labels.float())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (i+1) % 1 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					   .format(epochs+1, num_epochs, i+1, total_step, loss.item()))
			'''
			#梯度清零
			optimizer.zero_grad()
			#forward+backward
			outputs = net(inputs)
			print(outputs.dtype)
			labels = labels.float()
			# print("11111111111111111111111111111111111111")
			#还原预测的outputs
			#outputs = ss.inverse_transform(outputs.detach().numpy())
			#对比预测结果和labels，计算loss
			#　outputs = torch.from_numpy(outputs)
			#　print(outputs.shape)
			
			#反向传播
			loss.backward()
			#更新参数
			optimizer.step()
			# running_loss += loss.item()
			'''
	 


	print(outputs)
	print(loss)
	torch.save(net, 'net.pkl') 
