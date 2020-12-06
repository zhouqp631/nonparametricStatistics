import torch
import torch.nn as nn
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
#%%
# 定义全局变量
n_epochs = 10     # epoch 的数目
batch_size = 20  # 决定每次读取多少图片

# 定义训练集个测试集，如果找不到数据，就下载
train_data = datasets.MNIST(root = './data', train = True, download = False, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = True, download = False, transform = transforms.ToTensor())
# 创建加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = len(test_data), num_workers = 0)



#%%
# model
model = nn.Sequential(nn.Linear(784,512),nn.Linear(512,10))
# loss
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# train
losses = []
for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
    X_batch = X_batch.flatten(start_dim=1)
    # Zero gradient buffers
    optimizer.zero_grad()

    # Pass data through the network
    y_hat = model(X_batch)

    # Calculate loss
    loss = criterion(y_hat, y_batch)

    # Backpropagate
    loss.backward()

    # Update weights
    optimizer.step()

    if batch_idx%20==0:
        losses.append(loss.data.item())


#%% 训练误差
plt.plot(losses);plt.show()

#%% 预测
model.eval()
idx_test = 1245
image_test = test_data.test_data[idx_test]
image_test_flatten = torch.tensor(image_test.reshape(1,-1),dtype=torch.float)
label_test = test_data.test_data[idx_test]

y_predict = model(image_test_flatten)
label_predict = torch.argmax(y_predict).item()

plt.imshow(image_test,cmap='gray')
plt.title(f'Predicted Label:{label_predict}',fontsize=20)
plt.show()


#%% 准确度
for data, target in test_loader:
    data = data.flatten(start_dim=1)
    output = model(data)
    pred = torch.argmax(output,dim=1)
    correct = pred.eq(target)
    accuracy = np.mean([1 if c else 0 for c in correct])
print(f'Accuracy:{accuracy*100:.2f}%')