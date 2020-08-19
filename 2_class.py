#LIBRARIES
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

#Loading data
X_train=np.genfromtxt('eeg_data.csv',delimiter=',')

#Test-Train Split
y_train = X_train[1:,-1]
X_train=X_train[1:,1:-1]
y_train[y_train>1]=0
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=X_train[-1500:]
X_train=X_train[:-1500]
y_test=y_train[-1500:]
y_train=y_train[:-1500]

#Data to Tensors
X_train=torch.from_numpy(X_train)
y_train=torch.from_numpy(y_train)
X_test=torch.from_numpy(X_test)
y_test=torch.from_numpy(y_test)

#Model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(178,128)
        self.linear2=nn.Linear(128,32)
        self.linear3=nn.Linear(32,8)
        self.linear4=nn.Linear(8,2)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(p=0.5)
        
    def forward(self,input_feature):
        input_feature=input_feature.view(-1,1,178)
        output=self.linear1(input_feature)
        output=self.relu(output)
        output=self.dropout(output)
        output=self.linear2(output)
        output=self.relu(output)
        output=self.dropout(output)
        output=self.linear3(output)
        output=self.sigmoid(output)
        output=self.linear4(output)
        output=self.sigmoid(output)
        return output.view(-1,2)


#Preparing data and Model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion=nn.CrossEntropyLoss()

train_ds=TensorDataset(X_train,y_train)
test_ds=TensorDataset(X_test,y_test)
train_loader=DataLoader(train_ds, batch_size=64)
test_loader=DataLoader(test_ds, batch_size=32)

#Training Model
epochs=100
plot_values=[]
for epoch in range(epochs):
    loss=0
    for batch,y in train_loader:
        batch=batch.to(device)
        y=y.long()
        y=y.to(device)
        optimizer.zero_grad()
        output=model(batch.float())
        train_loss=criterion(output,y)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    scheduler.step()
    loss = loss / len(train_loader)
    if (epoch+1)%5==0:
        print(f'Epoch {epoch+1}/{epochs} : Training Error loss = {loss}')
    plot_values.append(loss)

#plotting result
plt.plot(np.arange(1,epochs+1),plot_values,label='Training error')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()

#Testing the model
loss=0
bs=0
acc=0
with torch.no_grad():
    for batch,y in test_loader:
        batch=batch.to(device)
        y=y.view(y.shape[0]).long()
        y=y.to(device)
        bs+=y.shape[0]
        output=model(batch.float())
        test_loss=criterion(output,y)
        loss+=test_loss
        _,preds=torch.max(output,dim=1)
        acc+=torch.sum(preds==y).item()
loss=loss/len(test_loader)
print(f'Loss on Test data = {loss}')
print(f'Accuracy on test data = {acc/bs}')
