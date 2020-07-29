import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


plt.ion()
class nueralnet(nn.Module):
    def __init__ (self,in_features, out_features):
        super(nueralnet, self).__init__()
        self.seq_model = nn.Sequential(nn.Linear(in_features, 256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, 256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, out_features))
    def forward(self, x):
        output = self.seq_model(x)
        return output
    
x = np.random.rand(10**5) * 360
inp = [[i,np.sin(i*(np.pi/180))] for i in x]
graphx = [i for i in range(0,361)]
    
trainloader = torch.utils.data.DataLoader(inp, batch_size=512, shuffle=True)

model = nueralnet(1,1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction="mean")

EPOCHS = 50
learning_rate = 0.01
loss = 0

# for i in tqdm(range(EPOCHS)):
#     if i%10==0:
#         print(loss)
for i in range(EPOCHS):
    print(i)
    graphx = np.array([i for i in range(0,361)])
    graphx = torch.tensor(graphx).to(device)
    graphx = graphx.reshape(-1,1)
    graphy = model(graphx.float())
    graphx = graphx.cpu()
    graphx = graphx.detach().numpy()
    graphy = graphy.cpu()
    graphy = graphy.detach().numpy()
    plt.plot(graphx,graphy)
    for num,sine in trainloader: 
        num, sine = num.to(device), sine.to(device)
        num = num.reshape(-1,1)
        sine = sine.reshape(-1,1)
        y_pred = model(num.float())
        loss = criterion(y_pred, sine.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


























    
        
        