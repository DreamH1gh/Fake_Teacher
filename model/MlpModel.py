import torch.nn as nn
import torch.nn.functional as F

class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        self.hidden1=nn.Linear(in_features=1536,out_features=50,bias=True)
        self.hidden2=nn.Linear(50,50)
        self.hidden3=nn.Linear(50,50)
        self.predict=nn.Linear(50,1)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=F.relu(self.hidden3(x))
        output=self.predict(x)
        return output[:,0]

class MLPregressionRel(nn.Module):
    def __init__(self):
        super(MLPregressionRel,self).__init__()
        self.hidden1=nn.Linear(in_features=1537,out_features=50,bias=True)
        self.hidden2=nn.Linear(50,50)
        self.predict=nn.Linear(50,1)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        output=self.predict(x)
        return output[:,0]
