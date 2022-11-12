import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F 
from sklearn import preprocessing
df = pd.read_csv("train.tsv",sep='\t')

df_modified = df.set_index('1')
n=0
k=0
a=[]
b=[]
a_train=[]
a_test=[]
b_test=[]
b_train=[]
name1 = df.iloc[n, 0]
name2 = df.iloc[n, 1]
landmarks = df.iloc[n, 1:]
landmarks = np.asarray(landmarks)

# Преобразование текста в понятное понятное пк
def text_to_seq(name1):
    char_counts = Counter(name1)
    char_counts = sorted(char_counts.items(), key = lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
   
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in name1])
    
    return sequence, char_to_idx, idx_to_char



def f1(n):
    for i in range(22000):
        n+=1
        name1 = df.iloc[n, 0]
        name2 = df.iloc[n, 1]
        name = df.iloc[n, 2]
        a_train.append(name1)
    return a_train
def f2(n):
    for i in range(22000):
        n+=1
        name = df.iloc[n, 2]
        b_train.append(name)
    return b_train
def f3(n):
    n=22000
    for i in range(6000):
        n+=1
        name1 = df.iloc[n, 0]
        a_test.append(name1)
    return a_test
def f4(n):
    n=22000
    for i in range(6000):
        n+=1 
        name = df.iloc[n, 2]
        b_test.append(name)
    return b_test

af,ad,ar,ak=f1(1), f2(1),f3(1),f4(1)
af1 = preprocessing.LabelEncoder()
af = af1.fit_transform(af)
ad1 = preprocessing.LabelEncoder()
ad = ad1.fit_transform(ad)

ad = torch.as_tensor(ad)
af=torch.as_tensor(af)
class TwoLayersNet(nn.Module):
    def __init__(self, nX, nH, nY):        
        super(TwoLayersNet, self).__init__()     
         
        self.fc1 = nn.Linear(nX, nH)             
        self.fc2 = nn.Linear(nH, nY)             
          
    def forward(self, x):                        
        x = self.fc1(x)                          
        x = nn.Sigmoid()(x)                      
        x = self.fc2(x)                         
        x = nn.Sigmoid()(x)                     
        return x
          
model = TwoLayersNet(2, 5, 1)                    
                   
 
loss= nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),          
                            lr=0.5, momentum=0.8)        
def fit(model, X,Y, batch_size=22000, train=True):    
      model.train(train)
      sumL, sumA, numB = 0, 0, int( len(X)/batch_size )  
       
      for i in range(0, numB*batch_size, batch_size):          
        xb =   ad                    
        yb =  af                   

        y = model(xb)                                  
        L = loss(y, yb)                               

        if train:                                 
            optimizer.zero_grad()                         
            L.backward()                                        
            optimizer.step()                             

        sumL += L.item()                                
        sumA += (y.round() == yb).float().mean()         
         
        return sumL/numB,  sumA/numB                         
                                                         

 
epochs = 1000                                           
for epoch in range(epochs):                              
    L,A = fit(model, af, ad)                               
     
    if epoch % 100 == 0 or epoch == epochs-1:                 
        print(f'epoch: {epoch:5d} loss: {L:.4f} accuracy: {A:.4f}' )   