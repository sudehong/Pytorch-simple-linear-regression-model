import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np


alpha=-7.4; beta=-9.0
amats = []
pienergys = []
noccs = []
emos = []
hamats =[]
no = []
with open('C:\\Users\\SU DEHONG\\Desktop\\study\\JMedChem.34.786-smiles.txt','rt') as f1:
  for l1 in f1.readlines():
    l2 = l1.split()
    smi = l2[2]
    m1 = Chem.MolFromSmiles(smi)
    amat = rdmolops.GetAdjacencyMatrix(m1, useBO=False)
    nao = amat.shape[0]; nocc = nao // 2
    hmat = np.identity(nao) * alpha + amat * beta
    emo,cmo = np.linalg.eigh(hmat)
    
    pienergy = sum(emo[0:nocc]) * 2.
    emos.append(emo)
    amats.append(amat)
    hamats.append(hmat)
    pienergys.append([pienergy])
    noccs.append(nocc)
    no.append(nao)

x = np.array(amats,dtype=object)
# #max_length_index
index = max(enumerate(x), key=lambda sub: len(sub[1]))[0]
max_length = x[index].shape[0]
#pad
k = []
for i in x:
  s_len = len(i)
  pad_len = max_length - s_len
  ii1 = np.pad(i,((0,pad_len),(0,pad_len)),'constant')
  k.append(ii1)
print(k)
# # # print(np.shape(x))
# print(np.shape(k))

b = []
for j in range(len(x)):
  b.append(np.identity(max_length))
# print(b)
# print(np.shape(b))
#去掉多余对角的 1

for ii in range(0,len(b)):
  zz = np.zeros((28, 28))
  for j in range(no[ii],28):
    zz[j][j] = 1
  b[ii] = b[ii] - zz

# print(b)
#对称矩阵C
c = []
for ii in range(0,len(b)):
  z1 = np.zeros((28, 28))
  for j in range(no[ii],28):
    z1[j][j] = 777
  c.append(z1)

# print(emos)
#屏蔽矩阵
d = []
for j1 in range(0,len(b)):
  z2 = np.zeros(28)
  for j2 in range(0,noccs[j1]):
    z2[j2] = 1
  d.append(z2)

# print(d)


# h = np.array(k)*(-1.0)+np.array(b)*(-7.4)+np.array(c)
# e,v = np.linalg.eigh(h)
#
# ee = np.multiply(e,np.array(d))
# print(e)
# print(pienergys)
# print(np.sum(ee,axis=1,keepdims=True)*2)

# k1 = torch.from_numpy(np.array(k))
# b1 = torch.from_numpy(np.array(b))
# c1 = torch.from_numpy(np.array(c))
# d1 = torch.from_numpy(np.array(d))
# #
# hh = k1*(-1.0)+b1*(-7.4)+c1
# e1,v1 = torch.linalg.eigh(hh)
# ee1 = e1*d1
# # print(ee1)
# sum = torch.sum(ee1,dim=1,keepdim=True)*2
# print(sum)
# print(sum.shape)

# #
xx = [k,b,c]
X = torch.tensor(np.array(xx),dtype=torch.float32)
d1 = torch.from_numpy(np.array(d))

class MyModel(torch.nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.w = torch.nn.Parameter(torch.rand(2,1, dtype=torch.float32))

  def forward(self,xx):
    hh = xx[0]*self.w[0]+xx[1]*self.w[1]+xx[2]
    e1,v1 = torch.linalg.eigh(hh)
    ee1 = e1*d1
    sum1 = torch.sum(ee1, dim=1,keepdim=True,dtype=torch.float32) * 2
    return sum1
#
num = 5000
lr = 0.5
criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()
model = MyModel()
# optimizer = torch.optim.SGD(model.parameters(),lr)
optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=0)

y = torch.tensor(pienergys,dtype=torch.float32)
w  = torch.rand(2,1, requires_grad =True, dtype=torch.float32)
# print(y.shape);raise  RuntimeError

for i  in range(num):
  optimizer.zero_grad()
  y_pred = model(X)
  # print(y,y_pred);raise RuntimeError
  loss = criterion(y,y_pred)
  loss.backward()
  print(f'iter{i},loss {loss}')
  optimizer.step()


print(f'final parameter: {model.w}')