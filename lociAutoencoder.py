import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv


# I generated this CSV to ensure the architecture, you might want to double check it
reader = csv.reader(open("/mnt/boost_fieldA/SorghumBioencoder/SorghumBioencoder/SorghumData/ConnectingNodes.csv", "rt", encoding = "ascii"), delimiter = ",")
x = list(reader)
result = np.array(x).astype("int")

reader = csv.reader(open("/mnt/boost_fieldA/SorghumBioencoder/SorghumBioencoder/SorghumData/pathway_connections.csv", "rt", encoding="ascii"), delimiter=",")
x = list(reader)
pathway = np.array(x).astype("int")

reader = csv.reader(open("/mnt/boost_fieldA/SorghumBioencoder/SorghumBioencoder/SNPData/TrainData.csv", "rt", encoding = "ascii"), delimiter = ",")
x = list(reader)
TrainData = np.array(x).astype("int")

reader = csv.reader(open("/mnt/boost_fieldA/SorghumBioencoder/SorghumBioencoder/SNPData/TestData.csv", "rt", encoding = "ascii"), delimiter = ",")
x = list(reader)
TestData = np.array(x).astype("int")

sparseMatrix = np.zeros((232302, 5686));

for i in range(0, 232301):
    sparseMatrix[i, result[i] - 1] = 1


# Final matrix to ensure the architecture, you might want to double check it
sparseMatrix = torch.from_numpy(sparseMatrix).float().t()

pathwayMatrix = np.zeros((5686,5686));

for i in range(0,len(pathwayMatrix)-1):
  pathwayMatrix[pathway[i,0] - 1, pathway[i,1] - 1] = 1

pathwayMatrix = torch.from_numpy(pathwayMatrix).float().t()


class Net(nn.Module):
  
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(232302, 5686)
    self.fc2 = nn.Linear(5686, 5686)
    self.fc3 = nn.Linear(5686, 256)
    self.fc4 = nn.Linear(256, 5686)
    self.fc5 = nn.Linear(5686, 5686)
    self.fc6 = nn.Linear(5686, 232302)

  def encode(self, x):
    x = x.view(-1, self.num_flat_features(x))
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    x = F.sigmoid(self.fc3(x))
    return x

  def decode(self, x):
    x = F.sigmoid(self.fc4(x))
    x = F.sigmoid(self.fc5(x))
    x = F.sigmoid(self.fc6(x))
    return x

  def forward(self, x):
    x = self.encode(x)
    x = self.decode(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1: ]# all dimensions except the batch dimension
    num_features = 1
    for s in size:
          num_features *= s
    return num_features

def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,weight_decay=2e-5)

criterion = nn.BCELoss()# criterion = nn.MSELoss()
net.zero_grad()
optimizer.zero_grad()# zero the gradient buffers

# to make initial weights as per the architecture
net.fc1.weight.data = torch.mul(net.fc1.weight.data, sparseMatrix);
net.fc6.weight.data = torch.mul(net.fc6.weight.data, sparseMatrix.t());
net.fc2.weight.data = torch.mul(net.fc2.weight.data,pathwayMatrix); 
net.fc5.weight.data = torch.mul(net.fc5.weight.data,pathwayMatrix.t());
#print(net.fc1.weight.data[4263][4]), uncomment for sanity check



for i in range(1, 2000):

  #randmly sample 5 traning samples
  randCols = [np.random.randint(0, 171) for p in range(0, 5)]
  inputsVar = TrainData[: , randCols].transpose()

  inputs = Variable(torch.from_numpy(inputsVar).float())
  target = Variable(torch.from_numpy(inputsVar).float())


  output = net(inputs)

  loss = criterion(output, target.view(-1, num_flat_features(target)))
  optimizer.zero_grad()
  loss.backward()

  # to make sure the architecture doesn 't change 
  net.fc1.weight.grad.data = torch.mul(net.fc1.weight.grad.data, sparseMatrix);
  net.fc6.weight.grad.data = torch.mul(net.fc6.weight.grad.data, sparseMatrix.t());
  net.fc2.weight.grad.data = torch.mul(net.fc2.weight.grad.data,pathwayMatrix); 
  net.fc5.weight.grad.data = torch.mul(net.fc5.weight.grad.data,pathwayMatrix.t());

  optimizer.step()# Does the update
  print('For epoc %d loss is %.6f' % (i, loss.data[0]))

  if i % 100 == 0:

    #randmly sample 5 test samples
    randCols = [np.random.randint(0, 171) for p in range(0, 5)]
    inputsVar = TestData[: , randCols].transpose()
    inputs = Variable(torch.from_numpy(inputsVar).float())
    target = Variable(torch.from_numpy(inputsVar).float())
    optimizer.zero_grad()
    output = net(inputs)
    outputBin = output.data.numpy() > 0.5
    finalOutput = np.equal(outputBin,inputs.data.numpy()>0)

    print('Accuarcy is %.6f' % (np.sum(finalOutput) / inputsVar.size))




net.save_state_dict('/mnt/boost_fieldA/SorghumBioencoder/SorghumBioencoder/trained.pt')
