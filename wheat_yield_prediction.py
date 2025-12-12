### Imports

import pandas as p
import numpy as n
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Dummy Data in CSV format

n.random.seed(42) # make consistency in random numbers
n_samples = 1000

# features
rainfall = n.random.uniform(low=50, high=300, size=n_samples) # mm
temperature = n.random.uniform(low=20, high=40, size=n_samples) # c
fertilizers = n.random.uniform(low=20, high=100, size=n_samples) # kg

# real data mimicking ()
yield_data = (0.01 * rainfall ) - (0.1 * temperature) + (0.05 * fertilizers ) + 2.0
yield_data = yield_data + n.random.normal(0, 0.2, n_samples)

# save csv
data = p.DataFrame({'Rainfall': rainfall,
                    'Temperature': temperature,
                    'Fertilizers': fertilizers,
                    'Yield': yield_data
                    })
data.to_csv('data.csv', index=False)
print('data generated')

### Preprocess

# load data
data = p.read_csv('data.csv')
X = data[['Rainfall', 'Temperature', 'Fertilizers']].values
y = data[['Yield']].values

### train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### scaling
scalar_x = StandardScaler()
scalar_y = StandardScaler()

X_train = scalar_x.fit_transform(X_train)
X_test = scalar_x.transform(X_test)
y_train = scalar_y.fit_transform(y_train)
y_test = scalar_y.transform(y_test)

# tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tesnor = torch.tensor(y_train, dtype=torch.float32)

### Model

import torch.nn as nn
import torch.optim as optim

# define the model
class YieldPredictor(nn.Module):
    def __init__(self, input_size):
        super(YieldPredictor, self).__init__()
        self.layer1 = nn.Linear(3,16) # 3 features -> 16 hidden neurons
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16,8) # 16 hidden -> 8 hidden
        self.layer3 = nn.Linear(8,1) # 8 hidden -> 1 output

    def forward (self, x):
      x = self.relu(self.layer1(x))
      x = self.relu(self.layer2(x))
      x = self.layer3(x)
      return x

model = YieldPredictor(input_size=3)

### Loss and Optimization

loss_is = nn.MSELoss() # mean squared error
optimizer = optim.Adam(model.parameters(), lr=0.01)

### Traning

epochs = 100
for epoch in range(epochs):
  # forward pass
  outputs = model(X_train_tensor)
  loss = loss_is(outputs, y_train_tesnor)

  #backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

### model save
torch.save(model.state_dict(), 'crop_yield.pth')
print('model saved')