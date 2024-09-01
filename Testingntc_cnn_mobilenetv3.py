
import torch
import torch.nn as nn
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64
learning_rate = 0.001
sequence = 30
features = 12
num_class = 10
num_epoch = 10

classes = ('AIM Chat',
           'Email',
           'Facebook Audio',
           'Facebook Chat',
           'Gmail Chat',
           'Hangouts Chat',
           'ICQ Chat',
           'Netflix',
           'Spotify',
           'Youtube')



x_train = np.load("x_train-MLP-Multiclass-ISCX-740features.npy")
y_train = np.load("y_train-MLP-Multiclass-ISCX-740features.npy")
x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")

x_train = np.delete(x_train, [12,13,14,15,16,17,18,19], 1)
x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)

x_train = x_train[: , :features*sequence]
x_test = x_test[: , :features*sequence]

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

print(x_train.shape)
print(x_train.shape)




from torch.utils.data import TensorDataset, DataLoader


train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset,
                          shuffle = True,
                          batch_size = batch_size)

test_loader = DataLoader(test_dataset,
                         shuffle = False,
                         batch_size = batch_size)

import gc
del train_dataset, test_dataset, x_train, y_train, x_test, y_test
gc.collect()


class NTCMobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_se=True, expansion_factor=4):
        super(NTCMobileNetV3Block, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_se = use_se

        # 1 x 1 Conv layer for expansion (Bottleneck)
        # bias=False due to batchnormalization
        self.expand_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Depthwise Conv (3 x 3 Conv)
        # each input channel is convolved separately, reduced computation
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, bias=False)
        # padding=kernel_size//2,
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 1 x 1 Conv layer for reduction (Bottleneck)
        self.project_conv = nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)
        
        # Squeeze and excite block
        if self.use_se:
            # Squeeze the features into 1, retains the channel number
            # Each channel is compress into only one features
            # Assign each of the channel their own weight, reducing irrelevant channel influence
            self.sePool = nn.AdaptiveAvgPool1d(1)
            # Output size: (Batch_size, channel, 1)
            self.seFlatten = nn.Flatten()
            self.seFC1 = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
            
            self.seFC2 = nn.Linear(hidden_dim // 4, hidden_dim, bias=False)
            self.seRelu = nn.ReLU(inplace=True)
            self.seSigmoid =nn.Hardsigmoid()
            
            
    
    def forward(self, x):
        out = self.hswish(self.bn1(self.expand_conv(x)))
        out = self.bn2(self.dwconv(out))
        if self.use_se:
            se = self.sePool(out)
            se = self.seFlatten(se)
            se = self.seFC1(se)
            se = self.seFC2(se)
            
            se = self.seRelu(se)
            se = self.seSigmoid(se)
            #print(se.shape)
            # Input shape: (batch_size, hidden_dim = (in_channels*4))
            se = se[:,:,None]
            # Ouput shape: (batch_size, hidden_dim, 1)
            #print(se.shape)
            out = out * se
            
        #print(out.shape)
        out = self.bn3(self.project_conv(out))
        return out

class NTCMobileNetV3(nn.Module):
    def __init__(self, num_class):
        super(NTCMobileNetV3, self).__init__()
        
        # Block 1
        self.block1 = NTCMobileNetV3Block(in_channels=sequence, out_channels=24, kernel_size=3, stride=1, use_se=True, expansion_factor=4)
        
        # Block 2
        self.block2 = NTCMobileNetV3Block(in_channels=24, out_channels=32, kernel_size=3, stride=1, use_se=True, expansion_factor=4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_class)
        
    def forward(self, x):
        x = x.view(-1, sequence, features)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        return x

model = NTCMobileNetV3(num_class).to(device)

from torchinfo import summary
summary(model, 
        input_size=[batch_size, sequence, features], 
        device=device, 
        col_names=["input_size","output_size", "num_params"])




import time
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

import wandb
wandb.init(project="torch-CNN",
           #mode='offline',
           config = {"learning_rate": learning_rate, "epochs": num_epoch, "batch_size": batch_size})

wandb.watch(model, criterion=criterion, log='all', log_freq=100)

start = time.time()
for epoch in range(num_epoch):
    avgloss = 0
    for i, (samples, labels) in enumerate (train_loader):
        model.train()
        samples, labels = samples.to(device), labels.to(device)

        prediction = model(samples)

        loss = criterion(prediction, labels)
        avgloss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    avg_val_loss = 0

    model.eval()
    with torch.inference_mode():
        for samples, labels in test_loader:
            
            samples, labels = samples.to(device), labels.to(device)

            prediction = model(samples)

            loss = criterion(prediction, labels)
            avg_val_loss += loss.item()

    avgloss /= len(train_loader)
    avg_val_loss /= len(test_loader)
    print(f"Epoch: {epoch+1}, Loss: {avgloss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    wandb.log({"epoch": (epoch+1), "loss": avgloss, "Validation Loss": avg_val_loss})

end = time.time()
print(f"Time taken: {(end-start)/60:.4f} Minutes")



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib as plt
all_preds = []
all_labels = []

model.eval()  # Set model to evaluation mode

  # Disable gradient computation for testing
with torch.inference_mode():    
    for samples, labels in test_loader:
        samples = samples.to(device).float()  # Move images to the appropriate device
        labels = labels.to(device).float()  # Move labels to the appropriate device

        predictions = model(samples)  # Get predictions from the model

        # Convert model output (predictions) to class indices
        preds = torch.argmax(predictions, dim=1)
        
        # Convert one-hot encoded labels to class indices
        labels = torch.argmax(labels, dim=1) # add this line for one hot encoded labels
        
        # Store predictions and true labels
        all_preds.extend(preds.cpu().numpy())  # Move to CPU and convert to numpy
        all_labels.extend(labels.cpu().numpy())  # Move to CPU and convert to numpy

# Generate and print the confusion matrix and classification report
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
print(confusion_matrix(all_labels, all_preds))


wandb.finish()
