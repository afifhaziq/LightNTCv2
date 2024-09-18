
import torch
import torch.nn as nn
import numpy as np
from torchview import draw_graph

device = ('cuda' if torch.cuda.is_available() else 'cpu')


torch.cuda.set_per_process_memory_fraction(0.4, 0)

import random


def seed_everything(seed: int) -> None:
    """Seeds everything so that experiments are deterministic.

    Args:
        seed (int): Seed value.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(134)



sequence = 20
features =18
learning_rate = 0.001
batch_size = 64
num_class = 10
num_epoch = 10

# classes = ['aimchat', 
#            'email', 
#            'facebookChat', 
#            'ftps',
#            'gmailchat', 
#            'hangoutaudio', 
#            'icqchat', 
#            'netflix', 
#            'scp', 
#            'sftp', 
#            'skypevideo', 
#            'spotify', 
#            'vimeo', 
#            'voipbuster', 
#            'youtube']

classes = ('AIM Chat','Email','Facebook Audio','Facebook Chat','Gmail Chat','Hangouts Chat','ICQ Chat','Netflix','Spotify','Youtube')



# x_train = np.load('x_train_30K.npy')
# y_train = np.load('y_train_30K.npy')
# x_test = np.load('x_test_30K.npy')
# y_test = np.load('y_test_30K.npy')

x_train = np.load("x_train-MLP-Multiclass-ISCX-740features.npy")
y_train = np.load("y_train-MLP-Multiclass-ISCX-740features.npy")
x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")



#x_train = np.delete(x_train, [12,13,14,15,16,17,18,19], 1)
#x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1)

# Replace the values in columns 12 to 19 with 0 in x_train
x_train[:, 12:20] = 0
# Replace the values in columns 12 to 19 with 0 in x_test
x_test[:, 12:20] = 0

x_train = x_train[:,:sequence*features]
x_test = x_test[:,:sequence*features]



x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

print(x_train.shape)
print(x_train.shape)




from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True,
                                           batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           shuffle=False,
                                           batch_size=batch_size)



import gc
del train_dataset, test_dataset, x_train, y_train, x_test, y_test
gc.collect()



class NtCNN(nn.Module):
    def __init__(self, use_se=True):
        super(NtCNN, self).__init__()

        self.use_se = use_se
        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(sequence, 16, kernel_size=1),
            nn.ReLU()
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(sequence, 24, kernel_size=1),  # Reduce channels from 36 to 24
            nn.Conv1d(24, 32, kernel_size=3, padding=1)  # 3x3 convolution 
            ,nn.ReLU()
            #,nn.GELU()    
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(sequence, 8, kernel_size=1),  # Reduce channels from 36 to 8
            nn.Conv1d(8, 16, kernel_size=5, padding=2)  # 5x5
            ,nn.ReLU()
            #,nn.GELU()            
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(sequence, 16, kernel_size=1)  # Reduce channels to 16
            ,nn.ReLU()
            #,nn.GELU()  
        )

        # Squeeze and excite block
        if self.use_se:
            hidden_dim = 80
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

        self.convReduce = nn.Conv1d(hidden_dim, 12, kernel_size=1, bias=False)
        self.hswish = nn.Hardswish(inplace=True)
        self.bn3 = nn.BatchNorm1d(sequence)
        #self.pool = nn.AdaptiveAvgPool1d(30)
        self.fc = nn.Linear(12*18, 64)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_class)
        # self.fc1 = nn.Linear(80*12, 128)
        # self.activation5 = nn.ReLU()

        # self.fc2 = nn.Linear(128, 64)
        # self.activation6 = nn.ReLU()
        
        # self.fc3 = nn.Linear(64,num_class)
        
    def forward(self, x):
        
        x = x.view(-1, sequence, features)
        
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        x = torch.cat(outputs, 1)
        #print(x.shape)
        if self.use_se:
            se = self.sePool(x)
            se = self.seFlatten(se)
            se = self.seFC1(se)
            se = self.seFC2(se)
            
            se = self.seRelu(se)
            se = self.seSigmoid(se)
            #print(se.shape)
            # Input shape: (batch_size, hidden_dim = (in_channels*4))
            se = se[:,:,None]
            # Ouput shape: (batch_size, hidden_dim, 1)
            # print(se.shape)
            # print(x.shape)
            x = x * se

        #x = self.pool(x)
        x = self.hswish(self.convReduce(x))
        #print(x.shape)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = x.view(-1, 12*18)
        x = self.fc(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)

        # x = x.view(-1, 80*12)
        # # #print(x.shape)
        # x = self.fc1(x)
        # x = self.activation5(x)
        # # #print(x.shape)
        # x = self.fc2(x)
        # x = self.activation6(x)
        
        # x = self.fc3(x)
        # # #print(x.shape)
        return x



model = NtCNN(use_se=True).to(device)

from torchinfo import summary
summary(model, 
        input_size=[batch_size, sequence, features], 
        device=device, 
        col_names=["input_size","output_size", "num_params"])



        


from tqdm.auto import tqdm
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
    for i, (sample, labels) in enumerate(tqdm(train_loader)):
        model.train()
        sample = sample.to(device)
        labels = labels.to(device)

        predictions = model(sample)
        
        loss = criterion(predictions,labels)

        avgloss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        
    avg_val_loss = 0

    model.eval()
    with torch.inference_mode():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device)

            predictions = model(samples)
            #labels = torch.argmax(labels, dim=1)
            
            val_loss = criterion(predictions, labels)

            avg_val_loss += val_loss.item()

    
    avgloss /= len(train_loader)
    avg_val_loss /= len(test_loader)
    print(f'Epoch : {epoch+1}/{num_epoch}, Loss: {avgloss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    wandb.log({"epoch": (epoch+1), "loss": avgloss, "Validation Loss": avg_val_loss})
end = time.time()

print(f'Time Taken:', (end-start))




from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib as plt
all_preds = []
all_labels = []

model.eval()  # Set model to evaluation mode

  # Disable gradient computation for testing
with torch.inference_mode():    
    for images, labels in test_loader:
        images = images.to(device).float()  # Move images to the appropriate device
        labels = labels.to(device).float()  # Move labels to the appropriate device

        predictions = model(images)  # Get predictions from the model

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

