#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
np.random.seed(12345)
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def identify_duplicates(x):
    
    unique_list = list(df['lesion_id'])
    tipo = list(df['dx'])
    
    if x in unique_list:
        return 'no duplicado'
    else:
        return 'imagen duplicada'
    
    
def identify_val_rows(x):
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

def plot_tensor(tns):
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(tns.permute(1, 2, 0))
    plt.show()
    plt.savefig('tensor.png')

def plot_heatmap(matrix):
    plt.figure(figsize=(10,10))
    sns.heatmap(matrix, cmap='YlOrBr', xticklabels=True, yticklabels=True, annot=True, fmt='g', vmax=100)
    plt.show()
    plt.savefig('matrix.png')
    
def classes(key):
    base_dict = {
        'nv': 0,
        'mel': 1,  
        'bkl': 2,  
        'bcc': 3,  
        'akiec': 4,
        'vasc': 5,
        'df': 6,
    }
    revd = dict([reversed(i) for i in base_dict.items()])
    base_dict.update(revd)
    return base_dict[key]

data_transform = {
    'train': transforms.Compose([
        transforms.RandomRotation(360),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(distortion_scale=0.1),
            transforms.RandomAffine(10)
        ]),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.5)
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class HAM10000Dataset(torch.utils.data.Dataset):

    def __init__(self, path='HAM10000_images_part_1/', anns_df=None, tnfs=None):

        image_ids = anns_df['image_id'].values
        targets = np.array([int(classes(x)) for x in anns_df['dx'].values])

        self.tnfs = tnfs
        self.image_ids = image_ids
        self.targets = targets
        self.path = path

    def __getitem__(self, idx):

        picture_id = self.image_ids[idx]
        image = Image.open(self.path + picture_id + '.jpg')

        if self.tnfs is not None:
            image = self.tnfs(image)
        else:
            image = transforms.ToTensor()(image)
            
        target = torch.tensor(self.targets[idx])
        return image, target

    def __len__(self):
        return len(self.image_ids)
    
def train(model, data_loader, criterion, optimizer, device):            
    model.train()
    model.to(device)
    running_loss = 0.
    running_acc = 0.
       
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
                
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
           
        _, preds = torch.max(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += torch.sum(preds == labels.data)
                
    running_loss /= len(data_loader.dataset)
    running_acc /= len(data_loader.dataset)
                
    return running_loss, running_acc 

def validate(model, data_loader, criterion, device):
    model.eval()
    model.to(device)
    running_loss = 0.
    running_acc = 0.
       
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
                        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
           
        _, preds = torch.max(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += torch.sum(preds == labels.data)
                
    running_loss /= len(data_loader.dataset)
    running_acc /= len(data_loader.dataset)
                
    return running_loss, running_acc

model = models.resnet34(pretrained=True)

for p in model.parameters():
    p.requires_grad = False

in_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(in_features, num_classes)

parameters_to_update = [p for p in model.parameters() if p.requires_grad]

batch_size = 15
num_workers = 8

df_data = pd.read_csv('HAM10000_metadata.csv')
df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)
df_data['duplicates'] = df_data['lesion_id']
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)
df_data['duplicates'] = np.where(df_data['dx']=='mel',
                           'no duplicado',      
                           df_data['duplicates']) 
df_data['duplicates'] = np.where(df_data['dx']=='bcc',
                           'no duplicado',      
                           df_data['duplicates']) 
df = df_data[df_data['duplicates'] == 'no duplicado']

metadata = df

train_df, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'].values)

data = {
    'train': HAM10000Dataset(tnfs=data_transform['train'], anns_df=train_df),
    'val': HAM10000Dataset(tnfs=data_transform['val'], anns_df=val_df),
}    

data_loader = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters_to_update, lr=0.0001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

num_epochs = 20

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train(model, data_loader['train'], criterion, optimizer, device)
        
        val_loss, val_acc = validate(model, data_loader['val'], criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print('[{}] loss: train={:.3f}, val={:.3f} -- accuracy: train={:.3f}, val={:.3f}'.format(epoch, train_loss, val_loss, train_acc, val_acc))
        
        
MODEL_NAME = 'nevos.0001lr_4bs'
torch.save(model.state_dict(), MODEL_NAME)

plt.figure(figsize=(10,10))
plt.plot(train_losses)
plt.plot(val_losses)
plt.show()
plt.savefig('loss.png')

plt.figure(figsize=(10,10))
plt.plot(train_accs)
plt.plot(val_accs)
plt.show()
plt.savefig('train.png')

MODEL_NAME = 'nevos.0001lr_4bs'
model = models.resnet34(pretrained=True)
in_features = model.fc.in_features
num_classes = 7
model.fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()

def validate_results(model, data_loader, device):
    model.eval()
    model.to(device)
    labels_arr = np.empty((0,1))
    predictions = np.empty((0,1))
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions = np.vstack([predictions, np.expand_dims(preds.cpu().detach().numpy(), axis=1)])
        labels_arr = np.vstack([labels_arr, np.expand_dims(labels.cpu().detach().numpy(), axis=1)])
    return labels_arr, predictions


labels_arr, predictions = validate_results(model, data_loader['val'], device)
print(classification_report(labels_arr, predictions))
cm = confusion_matrix(labels_arr, predictions, labels=[0,1,2,3,4,5,6])
plot_heatmap(cm)


# In[ ]:




