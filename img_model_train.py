import os
import pickle

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers import SwinModel


batch_size = 32

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class CustomDataset(Dataset):
    def __init__(self, address,label, transform=None):
        self.root_dir = ''
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像尺寸为 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        self.file_list = address
        self.labels = label




    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def load_dataset(train_batch_size, test_batch_size,workers=0):
    data = pd.read_csv('cv_address.csv',encoding='gbk', engine='python')
    len1 = int(len(list(data['labels'])))
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8)
    # Dataset
    train_set = CustomDataset(tr_sen, tr_lab)
    test_set = CustomDataset(te_sen, te_lab)
    # DataLoader
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                               pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True)
    return train_loader, test_loader

train_dataloader, test_dataloader = load_dataset(batch_size,batch_size)
LOCAL_SWIN_MODEL_PATH = os.environ.get(
    'SWIN_LOCAL_PATH',
    r'D:\ei\Sentiment_Analysis_Imdb-master\swin\weight'
)
DEFAULT_SWIN_MODEL_NAME = os.environ.get(
    'SWIN_MODEL_NAME',
    'microsoft/swin-tiny-patch4-window7-224'
)


def load_swin_backbone():
    if os.path.isdir(LOCAL_SWIN_MODEL_PATH):
        print(f"Loading local Swin weights from: {LOCAL_SWIN_MODEL_PATH}")
        return SwinModel.from_pretrained(LOCAL_SWIN_MODEL_PATH)

    print(f"Local Swin weights not found at: {LOCAL_SWIN_MODEL_PATH}")
    print(f"Trying to download pretrained Swin model: {DEFAULT_SWIN_MODEL_NAME}")
    return SwinModel.from_pretrained(DEFAULT_SWIN_MODEL_NAME)


class SwinClassifier(nn.Module):
    def __init__(self, swin_model, num_classes):
        super(SwinClassifier, self).__init__()
        self.swin_model = swin_model
        hidden_size = getattr(swin_model.config, 'hidden_size', swin_model.config.embed_dim)
        self.fc = nn.Linear(hidden_size, 1536)
        self.adjust = nn.Linear(1536, num_classes)

    def extract_features(self, x):
        outputs = self.swin_model(pixel_values=x)
        if getattr(outputs, 'pooler_output', None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, x):
        features = self.extract_features(x)
        features = self.fc(features)
        return self.adjust(features)

num_classes = 2
swin_model = load_swin_backbone()
combined_model = SwinClassifier(swin_model, num_classes)


num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)
combined_model = combined_model.to(device)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, ascii='>='):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    train_acc = correct / total

    return train_loss, train_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, ascii='>='):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    val_loss = running_loss / len(dataloader)
    val_acc = correct / total

    return val_loss, val_acc

best_loss, best_acc = float('inf'), 0
l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
for epoch in range(num_epochs):
    train_loss, train_acc = train(combined_model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(combined_model, test_dataloader, criterion, device)
    l_acc.append(train_acc)
    l_trloss.append(train_loss)
    l_teloss.append(val_loss)
    l_epo.append(epoch)
    if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
        best_acc, best_loss = val_acc, val_loss
        with open('swin_transformer.pkl', "wb") as file:
            pickle.dump(combined_model, file)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("--------------------------")
plt.plot(l_epo, l_acc)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('acc.png')

plt.plot(l_epo, l_teloss)
plt.ylabel('test-loss')
plt.xlabel('epoch')
plt.savefig('teloss.png')

plt.plot(l_epo, l_trloss)
plt.ylabel('train-loss')
plt.xlabel('epoch')
plt.savefig('trloss.png')
