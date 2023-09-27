import torch
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from utils import MakeData
from model import BERT
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(model, data):

    model.train()
    for parm in model.bert.parameters():
        parm.required_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterian = torch.nn.CrossEntropyLoss()

    num_epochs = 3
    train_loss = []
    for epoch in range(num_epochs):
        loop = tqdm(data)
        for i, (ids, mask, label) in enumerate(loop):
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            y_hate = model(ids=ids,mask=mask)
            loss = criterian(y_hate, label)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss = loss.item())
            train_loss.append(loss.item())

    torch.save(model.state_dict(), f'checkpoint/model_state{num_epochs}.pt')
    plt.plot(train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()
    pass


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

path = 'data/ClassificationDataset-train0.xlsx'
train_data = MakeData(path, tokenizer)

train_loader = DataLoader(train_data, batch_size=32)

model = BERT(bert=bert, h_dim=128, op_dim=3)
model = model.to(device)

training(model, train_loader)





