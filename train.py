import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from NN import bag_of_words,tokenize,stem
from brain import NeuralNet


file_path ='C:/Users/aasho/OneDrive/Desktop/Assisant with NN,ML,DL/.vscode/intent.json'
with open(file_path,'r') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent["tags"]
    tags.append(tag)


    for pattern in intent['patterns']:
        
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
 


ignore_words = [',','?','/','.','!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


xtrain = []
ytrain = []

for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    xtrain.append(bag)
    
     


    label = tags.index(tag)
    ytrain.append(label)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

num_epochs = 1000
batch_size = 8      
learning_rate = 0.001
input_size = len(xtrain[0])
hidden_size = 8
output_size = len(tags)

print('training the model..')

class chatDataSet(Dataset):

    def __init__(self):
        self.n_samples = len(xtrain)
        self.xdata = xtrain
        self.ydata = ytrain


    def __getitem__(self,index):
        return self.xdata[index], self.ydata[index]
    

    def __len__(self):
        return self.n_samples


dataset = chatDataSet()

train_loader = DataLoader(dataset = dataset,
                          batch_size=batch_size,
                          shuffle=True,num_workers=0)


device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr =learning_rate)

for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if(epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}],Loss:{loss.item():.4f}")     

print(f"Final Loss :{loss.item():.4f}")


data = {

    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags
}
    
file = "TrainData.pth"
torch.save(data,file)

print("Training Complete, file saved to (file)")

