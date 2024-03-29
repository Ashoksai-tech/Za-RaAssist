import random 
import json
import torch
from brain import NeuralNet
from NN import bag_of_words,tokenize
from train import file_path
from Speak import say


device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')
with open(file_path,'r') as json_data:
    intents = json.load(json_data)


file = "TrainData.pth"
data = torch.load(file)


input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']


model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


Name = "ZaRa AI"
from Listen import listen



def Main():


    sentence = listen()

    if sentence=="bye":
        exit()


    sentence = tokenize(sentence)
    x = bag_of_words(sentence,all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _ , predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item()  > 0.75:
        for intent in intents['intents']:
            if tag ==intent['tag']:
                reply = random.choice(intent["responses"])
                say(reply)
                


Main()




 


