import random
import json
import torch
from brain import NeuralNet
from NN import bag_of_words, tokenize
from train import file_path
from Speak import say
from Listen import listen  # Assuming Listen is a module with the listen function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intent data
with open(file_path, 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model data
file = "TrainData.pth"
data = torch.load(file)

# Extract model data
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

# Initialize the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Set the assistant's name
Name = "ZaRa AI"

def Main():
    # Listen for user input
    sentence = listen()

    # Check if the user wants to exit
    if sentence.lower() == "bye":
        print("Goodbye!")
        exit()

    # Process user input
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    # Make a prediction using the neural network
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Check if the prediction confidence is high enough
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Select a response based on the predicted tag
        for intent in intents['intents']:
            if tag == intent['tag']:
                reply = random.choice(intent["responses"])
                say(reply)
    else:
        # If confidence is not high, ask for clarification or handle accordingly
        print(f"{Name}: I'm not sure how to respond to that. Can you please rephrase?")
        # You might want to add additional logic here to handle low-confidence responses

# Execute the assistant
Main()
