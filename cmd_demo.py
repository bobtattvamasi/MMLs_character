import torch
import torch.nn as nn
from second_step import RNN
from second_step import input_size, hidden_size, output_size

# Load the saved model
model_path = "weights/rnn_model_epoch-10_loss-0.7852.pt"
model = RNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the word-to-index mapping
word_to_idx = {"Привет": 0, "Как": 1, "дела": 2, "Хорошо": 3, "спасибо": 4}

# Chat loop
while True:
    # Get user input
    user_input = input("Вы: ")

    # Convert input to index
    input_idx = [word_to_idx.get(word, -1) for word in user_input.split()]

    # Check if any word is out of vocabulary
    if -1 in input_idx:
        print("Прости, я не знаю этого слова.")
        continue

    # Convert input to tensor
    input_tensor = torch.tensor(input_idx)

    # Generate response
    with torch.no_grad():
        output_tensor = model(input_tensor)
        _, predicted_idx = torch.max(output_tensor, dim=1)

    # Convert index to word
    response = " ".join([list(word_to_idx.keys())[idx] for idx in predicted_idx])
    print("Бот:", response)
