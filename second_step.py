import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Пример входных данных и выходных меток
input_data = ["Привет", "Как", "дела"]
target_data = ["Привет", "Хорошо", "спасибо"]

# Преобразование токенов в индексы
word_to_idx = {"Привет": 0, "Как": 1, "дела": 2, "Хорошо": 3, "спасибо": 4}
input_idx = [word_to_idx[word] for word in input_data]
target_idx = [word_to_idx[word] for word in target_data]

# Преобразование в тензоры
input_tensor = torch.tensor(input_idx)
target_tensor = torch.tensor(target_idx).unsqueeze(1)  # Reshape target tensor

# Создание датасета и загрузчика данных
class TextDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = TextDataset(input_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# Определение модели
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        embed = self.embedding(input_seq)
        output, _ = self.rnn(embed.view(len(input_seq), 1, -1))
        output = self.fc(output.view(len(input_seq), -1))
        return output

# Параметры модели и обучения
input_size = len(word_to_idx)
hidden_size = 100
output_size = len(word_to_idx)
learning_rate = 0.01
n_epochs = 10

# Создание экземпляра модели
model = RNN(input_size, hidden_size, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(1, n_epochs + 1):
    for input_seq, target_idx in dataloader:
        optimizer.zero_grad()
        outputs = model(input_seq)
        print(f"outputs. shape = {outputs.shape}")
        print(f"target_idx. shape = {target_idx.shape}")
        target_idx = target_idx.squeeze()
        loss = criterion(outputs, target_idx)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")

# Set the path to save the model
model_path = f"weights/rnn_model_epoch-{epoch}_loss-{loss.item():.4f}.pt"
# Save the model
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")



# Генерация текста
def generate_text(model, start_word, max_length=10):
    model.eval()
    with torch.no_grad():
        word_idx = word_to_idx[start_word]
        input_seq = torch.tensor([word_idx])
        for _ in range(max_length):
            outputs = model(input_seq)
            _, predicted_idx = torch.max(outputs, dim=1)
            predicted_word = list(word_to_idx.keys())[predicted_idx.item()]
            print(predicted_word, end=" ")
            input_seq = predicted_idx

# Задание начального слова и максимальной длины генерируемой последовательности
start_word = "Привет"
max_length = 10

generate_text(model, start_word, max_length)

