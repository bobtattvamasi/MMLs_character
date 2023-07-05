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
target_tensor = torch.tensor(target_idx)


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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# Определение модели и оптимизатора
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        print(f"embed.shape ={embed.shape}")
        outputs, _ = self.rnn(embed.unsqueeze(0))  # Добавить измерение пакета
        logits = self.fc(outputs.squeeze(0))  # Убрать измерение пакета
        return logits


vocab_size = len(word_to_idx)
embedding_dim = 100
hidden_dim = 256

model = TextGenerator(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение модели
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def generate_text(model, start_word, max_length=10):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor([[word_to_idx[start_word]]])  # Изменено добавление измерения пакета
        for _ in range(max_length):
            outputs = model(input_seq)
            _, predicted_idx = torch.max(outputs, dim=2)
            predicted_word = list(word_to_idx.keys())[predicted_idx.item()]
            print(predicted_word, end=" ")
            input_seq = torch.cat([input_seq, predicted_idx], dim=1)

# Генерация текста
start_word = "Привет"
max_length = 10
generate_text(model, start_word, max_length)





