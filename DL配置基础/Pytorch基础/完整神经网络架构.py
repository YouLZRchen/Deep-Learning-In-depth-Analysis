# 本文件中的架构将实现对Fashion MNIST数据集的分类

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------- 1. 设置设备 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------------------- 2. 数据预处理与加载 ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 类别标签
class_names = [
    "T‑shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ---------------------- 3. 简单神经网络模型 ----------------------
class FashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

model = FashionNet().to(device)

# ---------------------- 4. 损失函数、优化器 ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------- 5. 训练函数 ----------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# ---------------------- 6. 测试评估函数 ----------------------
def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# ---------------------- 7. 开始训练 ----------------------
epochs = 8
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | Test Loss:{test_loss:.4f} Acc:{test_acc:.4f}")

# ---------------------- 8. 单样本推理演示 ----------------------
def predict_sample(model, dataset, index, device):
    model.eval()
    img, true_label = dataset[index]
    plt.figure()
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"True label: {class_names[true_label]}")
    plt.axis("off")

    with torch.no_grad():
        input_tensor = img.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
    print(f"预测类别：{class_names[pred_idx]}，真实类别：{class_names[true_label]}")
    plt.show()

# 拿测试集第0号样本做推理
predict_sample(model, test_dataset, 0, device)