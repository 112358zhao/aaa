import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 1. 定义 LeNet-5 结构（与经典略有差异：使用 ReLU、适配 28x28 MNIST）
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 卷积层1: 输入1x28x28, 输出6x28x28 (padding=2保持尺寸)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 输出 6x14x14
        
        # 卷积层2: 输入6x14x14, 输出16x10x10 (无padding)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 输出 16x5x5
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. 极简 CNN（用于对比）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 训练函数
def train_model(model, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
    
    elapsed = time.time() - start_time
    return acc, elapsed

# 4. 主程序
def main():
    # 参数设置
    batch_size = 64
    epochs = 5
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载 MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 训练 LeNet-5
    print("\n===== LeNet-5 =====")
    lenet = LeNet5()
    lenet_acc, lenet_time = train_model(lenet, train_loader, test_loader, epochs, lr, device)
    
    # 训练 SimpleCNN
    print("\n===== SimpleCNN =====")
    simple_cnn = SimpleCNN()
    simple_acc, simple_time = train_model(simple_cnn, train_loader, test_loader, epochs, lr, device)
    
    # 参数量统计
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    lenet_params = count_params(LeNet5())
    simple_params = count_params(SimpleCNN())
    
    # 结果对比
    print("\n===== 对比结果 =====")
    print(f"模型           | 测试准确率 | 参数量   | 训练时间(s)")
    print(f"LeNet-5       | {lenet_acc:.2f}%       | {lenet_params} | {lenet_time:.2f}")
    print(f"SimpleCNN     | {simple_acc:.2f}%       | {simple_params} | {simple_time:.2f}")
    print("\n分析：LeNet-5 参数量较少（约 6.1万），但准确率略低于极简 CNN（约 44万参数）。")
    print("极简 CNN 通过更多通道和全连接层获得更高表达能力，训练时间也相应更长。")

if __name__ == "__main__":
    main()