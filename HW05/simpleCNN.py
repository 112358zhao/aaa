import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] ='SimHei' # 使用支持中文的字体
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc = nn.Linear(in_features=16*14*14,out_features=10)
    def forward(self,x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1,16*14*14)
        x = self.fc(x)
        return x
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader,test_loader
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()  # 设置为训练模式
    train_losses = []  # 存储每个epoch的平均损失

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
      
        for i, (images, labels) in enumerate(train_loader):
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
          
            # 清零梯度
            optimizer.zero_grad()
          
            # 前向传播
            outputs = model(images)
          
            # 计算损失
            loss = criterion(outputs, labels)
          
            # 反向传播
            loss.backward()
          
            # 更新参数
            optimizer.step()
          
            # 累加损失
            running_loss += loss.item()
          
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          
        # 计算每个epoch的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
      
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses
def test(model, test_loader, criterion, device):
    model.eval()  # 设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度，节省内存
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
          
            # 前向传播
            outputs = model(images)
          
            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()
          
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return test_loss, accuracy
def display_data(data_loader, num_images=25):
    # 获取一批数据
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # 创建图形
    plt.figure(figsize=(10, 10))

    # 显示图像
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(grid_size, grid_size, i+1)
        # 转换图像格式：[C,H,W] -> [H,W]，并反归一化
        img = images[i][0].cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'标签: {labels[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.show()
def visualize_predictions(model, test_loader, device, num_images=5):
    model.eval()

    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # 将图像移动到指定设备并进行预测
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 将图像移回CPU用于显示
    images = images.cpu()

    # 创建图形
    plt.figure(figsize=(12, 4))

    # 显示图像和预测结果
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(1, num_images, i+1)
        img = images[i][0].numpy()
        plt.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'预测: {predicted[i]}\n真实: {labels[i]}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
def main():
    """主函数，协调整个流程：加载数据、创建模型、训练、测试和可视化
    """
    # 设置随机种子以便结果可复现
    # 这确保每次运行代码时得到相同的随机初始化结果
    torch.manual_seed(42)  # 42是一个常用的随机种子值

    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')  # 打印使用的计算设备

    # 加载训练和测试数据，批量大小为64
    train_loader, test_loader = load_data(batch_size=64)

    # 显示部分训练数据，帮助理解数据集
    display_data(train_loader)  # 默认显示25个样本

    # 创建模型实例并移动到指定设备（CPU或GPU）
    model = SimpleCNN().to(device)
    print(model)  # 打印模型结构

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率为0.001

    # 训练模型，运行5个训练周期（epoch）
    train_losses = train(model, train_loader, criterion, optimizer, device, epochs=5)

    # 在测试集上评估模型性能
    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    # 可视化一些预测结果，直观地展示模型效果
    visualize_predictions(model, test_loader, device)  # 默认显示5个样本

    # 保存训练好的模型参数，便于未来使用
    torch.save(model.state_dict(), 'simple_cnn_mnist.pth')  # 只保存模型参数，而非整个模型
    print('模型已保存为 simple_cnn_mnist.pth')

    # 绘制训练损失曲线，可视化训练过程
    plt.figure(figsize=(10, 5))  # 创建图形，大小为10x5英寸
    plt.plot(train_losses, label='Training Loss')  # 绘制损失曲线
    plt.title('Training Loss Over Epochs')  # 添加标题
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')  # y轴标签
    plt.legend()  # 显示图例
    plt.savefig('training_loss.png')  # 保存图像到文件
    plt.show()  # 显示图像
if __name__ == '__main__':
    main()
    