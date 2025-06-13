# src/utils.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Fashion-MNIST 类别名称
FASHION_MNIST_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def get_fashion_mnist_dataloader(batch_size=64, image_size=28):
    """
    加载Fashion-MNIST数据集并返回数据加载器。
    图像将被归一化到 [-1, 1] 范围。
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1, 1]
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    # 不使用测试集，只用训练集
    # test_dataset = torchvision.datasets.FashionMNIST(
    #     root='./data',
    #     train=False,
    #     download=True,
    #     transform=transform
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 # 根据你的CPU核心数调整
    )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4
    # )

    return train_loader #, test_loader

def labels_to_one_hot(labels, num_classes=10):
    """
    将数字标签转换为独热编码。
    """
    # 确保 labels 是 LongTensor
    labels = labels.long()
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def visualize_samples(images, labels, epoch, batch_idx, results_dir='./results'):
    """
    可视化生成的图像样本。
    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(8, 8, i + 1)
        # 反归一化图像到 [0, 1] 范围
        img = (images[i].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2
        img = np.clip(img, 0, 1) # 确保在有效范围内
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(FASHION_MNIST_CLASSES[labels[i].item()], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/generated_samples_epoch_{epoch:03d}_batch_{batch_idx:04d}.png")
    plt.close(fig)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

# 用于分类器评估
class SimpleClassifier(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128), # 28x28 -> 14x14 -> 7x7
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_classifier(dataloader, device, epochs=5):
    """
    训练一个简单的Fashion-MNIST分类器，用于评估GAN生成的图像。
    """
    model = SimpleClassifier(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training Classifier...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(f"Classifier Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    print("Classifier training complete.")
    return model

def evaluate_generated_images_with_classifier(classifier, generator, num_samples_per_class, latent_dim, device):
    """
    使用训练好的分类器评估GAN生成的图像。
    """
    classifier.eval() # 设置为评估模式
    total_correct = 0
    total_generated = 0
    num_classes = 10

    print("\nEvaluating generated images with classifier...")
    with torch.no_grad():
        for class_id in range(num_classes):
            # 为当前类别生成图像
            noise = torch.randn(num_samples_per_class, latent_dim, 1, 1).to(device)
            # 文本条件 (独热编码)
            class_labels = torch.full((num_samples_per_class,), class_id, dtype=torch.long).to(device)
            one_hot_labels = labels_to_one_hot(class_labels, num_classes=num_classes).to(device)

            fake_images = generator(noise, one_hot_labels)

            # 预测类别
            outputs = classifier(fake_images)
            _, predicted = torch.max(outputs.data, 1)

            # 统计正确预测的数量
            total_correct += (predicted == class_labels).sum().item()
            total_generated += num_samples_per_class

            print(f"Class {class_id} ({FASHION_MNIST_CLASSES[class_id]}): Generated {num_samples_per_class} images, Correctly classified: {(predicted == class_labels).sum().item()}")

    accuracy = 100 * total_correct / total_generated
    print(f"\nOverall Classifier Accuracy on Generated Images: {accuracy:.2f}%")
    return accuracy