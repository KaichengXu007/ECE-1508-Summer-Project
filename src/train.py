# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.networks import Generator, Discriminator
from src.utils import get_fashion_mnist_dataloader, labels_to_one_hot, \
                      visualize_samples, weights_init, \
                      SimpleClassifier, train_classifier, evaluate_generated_images_with_classifier, \
                      FASHION_MNIST_CLASSES

def train(epochs, batch_size, latent_dim, lr, beta1, device, save_interval):
    # 1. 数据加载
    dataloader = get_fashion_mnist_dataloader(batch_size=batch_size)
    num_classes = len(FASHION_MNIST_CLASSES)
    img_channels = 1 # Fashion-MNIST 是灰度图

    # 2. 初始化生成器和判别器
    netG = Generator(latent_dim, num_classes, img_channels).to(device)
    netD = Discriminator(num_classes, img_channels).to(device)

    # 对网络权重进行初始化
    netG.apply(weights_init)
    netD.apply(weights_init)

    # 3. 定义损失函数和优化器
    criterion = nn.BCELoss() # 二元交叉熵损失

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # 4. 定义真实和虚假标签
    real_label = 1.
    fake_label = 0.

    print(f"Starting training on {device}...")

    # For saving fixed generated samples noise and labels
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device) # Generating 64 images
    
    # NEW LINE: Generate 64 labels by cycling through classes
    # It will cycle through 0-9, then 0-9 again, and so on, until 64 labels are generated.
    fixed_labels_numeric = torch.tensor([i % num_classes for i in range(fixed_noise.shape[0])], dtype=torch.long).to(device)
    
    fixed_labels_one_hot = labels_to_one_hot(fixed_labels_numeric, num_classes).to(device)

    # 训练分类器用于评估
    print("Pre-training a classifier for evaluation...")
    classifier = train_classifier(dataloader, device, epochs=5) # 用真实数据训练一个分类器
    classifier_path = './models/fashion_mnist_classifier.pth'
    torch.save(classifier.state_dict(), classifier_path)
    print(f"Classifier saved to {classifier_path}")

    # 训练循环
    for epoch in range(epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            real_images, real_labels_numeric = data
            real_images = real_images.to(device)
            real_labels_one_hot = labels_to_one_hot(real_labels_numeric, num_classes).to(device)
            b_size = real_images.size(0)

            # (1) 更新判别器：最大化 D(x, c) + D(G(z, c), c)
            netD.zero_grad()
            
            # 训练真实样本
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images, real_labels_one_hot).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # 训练生成样本
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            # 随机选择标签用于生成假图像
            fake_labels_numeric = torch.randint(0, num_classes, (b_size,), device=device)
            fake_labels_one_hot = labels_to_one_hot(fake_labels_numeric, num_classes).to(device)
            
            fake_images = netG(noise, fake_labels_one_hot)
            label.fill_(fake_label)
            output = netD(fake_images.detach(), fake_labels_one_hot).view(-1) # detach() 避免梯度回传到G
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) 更新生成器：最大化 D(G(z, c), c)
            netG.zero_grad()
            label.fill_(real_label) # 生成器希望判别器将假图像识别为真
            output = netD(fake_images, fake_labels_one_hot).view(-1) # 再次计算，因为D的参数已更新
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 100 == 0:
                print(f'[{epoch+1}/{epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
                # 可视化生成结果
                with torch.no_grad():
                    # 确保fixed_noise和fixed_labels_one_hot维度匹配生成器的输入
                    fixed_noise_reshaped = fixed_noise.squeeze() # (64, latent_dim) if Generator expects 2D
                    generated_samples = netG(fixed_noise, fixed_labels_one_hot).detach()
                    visualize_samples(generated_samples, fixed_labels_numeric, epoch, i)

        # 每个epoch结束时评估生成图像
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        evaluate_generated_images_with_classifier(classifier, netG, 100, latent_dim, device) # 每类生成100张

        # 保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f'./models/netG_epoch_{epoch+1}.pth')
            torch.save(netD.state_dict(), f'./models/netD_epoch_{epoch+1}.pth')
            print(f"Models saved at epoch {epoch+1}")

    print("Training finished!")