import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from src.dataloader import get_dataloader
from src.generator import Generator
from src.discriminator import Discriminator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for results & models
# 使用相对路径，确保在任何位置运行 main.py 都能找到正确的目录
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

# Training function
def train(parquet, img_dir, epochs, batch_size, lr, latent_dim, embed_dim):
    loader = get_dataloader(parquet, img_dir, batch_size) #
    G = Generator(latent_dim, embed_dim).to(device) #
    D = Discriminator(embed_dim).to(device) #
    opt_G = Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    
    # --- 关键修改：使用更稳定的损失函数 ---
    criterion = nn.BCEWithLogitsLoss()

    # 创建固定的噪声和嵌入，用于可视化训练过程中的效果演变
    fixed_noise = torch.randn(16, latent_dim, device=device)
    fixed_embeds_batch = next(iter(loader))
    fixed_embeds = fixed_embeds_batch[1][:16].to(device)


    for e in range(1, epochs+1):
        for i, (imgs, embeds, _) in enumerate(loader):
            imgs, embeds = imgs.to(device), embeds.to(device)
            bs = imgs.size(0)
            
            # 真实标签和伪造标签
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # --- (1) 训练判别器 ---
            D.zero_grad()
            
            # 在真实数据上
            real_output = D(imgs, embeds)
            loss_D_real = criterion(real_output, real_labels)
            
            # 在伪造数据上
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z, embeds)
            fake_output = D(fake_imgs.detach(), embeds)
            loss_D_fake = criterion(fake_output, fake_labels)
            
            # 合并损失并更新
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # --- (2) 训练生成器 ---
            G.zero_grad()
            # 生成器的目标是让判别器认为伪造图片是真实的
            pred_on_fake = D(fake_imgs, embeds)
            loss_G = criterion(pred_on_fake, real_labels) # 使用真实标签来计算损失
            loss_G.backward()
            opt_G.step()
            
            if i % 100 == 0:
                 print(f"Epoch [{e}/{epochs}] Batch [{i}/{len(loader)}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        # --- (3) 每个 epoch 结束后保存检查点和样本 ---
        if e % 10 == 0:
            G.eval() # 切换到评估模式
            with torch.no_grad():
                samples = G(fixed_noise, fixed_embeds)
                save_image((samples + 1) * 0.5, f"{results_dir}/sample_epoch{e}.png", nrow=4)
            G.train() # 切换回训练模式

            # 保存模型权重
            torch.save(G.state_dict(), f"{models_dir}/generator_{e}.pth")
            torch.save(D.state_dict(), f"{models_dir}/discriminator_{e}.pth")
            

    # 训练结束后保存最终模型
    torch.save(G.state_dict(), f"{models_dir}/generator_final.pth")
    torch.save(D.state_dict(), f"{models_dir}/discriminator_final.pth")

    print("Training finished!")