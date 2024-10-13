from loss import *
from utils import *
from generator import MSHAN
from discriminator import MSHAD

import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader, num_epochs=50, save_interval=25, sample_interval=5, checkpoint_path="models/checkpoint.pth"):
    # optimizers
    g_optimizer = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

    # schedulers
    g_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[30, 60, 90], gamma=0.5)
    d_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[30, 60, 90], gamma=0.5)

    # loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_ssim = SSIM(window_size=11, size_average=True).to(device)

    # training parameters
    start_epoch = 1
    lambda_GAN = 0.001
    lambda_pixel = 1.0
    lambda_perceptual = 0.01
    lambda_ssim = 0.1

    # load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        start_epoch = load_model(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)
        print(f"Resuming from epoch {start_epoch}")
        start_epoch += 1
    
    generator.train()

    # training loop
    for epoch in range(start_epoch, num_epochs+1):
        for lr_imgs, hr_imgs in dataloader:
            valid = torch.ones((lr_imgs.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((lr_imgs.size(0), 1), requires_grad=False).to(device)

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # train generator
            gen_imgs = generator(lr_imgs)

            loss_GAN = criterion_GAN(discriminator(gen_imgs), valid)
            loss_pixel = criterion_pixelwise(gen_imgs, hr_imgs)
            loss_perceptual = criterion_perceptual(gen_imgs, hr_imgs)
            loss_ssim = criterion_ssim(gen_imgs, hr_imgs)
            
            g_loss = (
                lambda_GAN * loss_GAN +
                lambda_pixel * loss_pixel +
                lambda_perceptual * loss_perceptual + 
                lambda_ssim * loss_ssim
            )
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Train Discriminator
            real_loss = criterion_GAN(discriminator(hr_imgs), valid)
            fake_loss = criterion_GAN(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        
        # step through the scheduler at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()

        # save and sample checkpoints
        if epoch % sample_interval == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            save_lr_hr_grid(gen_imgs.detach(), hr_imgs, f"samples/epoch_{str(epoch).zfill(len(str(num_epochs)))}.png", normalize=True, num_pairs=4, nrow=2)
        if epoch % save_interval == 0:
            if epoch > save_interval:
                os.rename(checkpoint_path, f"models/checkpoint-{epoch-save_interval}.pth")
            save_model(epoch, generator, discriminator, g_optimizer, d_optimizer, checkpoint_path)


if __name__ == "__main__":
    # instantiate models
    generator = MSHAN(in_channels=3, out_channels=3, dim=64, num_blocks=6, num_heads=8, upscale_factor=2)
    discriminator = MSHAD(in_channels=3, base_channels=64)

    # instantiate data loader
    dataset = ImageDataset(
        hr_dir="",
        lr_dir=""
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # device configuration
    generator.to(device)
    discriminator.to(device)
    
    try:
        # start training
        train(dataloader, num_epochs=100, save_interval=25, sample_interval=1)
    except KeyboardInterrupt:
        pass
