import torch
import torch.nn as nn
import torchvision.models as models


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    @staticmethod
    def gaussian_window(size, sigma):
        """
        Generates a 1-D Gaussian window for computing local means and variances.
        """
        coords = torch.arange(size).float() - size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """
        Create a 2D Gaussian window for SSIM calculation.
        """
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
        Calculate the SSIM index between two images.
        """
        mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        # NOTE: scaled for -1, 1
        C1 = (0.01 * 2) ** 2 
        C2 = (0.03 * 2) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window is not None:
            pass
        else:
            self.channel = channel
            self.window = self.create_window(self.window_size, self.channel).to(img1.device)

        return 1 - self.ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        weights = models.VGG19_Weights.DEFAULT
        # QUESTION: do we only want the first 20 features?
        vgg = models.vgg19(weights=weights).features[:20].eval()
        
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # Using the functional module for L1 loss
        loss = nn.functional.l1_loss(self.vgg(generated), self.vgg(target))
        return loss
