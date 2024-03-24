from torchvision.models import resnet50, resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch


class ThreeInOne(nn.Module):
    def __init__(self, num_classes, kernel=13, hard=False, cat=False, additional_out=False, extra=False):
        super(ThreeInOne, self).__init__()
        self.extra = extra
        self.additional_out = additional_out
        self.hard = hard
        self.base = resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(self.base.children())[:-1])
        # if self.extra:
        #     self.final_fc1 = nn.Linear(2048 + 5 * 128, 4096, bias=True)
        # else:
        #     self.final_fc1 = nn.Linear(2048, 4096, bias=True)
        # self.final_fc2 = nn.Linear(4096, num_classes, bias=True)

        if self.extra:
            self.final_fc = nn.Linear(2048 + 5 * 128, 4096, bias=True)
        else:
            self.final_fc = nn.Linear(512, num_classes, bias=True)

        # kernel = 23
        # 用来找每个focal plane的重要区域，kernel表示关键区域大小
        # 原本意思是用2个conv计算二次导数，然后计算清晰度
        self.cat = cat
        if not cat:
            self.conv1 = nn.Conv2d(9, 64, (kernel, kernel), (1, 1), padding=(kernel // 2, kernel // 2))
            self.conv2 = nn.Conv2d(64, 128, (kernel, kernel), (1, 1), padding=(kernel // 2, kernel // 2))
            self.conv3 = nn.Conv2d(128, 3, (kernel, kernel), (1, 1), padding=(kernel // 2, kernel // 2))
        # torchv1.6:nn.Conv2d(128, 3, kernel, 1, padding=kernel//2)
        # self.RGB = torch.Tensor([0.299, 0.587, 0.114]).unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.RGB = [0.299, 0.587, 0.114]

        if self.extra:
            self.embeeding_stage = nn.Embedding(num_embeddings=7, embedding_dim=128)
            self.embeeding_icm = nn.Embedding(num_embeddings=5, embedding_dim=128)
            self.embeeding_t = nn.Embedding(num_embeddings=5, embedding_dim=128)
            self.embeeding_i1 = nn.Embedding(num_embeddings=8, embedding_dim=128)
            self.embeeding_i2 = nn.Embedding(num_embeddings=5, embedding_dim=128)

    def forward(self, x):
        # print(x.shape)
        # input = bs * 3(focal plane) * 3(channel) * 224 * 224
        if self.extra:
            x, extra = x
            stage, icm, t, i1, i2 = torch.split(extra, split_size_or_sections=1, dim=1)
            # print(extra)
            stage = self.embeeding_stage(stage)
            icm = self.embeeding_icm(icm)
            t = self.embeeding_t(t)
            i1 = self.embeeding_i1(i1)
            i2 = self.embeeding_i2(i2)
            # print('end')
            # extra = torch.cat([stage, icm, t], dim=2).squeeze(dim=1)
            extra = torch.cat([stage, icm, t, i1, i2], dim=2).squeeze(dim=1)
        bs, f, c, h, w = x.shape
        x_weight = x.reshape(bs, f * c, h, w)
        if not self.cat:
            x_weight = self.conv1(x_weight)
            x_weight = self.conv2(x_weight)
            x_weight = self.conv3(x_weight)
            if self.hard:
                x_weight = F.gumbel_softmax(x_weight, dim=1, hard=True)
            # 对应每个focal plane在每个位置的占比大小
            # f1 = x_weight[:, 0, ...].unsqueeze(dim=1)
            # f2 = x_weight[:, 1, ...].unsqueeze(dim=1)
            # f3 = x_weight[:, 2, ...].unsqueeze(dim=1)
            # 改进写法：
            f1, f2, f3 = torch.split(x_weight, 1, dim=1)

            x = x[:, 0, ...] * f1 + x[:, 1, ...] * f2 + x[:, 2, ...] * f3
            # x = torch.cat([x[:,0,...] * f1, x[:,1,...] * f2, x[:,2,...] * f3],dim=1)
            # RGB = self.RGB.expand([bs,-1,-1,-1])
            # x_gray = (self.RGB * x).sum(dim=1)#.unsqueeze(dim=1)
            x_gray = x[:, 0, ...] * self.RGB[0] + x[:, 1, ...] * self.RGB[1] + x[:, 2, ...] * self.RGB[2]
        else:
            x1, x2, x3 = torch.split(x, 1, dim=1)
            x1 = (x1[:, :, 0, ...] * self.RGB[0] + x1[:, :, 1, ...] * self.RGB[1] + x1[:, :, 2, ...] * self.RGB[
                2])  # .unsqueeze(dim=1)
            x2 = (x2[:, :, 0, ...] * self.RGB[0] + x2[:, :, 1, ...] * self.RGB[1] + x2[:, :, 2, ...] * self.RGB[
                2])  # .unsqueeze(dim=1)
            x3 = (x3[:, :, 0, ...] * self.RGB[0] + x3[:, :, 1, ...] * self.RGB[1] + x3[:, :, 2, ...] * self.RGB[
                2])  # .unsqueeze(dim=1)
            x = torch.cat([x1, x2, x3], dim=1)
            x_gray = x

        # x_fft = torch.fft.fftshift(torch.fft.fft2(x_gray))
        # x_fft[h//2 - 30:h//2 + 30, w//2 - 30:w//2 + 30] = 0
        # x_hp = torch.fft.ifft2(torch.fft.ifftshift(x_fft))
        # magnitude = torch.log(torch.abs(x_hp))
        # mean = torch.mean(magnitude)

        feature_out = self.feature(x)
        x = feature_out.reshape(bs, -1)
        if self.extra:
            x = torch.cat([x, extra], dim=1)
        # print(x.shape)
        x = self.final_fc(x)
        # x = self.final_fc2(x)
        if self.additional_out:
            output = x, feature_out
        else:
            output = x
        return output


if __name__ == '__main__':
    # data = torch.rand(4, 3, 3, 224, 224).cuda()
    # net = ThreeInOne(num_classes=2, cat=True).cuda()
    # net(data)
    data = [torch.rand(4, 3, 3, 224, 224).cuda(), torch.randint(low=0, high=6, size=(4, 5)).cuda()]
    net = ThreeInOne(num_classes=2, cat=True, extra=True).cuda()
    net(data)
