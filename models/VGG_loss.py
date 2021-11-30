# class VGGPerceptualLoss(torch.nn.Module):
#     ## input (N, C, H, W)
#     ## output (N, C, H, W)
#     def __init__(self, mask=None, resolution=(32, 32) ,resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         blocks = []
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
#         self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
#         # self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
#         # self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
#         self.resize = resize
#         self.image_resolution = resolution
#         self.mask = mask

    # def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
    # def forward(self, model_output, gt, feature_layers=(0, 1, 2, 3), style_layers=()):
    #     pred_img = dataio.lin2img(model_output['model_out'], self.image_resolution)
    #     gt_img = dataio.lin2img(gt['img'], self.image_resolution)
    #     if self.mask is not None:
    #         pred_img = pred_img * self.mask
    #         gt_img = gt_img * self.mask
    #     if pred_img.shape[1] != 3:
    #         pred_img = pred_img.repeat(1, 3, 1, 1)
    #         gt_img = gt_img.repeat(1, 3, 1, 1)
    #     pred_img = (pred_img-self.mean) / self.std
    #     gt_img = (gt_img-self.mean) / self.std
    #     if self.resize:
    #         pred_img = self.transform(pred_img, mode='bilinear', size=(224, 224), align_corners=False)
    #         gt_img = self.transform(gt_img, mode='bilinear', size=(224, 224), align_corners=False)
    #     loss = 0.0
    #     x = pred_img
    #     y = gt_img
    #     for i, block in enumerate(self.blocks):
    #         x = block(x)
    #         y = block(y)
    #         if i in feature_layers:
    #             loss += torch.nn.functional.l1_loss(x, y)
    #         if i in style_layers:
    #             act_x = x.reshape(x.shape[0], x.shape[1], -1)
    #             act_y = y.reshape(y.shape[0], y.shape[1], -1)
    #             gram_x = act_x @ act_x.permute(0, 2, 1)
    #             gram_y = act_y @ act_y.permute(0, 2, 1)
    #             loss += torch.nn.functional.l1_loss(gram_x, gram_y)
    #     return {'img_loss': loss}

    # def forward(self, fake, real, feature_layers=(0, 1, 2, 3), style_layers=()):
    #     # pred_img = dataio.lin2img(model_output['model_out'], self.image_resolution)
    #     # gt_img = dataio.lin2img(gt['img'], self.image_resolution)
    #     if self.mask is not None:
    #         fake= fake * self.mask
    #         real = real * self.mask
    #     if fake.shape[1] != 3:
    #         fake = fake.repeat(1, 3, 1, 1)
    #         real = real.repeat(1, 3, 1, 1)
    #     fake = (pred_img-self.mean) / self.std
    #     gt_img = (gt_img-self.mean) / self.std
    #     if self.resize:
    #         pred_img = self.transform(pred_img, mode='bilinear', size=(224, 224), align_corners=False)
    #         gt_img = self.transform(gt_img, mode='bilinear', size=(224, 224), align_corners=False)
    #     loss = 0.0
    #     x = pred_img
    #     y = gt_img
    #     for i, block in enumerate(self.blocks):
    #         x = block(x)
    #         y = block(y)
    #         if i in feature_layers:
    #             loss += torch.nn.functional.l1_loss(x, y)
    #         if i in style_layers:
    #             act_x = x.reshape(x.shape[0], x.shape[1], -1)
    #             act_y = y.reshape(y.shape[0], y.shape[1], -1)
    #             gram_x = act_x @ act_x.permute(0, 2, 1)
    #             gram_y = act_y @ act_y.permute(0, 2, 1)
    #             loss += torch.nn.functional.l1_loss(gram_x, gram_y)
    #     return {'img_loss': loss}

import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss