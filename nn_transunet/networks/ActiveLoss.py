import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASS=4
# activeLOSS
class SemiActiveLoss(nn.Module):
    def __init__(self, device, alpha =1e-9, beta = 1e-1, lamda = 1e-3,apply_nonlin=None, batch_dice=False):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
    def LevelsetLoss(self, image, y_pred, kernel_size=5, smooth=1e-5):
        #The loss function of the level set method is introduced to compare the difference between the prediction results and the local and global features of the input image to promote contour learning.
        kernel = torch.ones(1, y_pred.size(1), kernel_size, kernel_size, device=self.device) / kernel_size**2
        padding = kernel_size //2 # Boundary processing for convolution operations
        lossRegion = 0.0 # The initial level set loss is 0
        y_pred_fuzzy = y_pred
        for ich in range(image.size(1)): # Operate on each channel of the input image
            target_ = image[:,ich:ich+1]# Extract the target image of the current channel from the image image
            # Calculate the center of mass of each pixel
            pcentroid_local = F.conv2d(target_ * y_pred_fuzzy + smooth, kernel, padding = padding) \
                                / F.conv2d(y_pred_fuzzy + smooth, kernel, padding = padding)
            plevel_local = target_ - pcentroid_local
            loss_local = plevel_local * plevel_local * y_pred_fuzzy
            # 计算全局质心
            pcentroid_global = torch.sum(target_ * y_pred_fuzzy, dim=(2,3),keepdim=True) \
                                / torch.sum(y_pred_fuzzy+smooth, dim=(2,3),keepdim = True)
            plevel_global = target_ - pcentroid_global
            loss_global = plevel_global * plevel_global * y_pred_fuzzy

            lossRegion += torch.sum(loss_local) + self.beta * torch.sum(loss_global)
        return lossRegion

    def GradientLoss(self, y_pred, penalty="l1"):
        # The gradient loss function is used to measure the gradient information of the predicted result. Usually as an additional function
        dD = torch.abs(y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :])
        dH = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dW = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if penalty == "l2":
            dD = dD * dD
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dD) + torch.sum(dH) + torch.sum(dW)
        return loss

    def ActiveContourLoss(self, y_true, y_pred, smooth=1e-5):
        dim = (1,2,3)
        yTrueOnehot = torch.zeros(y_true.size(0), NUM_CLASS, y_true.size(2), y_true.size(3), y_true.size(4), device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true.to(torch.int64), 1)[:,:, :]
        y_pred = y_pred[:, :, :]

        active = - torch.log(1-y_pred+smooth) * (1-yTrueOnehot) - torch.log(y_pred+smooth) * yTrueOnehot
        loss = torch.sum(active, dim = dim) / torch.sum(yTrueOnehot + y_pred - yTrueOnehot * y_pred +smooth, dim = dim)
        return torch.mean(loss)

    def forward(self, y_true, y_pred):
        y_pred = y_pred['pred_masks']
        if self.apply_nonlin is not None:
            y_pred = self.apply_nonlin(y_pred)

        shp_x = y_pred.shape
        shp_y = y_true.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(y_pred.shape, y_true.shape)]):
                # If this is the case, then gt is probably already a one-hot code.
                y_onehot = y_true
                onehot_out = y_onehot
            else:
                y_true = y_true.long()
                y_onehot = torch.zeros(shp_x)
                if y_pred.device.type == "cuda":
                    y_onehot = y_onehot.cuda(y_pred.device.index)
                y_onehot.scatter_(1, y_true, 1)
                onehot_out = y_onehot



        active = self.ActiveContourLoss(y_true, onehot_out)
        # levelset =  self.LevelsetLoss(image, onehot_out)
        # length = self.GradientLoss(onehot_out)
        # return active + self.alpha * (levelset + self.lamda * length)
        return active

 # DOUloss
class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
