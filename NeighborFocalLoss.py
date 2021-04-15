import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class NeighborFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True, loss_filter_size=5):
        super(NeighborFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.loss_filter_size = loss_filter_size

        self.loss_filter_5x5 = Variable(torch.tensor([[[[1., 4., 6., 4., 1.],
                                                        [4., 16., 24., 16., 4.],
                                                        [6., 24., 36., 24., 6.],
                                                        [4., 16., 24., 16., 4.],
                                                        [1., 4., 6., 4., 1.]]]]),
                                        requires_grad=False)
        self.loss_filter_3x3 = Variable(torch.tensor([[[[1., 2., 1.],
                                                        [2., 4., 2.],
                                                        [1., 2., 1.]]]]),
                                        requires_grad=False)

        if self.loss_filter_size == 3:
            self.loss_filter = self.loss_filter_3x3
        elif self.loss_filter_size == 5:
            self.loss_filter = self.loss_filter_5x5

        self.loss_filter = self.loss_filter / torch.sum(self.loss_filter)
        print(self.loss_filter)
        if torch.cuda.is_available():
            self.loss_filter = self.loss_filter.cuda()

    def nei_loss(self, pt):
        return F.conv2d(pt, self.loss_filter, padding=self.loss_filter_size // 2)

    def nchw2nhwc(self, m):
        m = m.view(m.size(0), m.size(1), -1)  # N, C, H, W ==> N, C, H*W
        m = m.transpose(1, 2)  # N, C, H*W ==> N, H*W, C
        m = m.contiguous().view(-1, m.size(2))  # N, C, H*W ==> N*H*W, C

        return m

    def forward(self, preds, target):

        if preds.dim() > 2:
            preds = self.nchw2nhwc(preds)

        target = F.softmax(target, dim=1)
        target = torch.argmax(target, dim=1)
        target = target.long()

        with torch.no_grad():
            pt_0 = preds[:, 0].reshape(target.shape)
            pt_1 = preds[:, 1].reshape(target.shape)

            pt = torch.empty_like(pt_0)
            pt[target == 0] = pt_0[target == 0]
            pt[target == 1] = pt_1[target == 1]
            pt = torch.unsqueeze(pt, 1)
            pt = Variable(self.nei_loss(pt))

            pt = torch.squeeze(self.nchw2nhwc(pt), 1)

        target = target.view(-1, 1)

        logpt_1 = F.log_softmax(preds)
        logpt_1 = logpt_1.gather(1, target)

        logpt_1 = logpt_1.view(-1)

        if self.alpha is not None:
            if self.alpha.type() != preds.data.type():
                self.alpha = self.alpha.type_as(preds.data)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt_1 = logpt_1 * Variable(at)

        loss_1 = -1 * (1 - pt) ** self.gamma * logpt_1

        if self.size_average:
            return loss_1.mean()
        else:
            return loss_1.sum()

