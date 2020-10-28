import torch


class Lamuda: #calculate lamuda type:float
    L = 1

    def __init__(self, y, ensemble, error_per):
        self.dstb = 1
        self.NE = ensemble
        self.y = y
        self.error_per = error_per
        self.update()

    def update(self):
        with torch.no_grad():
            tmp = torch.stack([self.y]*self.NE).reshape(self.NE, -1).t()
            Cd_half = torch.eye(tmp.shape[0]).float().cuda() * self.error_per
            # 4 is the opposite number of the Coefficient of Variation.
            # error = torch.mm(Cd_half, torch.randn(tmp.shape).float().cuda())
            # dstb_y = tmp + error * tmp + 3 * error
            dstb_y = tmp + tmp * torch.mm(Cd_half, torch.randn(tmp.shape).float().cuda()) + self.L * torch.mm(Cd_half, torch.randn(tmp.shape).float().cuda())
            # dstb_y = tmp + tmp * torch.mm(Cd_half, torch.randn(tmp.shape).float().cuda())
            self.dstb = dstb_y
        del tmp, Cd_half, dstb_y
        torch.cuda.empty_cache()

    def lamuda(self, pred):  # pred: output of ENN; dstb:
        data = pred.reshape(self.NE, -1).t()
        y = data-self.dstb
        # lamuda = torch.sum(y*y) / (self.error_per**2*data.shape[0]*self.NE*2000)
        lamuda = torch.sum(y * y) / (1 * self.error_per ** 2 * data.shape[0] * self.NE)
        del data, y
        return float(lamuda)

    def std(self, pred):
        data = pred.reshape(self.NE, -1).t()
        y = data - self.dstb
        std = torch.std((y**2).sum(0)/data.size()[0])
        del data, y
        return float(std)
