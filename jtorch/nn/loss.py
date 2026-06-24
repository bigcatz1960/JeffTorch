class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred.data - target.data
        return Tensor(np.mean(diff * diff))
