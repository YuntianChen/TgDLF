import torch


def ENNLoss(output, target):  # calculate loss, type:list
    with torch.no_grad():
        # output = output.mean(0)
        loss = torch.mean((output - target).abs(), 0)
    return loss.tolist()
