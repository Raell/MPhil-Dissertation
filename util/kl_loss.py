import torch


def KL_Loss(y_pred, classes):
    y_joint = torch.reshape(y_pred, (-1, y_pred.shape[1] * classes))
    # y_class = torch.unsqueeze(torch.sum(y_pred, 1), 1)
    # y_domain = torch.unsqueeze(torch.sum(y_pred, 2), -1)
    #
    # y_ind_joint = torch.reshape((y_domain * y_class), (-1, y_pred.shape[1] * classes))
    #
    # return torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(
    #     torch.log(y_joint),
    #     torch.log(y_ind_joint)
    # )

    y_class = torch.unsqueeze(torch.logsumexp(y_pred, 1), 1)
    y_domain = torch.unsqueeze(torch.logsumexp(y_pred, 2), -1)

    y_ind_joint = torch.reshape((y_domain + y_class), (-1, y_pred.shape[1] * classes))

    return torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(
        y_joint,
        y_ind_joint
    )
