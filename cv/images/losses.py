from torch.nn import _reduction as _Reduction
import logging

logger = logging.getLogger()
# logger.setLevel(logging.INFO)


class _Loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


def adj_mse_loss(input, target, size_average=None, reduce=None, reduction='mean', classes_mask=None):
    if not (input.size() == target.size()):
        logger.warning("Using a target size ({}) that is different to the input size ({}). "
                       "This will likely lead to incorrect results due to broadcasting. "
                       "Please ensure they have the same size.".format(target.size(), input.size()),
                       stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if input.requires_grad:
        is_0, is_1, is_2, is_3, is_4 = target.eq(0).type(input.type()), target.eq(1).type(input.type()), target.eq(
            2).type(input.type()), \
                                       target.eq(3).type(input.type()), target.eq(4).type(input.type())

        pen_coef = 1.3
        wh_1 = target.new_full(target.shape, 1, device=device)
        wh_2 = target.new_full(target.shape, pen_coef, device=device)

        bo1 = target.new_full(target.shape, .7, device=device)
        bo2 = target.new_full(target.shape, 1.5, device=device)
        bo3 = target.new_full(target.shape, 2.5, device=device)
        bo4 = target.new_full(target.shape, 3.25, device=device)

        ret = torch.mul(is_0, torch.where(input > bo1, wh_2, wh_1) * (input - target) ** 2) + \
              torch.mul(is_1,
                        torch.where(input > bo2, wh_2, torch.where(input < bo1, wh_2, wh_1)) * (input - target) ** 2) + \
              torch.mul(is_2,
                        torch.where(input > bo3, wh_2, torch.where(input < bo2, wh_2, wh_1)) * (input - target) ** 2) + \
              torch.mul(is_3,
                        torch.where(input > bo4, wh_2, torch.where(input < bo3, wh_2, wh_1)) * (input - target) ** 2) + \
              torch.mul(is_4, torch.where(input < bo4, wh_2, wh_1) * (input - target) ** 2)

        #               torch.mul(is_0, torch.where(input > bo1, wh_2, wh_1) * (torch.max(input.new_full(input.shape, -0.5, device=device), input) - target) ** 2) +\
        #               torch.mul(is_4, torch.where(input < bo4, wh_2, wh_1) * (torch.min(input.new_full(target.shape, 4, device=device), input) - target) ** 2)
        # v1:           ret = torch.mul(is_0, torch.where(input > bo1, wh_2, wh_1) * (torch.max(input.new_full(input.shape, 0.01, device=device), input) - target) ** 2) +\
        #               torch.mul(is_1, torch.where(input > bo2, wh_2, torch.where(input < bo1, wh_2, wh_1)) * (input - target) ** 2) +\
        #               torch.mul(is_2, torch.where(input > bo3, wh_2, torch.where(input < bo2, wh_2, wh_1)) * (input - target) ** 2) +\
        #               torch.mul(is_3, torch.where(input > bo4, wh_2, torch.where(input < bo3, wh_2, wh_1)) * (input - target) ** 2) +\
        #               torch.mul(is_4, torch.where(input < bo4, wh_2, wh_1) * (torch.min(input.new_full(target.shape, 3.7, device=device), input) - target) ** 2)

        #         ret = torch.sum(torch.mul(classes_mask.index_select(0, labels), c.transpose(0, 1)), dim=1)

        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(preds, labels)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret


class AdjMSELoss(_Loss):
    __constants__ = ['n_classes']

    def __init__(self, size_average=None, reduce=None, reduction='mean', n_classes=5):
        super(AdjMSELoss, self).__init__(size_average, reduce, reduction)
        self.mask = torch.eye(n_classes, device=device)  # TODO set N_CLASSES

    #     @weak_script_method
    def forward(self, input, target):
        return adj_mse_loss(input, target, reduction=self.reduction, classes_mask=self.mask)
