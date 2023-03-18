"""Mean Teacher model for semi-supervised image segmentation.
"""

from torch.optim.lr_scheduler import MultiStepLR

from deepdraw.binseg.engine.adabound import AdaBound
from deepdraw.binseg.models.losses import SemiLoss
from deepdraw.binseg.models.mean_teacher import mean_teacher

# config
lr = 0.0006
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
final_lr = 0.1
gamma = 1e-3
eps = 1e-8
amsbound = False
scheduler_milestones = [300]
scheduler_gamma = 0.1

weight = "/home/chao/Desktop/drive/driu/model/model_lowest_valid_loss.pth"  # path to the pretrained model

model = mean_teacher(weight)

# optimizer
optimizer = AdaBound(
    model.S_model.parameters(),
    lr=lr,
    betas=betas,
    final_lr=final_lr,
    gamma=gamma,
    eps=eps,
    weight_decay=weight_decay,
    amsbound=amsbound,
)

# criterion
criterion = SemiLoss()


# scheduler
scheduler = MultiStepLR(
    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
)
