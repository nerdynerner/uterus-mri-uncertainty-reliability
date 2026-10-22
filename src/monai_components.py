"""Unexecuted MONAI components, NOT a validated end-to-end training pipeline.

Runtime integration is blocked. These components intentionally require an
audited manifest and separately validated native-space preprocessing/inversion.
"""
import itertools
import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss


def model(dropout=0.):
    return UNet(spatial_dims=2, in_channels=1, out_channels=1,
                channels=(16,32,64,128), strides=(2,2,2),
                num_res_units=2, norm='INSTANCE', dropout=dropout)


def train_epoch(net, loader, optimizer, device):
    """Loader supplies patient-balanced, verified preprocessed 2D slices.

    Caller must keep validation/calibration/test out of this loader.
    """
    net.train()
    objective = DiceCELoss(sigmoid=True, squared_pred=False)
    total, count = 0., 0
    for batch in loader:
        x = batch['image'].to(device)
        y = batch['label'].float().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())*len(x)
        count += len(x)
    if not count:
        raise ValueError('Empty training loader')
    return total/count


@torch.inference_mode()
def stochastic_probability(net, images, passes=30):
    """Only dropout layers are stochastic; leave normalization in eval mode."""
    net.eval()
    kinds=(torch.nn.Dropout,torch.nn.Dropout2d,torch.nn.Dropout3d)
    layers=[m for m in net.modules() if isinstance(m,kinds)]
    if not layers or not any(m.p > 0 for m in layers):
        raise ValueError('MC dropout requires a model trained with nonzero dropout')
    try:
        for layer in layers:
            layer.train()
        return torch.stack([torch.sigmoid(net(images)) for _ in range(passes)])
    finally:
        net.eval()


@torch.inference_mode()
def ensemble_probability(networks, images):
    """Networks must have identical training data and independent seeds."""
    if len(networks)<2:
        raise ValueError('At least two independently trained models required')
    for net in networks:
        net.eval()
    return torch.stack([torch.sigmoid(net(images)) for net in networks])


def volume_pairwise_disagreement(member_probabilities):
    """Input [members,...full native-volume voxels], one binary structure.

    Caller must reconstruct full volumes and invert preprocessing first.
    """
    masks=(member_probabilities>=.5).flatten(start_dim=1)
    values=[]
    for a,b in itertools.combinations(masks,2):
        denominator=a.sum()+b.sum()
        dice=1. if denominator==0 else float(2*(a&b).sum()/denominator)
        values.append(1.-dice)
    if not values:
        raise ValueError('At least two members required')
    return sum(values)/len(values)
