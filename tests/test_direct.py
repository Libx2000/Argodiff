import torch

from diffargo.diffusion import masked_weighted_mse
from diffargo.model import ConditionalUNet
from train_direct import direct_prediction, physical_sums, score_from_sums


def test_direct_persistence_anchor_and_gradient():
    torch.manual_seed(3)
    model = ConditionalUNet(
        target_channels=4, condition_channels=7, base_channels=8,
        channel_multipliers=(1, 2), time_embedding_dim=32,
        dropout=0.0, attention_at_bottleneck=False,
    )
    torch.nn.init.zeros_(model.output_conv.conv.bias)
    condition = torch.randn(1, 7, 16, 24)
    target = torch.randn(1, 4, 16, 24)
    baseline = torch.tensor([0.1, -0.2, 0.3, -0.4]).view(1, 4, 1, 1)
    prediction = direct_prediction(model, condition, 4, baseline)
    torch.testing.assert_close(prediction, baseline.expand_as(prediction))
    mask = torch.ones_like(target)
    latitude_weight = torch.ones(1, 1, 16, 1)
    loss = masked_weighted_mse(prediction, target, mask, latitude_weight)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output_conv.conv.weight.grad is not None


def test_physical_score_uses_same_mask_and_prefers_better_forecast():
    target = torch.ones(1, 4, 2, 2)
    prediction = target * 0.5
    mask = torch.ones_like(target)
    mask[:, :, 0, 0] = 0
    weight = torch.ones(1, 1, 2, 1)
    mean = torch.zeros(1, 4, 1, 1)
    std = torch.ones(1, 4, 1, 1)
    sums = physical_sums(prediction, target, mask, weight, mean, std)
    score = score_from_sums(sums)
    assert score["temperature_rmse"] == 0.5
    assert score["temperature_persistence_rmse"] == 1.0
    assert score["salinity_rmse"] == 0.5
    assert score["selection_score"] == 0.5
