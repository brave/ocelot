import torch

from methods.trl_vision_safe import concatenated_vision_kwargs


def _patch_count(grid: torch.Tensor) -> int:
    return int(grid.prod(-1).sum().item())


def test_concatenated_vision_dup_two_images_single_example():
    """Regression: batch_size=1 with two images must duplicate both pv and grid."""
    patches_per_image = 1024
    pv = torch.randn(patches_per_image * 2, 64)
    grid = torch.tensor([[1, 32, 32], [1, 32, 32]], dtype=torch.int64)
    batch = {"prompt_pixel_values": pv, "prompt_image_grid_thw": grid}

    kwargs = concatenated_vision_kwargs(batch)

    assert kwargs["pixel_values"].shape[0] == pv.shape[0] * 2
    assert kwargs["image_grid_thw"].shape[0] == grid.shape[0] * 2
    assert _patch_count(kwargs["image_grid_thw"]) == kwargs["pixel_values"].shape[0]


def test_concatenated_vision_dup_keeps_grid_and_patches_aligned():
    micro_batch = 2
    pv = torch.randn(2048, 64)
    grid = torch.tensor([[1, 32, 32], [1, 32, 32]], dtype=torch.int64)
    batch = {"prompt_pixel_values": pv, "prompt_image_grid_thw": grid}

    kwargs = concatenated_vision_kwargs(batch)

    assert kwargs["pixel_values"].shape[0] == pv.shape[0] * 2
    assert kwargs["image_grid_thw"].shape[0] == grid.shape[0] * 2
    assert _patch_count(kwargs["image_grid_thw"]) == kwargs["pixel_values"].shape[0]
    assert micro_batch == 2  # documents expected batch layout for this fixture
