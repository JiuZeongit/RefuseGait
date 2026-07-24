"""Event-prompt insertion and aspect-ratio-aware resizing."""

import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from torch.nn import functional as F


def padding_resize(x, ratios, target_h, target_w):
    """Resize samples while preserving bbox-derived aspect ratios."""
    n, h, w = x.size(0), target_h, target_w
    ratios = ratios.view(-1)

    need_w = (h * ratios).int()
    need_padding_mask = need_w < w

    pad_left = torch.where(
        need_padding_mask,
        torch.div(
            w - need_w,
            2,
            rounding_mode="floor",
        ),
        torch.zeros(
            1,
            dtype=need_w.dtype,
            device=x.device,
        ),
    )

    pad_right = torch.where(
        need_padding_mask,
        w - need_w - pad_left,
        torch.zeros(
            1,
            dtype=need_w.dtype,
            device=x.device,
        ),
    ).tolist()

    need_w = need_w.tolist()
    pad_left = pad_left.tolist()

    resized = []

    for i in range(n):
        sample = F.interpolate(
            x[i:i + 1, ...],
            (h, need_w[i]),
            mode="bilinear",
            align_corners=False,
        )

        if need_padding_mask[i]:
            sample = F.pad(
                sample,
                (pad_left[i], pad_right[i]),
            )
        else:
            sample = sample[
                ...,
                pad_left[i]:pad_left[i] + w,
            ]

        resized.append(sample)

    return torch.concat(resized, dim=0)


class InsertEventPrompt(nn.Module):
    """Insert learnable event-modality prompts after the CLS token."""

    def __init__(
        self,
        cfg,
        patch_size,
        feature_dim,
        num_heads,
    ):
        super().__init__()

        # Retained for constructor compatibility with trained models.
        self.num_heads = num_heads
        self.patch_size = (patch_size, patch_size)

        event_cfg = cfg["EventEncoder"]

        self.use_event_modality_prompts = event_cfg[
            "use_event_modality_prompts"
        ]
        self.num_event_modality_prompts = event_cfg[
            "num_event_modality_prompts"
        ]

        if self.use_event_modality_prompts:
            self.event_modality_prompts = nn.Parameter(
                torch.zeros(
                    self.num_event_modality_prompts,
                    feature_dim,
                ),
                requires_grad=True,
            )

            self._initialize_event_modality_prompts(
                self.patch_size,
                feature_dim,
            )

    def _initialize_event_modality_prompts(
        self,
        patch_size,
        prompt_dim,
    ):
        value = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, patch_size, 1)
                + prompt_dim
            )
        )

        nn.init.uniform_(
            self.event_modality_prompts.data,
            -value,
            value,
        )

    def forward(self, x, batch_size, frames_num):
        if not self.use_event_modality_prompts:
            return x

        prompts = self.event_modality_prompts.expand(
            batch_size * frames_num,
            -1,
            -1,
        ).to(x.device)

        return torch.cat(
            (
                x[:, :1, :],
                prompts,
                x[:, 1:, :],
            ),
            dim=1,
        )
