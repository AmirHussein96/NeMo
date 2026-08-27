# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Generator, Iterable

import torch
from lightning import LightningModule

from nemo.core.classes.common import safe_instantiate
from nemo.core.optim import patch_flashoptim_uneven_shard_support
from nemo.utils import logging


def configure_optimizers(model: LightningModule):
    """
    Re-usable optimizer configuration function for top-level PyTorch Lightning modules in this collection.
    It sets up parameter freezing, optimizer, and LR scheduler.

    The ``model`` object is expected to have a ``model.cfg`` attribute with OmegaConf configuration.
    The following fields are expected:

    * ``optimizer`` with hydra-style ``_target_`` pointing to optimizer class, and the remaining options
        passed directly to its ``__init__`` method.

    * (optional) ``freeze_params`` with a list of regex pattern for identifying frozen parameters.

    * (optional) ``prevent_freeze_params`` with a list of regex pattern for keeping specific parameters trainable
        (overrides ``freeze_params``).

    * (optional) ``lr_scheduler`` with hydra-style ``_target_`` pointing to LR scheduler class,
        and the remaining options passed directly to its ``__init__`` method.

    Returns:
        PyTorch Lightning Trainer-compatible dict with structure::

            {
                "optimizer": <optimizer>,
                "lr_scheduler": {"scheduler": <lr_scheduler>, "interval": "step", "frequency": 1}
            }

    """
    assert hasattr(model, "cfg"), "Expected `model.cfg` attribute to exist."
    assert "optimizer" in model.cfg, "Expected `model.cfg` to contain 'optimizer' configuration."
    parameters = freeze_and_subset(
        model.named_parameters(),
        exclude_patterns=model.cfg.get("freeze_params", []),
        keep_patterns=model.cfg.get("prevent_freeze_params", []),
    )
    optimizer = safe_instantiate(model.cfg.optimizer, parameters, _convert_='all')
    patch_flashoptim_uneven_shard_support(optimizer)

    # Defense-in-depth: verify no `requires_grad=False` parameter ended up inside any
    # optimizer param group (which would make it eligible for weight decay / momentum-driven
    # updates despite being "frozen"). Independent of `freeze_and_subset`'s own logic, so it
    # catches a regression there, a bad `prevent_freeze_params` override, or a future refactor
    # that accidentally reintroduces frozen params into the optimizer.
    assert_frozen_params_excluded_from_optimizer(model, optimizer)

    ans = {"optimizer": optimizer}
    if "lr_scheduler" in model.cfg:
        lr_scheduler = safe_instantiate(model.cfg.lr_scheduler, optimizer)
        ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    return ans


def configure_optimizers_exclude_norm_from_wd(model: LightningModule):
    """
    Advanced optimizer configuration function for top-level PyTorch Lightning modules.

    This function sets up parameter freezing and instantiates the optimizer and LR scheduler,
    but specifically separates parameters into two groups:
      1. Standard weights: Receive the configured weight decay.
      2. Biases and Normalization layers (e.g., LayerNorm): Receive 0.0 weight decay to improve
         mixed precision stability during training.

    The ``model`` object is expected to have a ``model.cfg`` attribute with OmegaConf configuration.
    The following fields are expected:

    * ``optimizer`` with hydra-style ``_target_`` pointing to optimizer class.
    * (optional) ``freeze_params`` with a list of regex patterns for identifying frozen parameters.
    * (optional) ``prevent_freeze_params`` with a list of regex patterns for keeping specific parameters trainable.
    * (optional) ``lr_scheduler`` with hydra-style ``_target_`` pointing to LR scheduler class.

    Returns:
        PyTorch Lightning Trainer-compatible dict with structure::

            {
                "optimizer": <optimizer>,
                "lr_scheduler": {"scheduler": <lr_scheduler>, "interval": "step", "frequency": 1}
            }
    """
    assert hasattr(model, "cfg"), "Expected `model.cfg` attribute to exist."
    assert "optimizer" in model.cfg, "Expected `model.cfg` to contain 'optimizer' configuration."

    # 1. Identify trainable parameters using the standard freezing logic
    trainable_params_gen = freeze_and_subset(
        model.named_parameters(),
        exclude_patterns=model.cfg.get("freeze_params", []),
        keep_patterns=model.cfg.get("prevent_freeze_params", []),
    )

    # freeze_and_subset yields parameters, but we need to track names to separate by norm/bias.
    # So we get the set of id(param) that are trainable to filter named_parameters.
    trainable_param_ids = {id(p) for p in trainable_params_gen}

    # 2. Identify layers for Weight Decay exclusion
    no_decay_keywords = ["bias", "norm", "layernorm"]

    decay_group = []
    no_decay_group = []
    no_decay_names = []
    total_trainable_layers = 0

    for name, param in model.named_parameters():
        if id(param) in trainable_param_ids:
            total_trainable_layers += 1
            if any(nd in name.lower() for nd in no_decay_keywords):
                no_decay_group.append(param)
                no_decay_names.append(name)
            else:
                decay_group.append(param)

    # Logging audit trail
    logging.info("=" * 70)
    logging.info("OPTIMIZER STRATEGY: Mixed Precision Stability")
    logging.info(f"Total Trainable Layers: {total_trainable_layers}")
    logging.info("-" * 70)
    logging.info(
        f"REGULARIZATION: Applying weight_decay={model.cfg.optimizer.get('weight_decay', 0.1)} "
        f"to {len(decay_group)} weight layers."
    )
    logging.info(f"STABILITY: Excluding {len(no_decay_names)} Normalization and Bias layers from weight decay.")

    for n in no_decay_names[:10]:
        logging.info(f"  [WD=0.0] -> {n}")
    if len(no_decay_names) > 10:
        logging.info(f"  ... (+ {len(no_decay_names) - 10} additional normalization/bias layers)")
    logging.info("=" * 70)

    # 3. Parameter Grouping
    # Note: We must exclude 'weight_decay' from the main config so we can apply it per-group
    # Hydra's instantiate will fail if we pass grouped dicts to the main positional argument
    # but weight_decay is also defined in model.cfg.optimizer.
    base_wd = model.cfg.optimizer.get("weight_decay", 0.01)
    optim_groups = [
        {"params": decay_group, "weight_decay": base_wd},
        {"params": no_decay_group, "weight_decay": 0.0},
    ]

    # 4. Instantiate via Hydra
    optimizer = safe_instantiate(model.cfg.optimizer, optim_groups, _convert_='all')
    patch_flashoptim_uneven_shard_support(optimizer)

    ans = {"optimizer": optimizer}
    if "lr_scheduler" in model.cfg:
        lr_scheduler = safe_instantiate(model.cfg.lr_scheduler, optimizer)
        ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

    return ans


def freeze_and_subset(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    exclude_patterns: list[str],
    keep_patterns: list[str] = None,
) -> Generator[torch.nn.Parameter, None, None]:
    """
    Utility used to freeze select model parameters, and skip them for the purpose
    of initializing an optimizer's parameter group.

    Args:
        named_parameters: The output of `torch.nn.Module.named_parameters()`
        exclude_patterns: A list of regex patterns matching parameter names to be frozen
            and excluded from optimization.
        keep_patterns: A list of regex patterns matching parameter names to be trained.
            This list overrides all matches to `exclude_patterns`.

    Returns:
        A generator over parameters, equivalent to calling `torch.nn.Module.parameters()`,
            that will be passed to the optimizer and trained.

    Example:

        >>> model = MyModel()
        ... # freeze all LLM parameters in "model.llm"
        ... params = freeze_and_subset(model.named_parameters(), [r'^llm\\.\\..+$'])
        ... optimizer = torch.optim.AdamW(params, lr=1e-3)

    """
    exclude_counter = {p: 0 for p in exclude_patterns}

    if not keep_patterns:
        keep_counter = {}

        def _must_keep(_) -> bool:
            return False

    else:
        keep_counter = {p: 0 for p in keep_patterns}
        compiled_keep_patterns = [re.compile(p) for p in keep_patterns]

        def _must_keep(name: str) -> bool:
            for p in compiled_keep_patterns:
                if p.match(name) is not None:
                    keep_counter[p.pattern] += 1
                    return True
            return False

    compiled_exclude_patterns = [re.compile(p) for p in exclude_patterns]

    def _exclude(name: str) -> bool:
        for p in compiled_exclude_patterns:
            if p.match(name) is not None:
                exclude_counter[p.pattern] += 1
                return True
        return False

    trainable, nontrainable = 0, 0
    for name, param in named_parameters:
        # Honor module-level freezing (e.g. ConformerMultiLayerFeatureExtractor freezes tail
        # layers in its __init__). Without this guard, a param with ``requires_grad=False`` that
        # no exclude regex matches would still be yielded into the optimizer — the optimizer
        # would then synthesize empty state for it at DCP load, causing "Missing key in
        # checkpoint state_dict" since the saved checkpoint has no state for never-trained params.
        if not param.requires_grad:
            nontrainable += param.numel()
            continue
        discard = False
        if _exclude(name) and not _must_keep(name):
            param.requires_grad = False
            discard = True
        if not discard:
            yield param
            trainable += param.numel()
        else:
            nontrainable += param.numel()
    total = trainable + nontrainable

    logging.info(f"Parameters | trainable={trainable} ({trainable / total:.2%}) | total={total}")

    if unused_excluded_patterns := [k for k, v in exclude_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_excluded_patterns) + "']"
        logging.warning(f"Parameter freezing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")

    if unused_keep_patterns := [k for k, v in keep_counter.items() if v == 0]:
        msg = "['" + "', '".join(unused_keep_patterns) + "']"
        logging.warning(f"Parameter freeze-preventing patterns UNMATCHED against any parameter: {msg} (bad regexp?)")


def is_frozen(module: torch.nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())


# ---------------------------------------------------------------------------
# Frozen-parameter integrity checks.
#
# `freeze_and_subset` already correctly excludes every `requires_grad=False` parameter
# from the optimizer's param groups, so those parameters should never be touched by
# `optimizer.step()` (no gradient descent, no weight decay, no momentum). The helpers
# below add a second, independent line of defense: a structural check that no frozen
# param leaked into the optimizer despite that logic, and a value-level fingerprint
# check that proves frozen params truly stayed bit-for-bit unchanged across training --
# regardless of *how* an unwanted update might occur (a future refactor of the freeze
# logic, an unrelated callback writing into `state_dict()`, a DDP/bucket-view aliasing
# issue, etc.). Catching drift here, at checkpoint-save time, is far cheaper than
# discovering it later via degraded downstream eval metrics.
# ---------------------------------------------------------------------------


def assert_frozen_params_excluded_from_optimizer(model: LightningModule, optimizer: torch.optim.Optimizer) -> None:
    """
    Raise if any `requires_grad=False` parameter is present in any of the
    optimizer's param groups. A frozen parameter has no gradient, so it should
    never have been added to a group in the first place.
    """
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    leaked = [name for name, p in model.named_parameters() if not p.requires_grad and id(p) in optimizer_param_ids]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} frozen parameter(s) (requires_grad=False) were found inside the "
            f"optimizer's param groups -- they would be subject to weight decay / momentum "
            f"updates despite being marked frozen. First few: {leaked[:10]}. This indicates a "
            f"bug in freeze_and_subset or its caller."
        )


def _tensor_fingerprint(t: torch.Tensor) -> int:
    """
    Cheap, exact (bit-level) fingerprint of a tensor's current values. Works
    uniformly across dtypes (bf16/fp16/fp32/...) by hashing the raw byte
    representation, so it can't be fooled by NaN/Inf-related equality quirks
    and needs no dtype-specific handling. Not cryptographically collision-proof,
    but astronomically unlikely to collide for real (non-adversarial) model
    weight drift, and O(1) memory per tensor (no full clone retained).
    """
    with torch.no_grad():
        byte_view = t.detach().reshape(-1).contiguous().cpu().view(torch.uint8).to(torch.int64)
        n = byte_view.numel()
        # Position-weighted sum so a byte-order permutation can't hide a change.
        weights = torch.arange(1, n + 1, dtype=torch.int64)
        return int((byte_view * weights).sum().item())


def snapshot_frozen_param_fingerprints(model: LightningModule) -> dict[str, int]:
    """Return {param_name: fingerprint} for every currently-frozen parameter."""
    return {name: _tensor_fingerprint(p) for name, p in model.named_parameters() if not p.requires_grad}


def verify_frozen_params_unchanged(model: LightningModule, snapshot: dict[str, int]) -> list[str]:
    """
    Recompute fingerprints for every parameter name in ``snapshot`` and return
    the list of names whose fingerprint no longer matches (i.e. drifted).
    Empty list means all frozen parameters are still bit-for-bit unchanged.
    Parameters that no longer exist or are no longer frozen are skipped with a
    warning rather than silently ignored, so an unexpected architecture change
    doesn't quietly disable the check.
    """
    current = dict(model.named_parameters())
    drifted: list[str] = []
    for name, old_fp in snapshot.items():
        p = current.get(name)
        if p is None:
            logging.warning(f"[frozen-param-check] '{name}' no longer exists on the model; skipping.")
            continue
        if p.requires_grad:
            logging.warning(f"[frozen-param-check] '{name}' is no longer frozen (requires_grad=True); skipping.")
            continue
        if _tensor_fingerprint(p) != old_fp:
            drifted.append(name)
    return drifted
