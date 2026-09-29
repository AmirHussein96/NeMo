#!/usr/bin/env python3
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
"""Uniform weight averaging ("model soup") of N VoiceTranslate Lightning checkpoints.

Each input .ckpt is a PyTorch-Lightning checkpoint whose model weights live
under the top-level ``state_dict`` key. Only ``state_dict`` is averaged;
optimizer states are ignored (never paged in via mmap), so the output file is
a fraction of the ~31 GB inputs.

Averaging rules per tensor key:
  * floating-point tensors  → mean over all inputs, accumulated in float32,
    cast back to the reference dtype.
  * non-floating tensors (int/bool buffers such as ``num_batches_tracked``,
    position ids, frozen index buffers) → taken from the first checkpoint.
    These are architecture constants; we assert they are byte-equal across
    all checkpoints and fail loudly otherwise.

Checkpoints are processed one at a time (streaming accumulation), so peak
memory usage is roughly 2x a single checkpoint's size regardless of how many
inputs are averaged, instead of Nx from loading everything at once.

Usage
-----
::

    python checkpoint_averaging.py \\
        --out /path/to/averaged.ckpt \\
        /path/to/step=18001.ckpt \\
        /path/to/step=29003.ckpt \\
        /path/to/step=51005.ckpt

The output file can be passed directly to ``voicetranslate_eval.py``
via ``--checkpoints /path/to/averaged.ckpt``.
"""
import argparse
import gc
import sys

import torch


def load_state_dict(path: str) -> dict:
    """Load only the 'state_dict' key from a Lightning checkpoint, mmap when possible."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError) as e:
        print(f"[avg] mmap load failed for {path} ({e}); falling back to full load", flush=True)
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in obj:
        raise KeyError(
            f"{path} has no top-level 'state_dict' key; keys={list(obj.keys())[:10]}"
        )
    return obj["state_dict"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output averaged .ckpt path")
    ap.add_argument("ckpts", nargs="+", help="Input .ckpt paths (≥ 2 required)")
    args = ap.parse_args()

    if len(args.ckpts) < 2:
        ap.error("At least 2 checkpoints are required for averaging.")

    n = len(args.ckpts)
    print(f"[avg] averaging {n} checkpoints (streaming, one at a time):", flush=True)
    for c in args.ckpts:
        print(f"       {c}", flush=True)

    ref_keys = None
    ref_dtypes: dict = {}
    acc: dict = {}  # float32 running sum for float tensors, reference tensor for non-float
    n_float = n_kept = 0

    for i, path in enumerate(args.ckpts):
        sd = load_state_dict(path)

        if ref_keys is None:
            ref_keys = list(sd.keys())
            print(f"[avg] reference state_dict has {len(ref_keys)} tensors", flush=True)
        elif set(sd.keys()) != set(ref_keys):
            only_ref = sorted(set(ref_keys) - set(sd.keys()))[:10]
            only_i = sorted(set(sd.keys()) - set(ref_keys))[:10]
            raise KeyError(
                f"Key mismatch between ckpt[0] and ckpt[{i}] ({path}):\n"
                f"  only in ckpt[0]:    {only_ref}\n"
                f"  only in ckpt[{i}]: {only_i}"
            )

        for k in ref_keys:
            t = sd[k]
            if i == 0:
                ref_dtypes[k] = t.dtype
                if torch.is_floating_point(t):
                    acc[k] = t.to(torch.float32).clone()
                    n_float += 1
                else:
                    acc[k] = t.clone()
                    n_kept += 1
            else:
                if torch.is_floating_point(t):
                    if t.shape != acc[k].shape:
                        raise ValueError(
                            f"Shape mismatch at key '{k}': {acc[k].shape} vs {t.shape}"
                        )
                    acc[k] += t.to(torch.float32)
                else:
                    if not torch.equal(acc[k], t):
                        raise ValueError(
                            f"Non-float buffer '{k}' differs between ckpt[0] and ckpt[{i}]; "
                            f"cannot safely average. dtype={t.dtype}"
                        )

        del sd
        gc.collect()
        print(f"[avg] processed {i + 1}/{n}: {path}", flush=True)

    out: dict = {}
    for k in ref_keys:
        if ref_dtypes[k].is_floating_point:
            out[k] = (acc[k] / n).to(ref_dtypes[k])
        else:
            out[k] = acc[k]

    print(
        f"[avg] averaged {n_float} float tensors; kept {n_kept} non-float buffers",
        flush=True,
    )
    print(f"[avg] writing → {args.out}", flush=True)
    torch.save({"state_dict": out}, args.out)
    print("[avg] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
