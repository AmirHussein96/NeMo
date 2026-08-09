# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
NemotronVoiceTranslateSyncV3 — timing-preserved direct-ID-mapping sync.

Compared to NemotronVoiceTranslateSyncV2 (skip structural tokens), this
version maps LLM PAD/BOS/EOS tokens within the BOS-EOS content window to
TTS PAD (id=12) rather than dropping them, so the inter-word silence timing
from the LLM is preserved in the TTS token sequence:

  LLM PAD  (id=0) → last content token (freeze — acoustic decoder sustains)
  LLM BOS  (id=1) → ignored (intermediate BOS skipped entirely)
  LLM EOS  (id=2) → TTS PAD (id=12)  — (rare mid-window EOS) → silence
  IDs 10-13       → TTS PAD (id=12)  — code fill-in-the-middle tokens
  All other IDs   → identity          — same in both vocabularies (131,068/131,072)

Effect on the TTS sequence:
  v2 (skip):  [BOS, "▁the", "▁cat", "▁sat",  PAD×window,  EOS, ...]
  v3 (timing):[BOS, "▁the", PAD, "▁cat", PAD, PAD, "▁sat", PAD×window, EOS, ...]
                                  ↑ LLM timing gap preserved as TTS silence

Tokenizer compatibility check (Riva-Translate-4B vs Nemotron-Nano-9B)
----------------------------------------------------------------------
Both tokenizers share the same 131,072-entry vocabulary with identical
(token-string, token-id) pairs for 131,068 / 131,072 entries.

Only 4 raw vocabulary entries differ (IDs 10-13) — all code fill-in-the-
middle tokens that never appear in translation output:

  ID |  LLM token   |  TTS token     |  decode() output  |  Handling
  ---+--------------+----------------+-------------------+----------
  10 |  <pad>        |  <SPECIAL_10>  |  ''  (both)       |  → TTS PAD (id=12)
  11 |  [PREFIX]     |  <SPECIAL_11>  |  ''  (both)       |  → TTS PAD (id=12)
  12 |  [MIDDLE]     |  <SPECIAL_12>  |  ''  (both)       |  → TTS PAD (id=12)
  13 |  [SUFFIX]     |  <SPECIAL_13>  |  ''  (both)       |  → TTS PAD (id=12)

Special token IDs at runtime (set via YAML: bos_token/eos_token/pad_token):
  BOS: id=1   (<s>)            — LLM and TTS: SAME
  EOS: id=2   (</s>)           — LLM and TTS: SAME
  PAD: id=12  (<SPECIAL_12>)   — TTS only (LLM has no pad_token; NeMo AutoTokenizer
                                  resolves text_pad_id via tokenizer.pad → pad_id →
                                  tokens_to_ids(pad_token="<SPECIAL_12>") = 12)

Key insight: TTS EOS (id=2) == LLM EOS (id=2). The TTS PAD token is
<SPECIAL_12> (id=12), which fills silence frames outside/before/after speech.

Sync layout (same as NemotronVoiceTranslateSync v1)
----------------------------------------------------
Following nemotron_voicechat.py, the TTS receives token IDs at the exact
frame where the LLM produced them.  Silence frames before BOS are TTS PAD.

  frame:  0 … bos_pos-1 | bos_pos | bos_pos+1 … bos_pos+N | … EOS … | …
  token:  PAD … PAD      | TTS BOS | w1  w2  …  wN          | PAD      EOS  PAD

where EOS = bos_pos + 1 + N + int((1+N) × pad_factor × eos_pct)
"""

import logging

import torch

from nemo.collections.speechlm2.models.nemotron_voicetranslate import NemotronVoiceTranslate


# ---------------------------------------------------------------------------
# Token-ID remap table for the 4 vocabulary positions that differ between
# Riva-Translate-4B (LLM) and Nemotron-Nano-9B (TTS).
#
# These are code fill-in-the-middle tokens that never appear in translation
# output.  They are mapped to TTS PAD (id=12, <SPECIAL_12>) inside the
# _build_tts_sequence method where the runtime tts_pad_id is available.
#
# The set of IDs to remap:
_LLM_REMAP_IDS: frozenset[int] = frozenset({10, 11, 12, 13})


class NemotronVoiceTranslateSyncV3(NemotronVoiceTranslate):
    """
    NemotronVoiceTranslate with voicechat-style direct token-ID sync.

    Key differences from NemotronVoiceTranslateSync (v1):
      • No text decode/re-encode round-trip — LLM content token IDs are
        passed directly to the TTS sequence (same philosophy as
        nemotron_voicechat.py feeding raw STT tokens to TTS).
      • A small static remap table handles the 4 differing special tokens
        (IDs 10-13).  All other IDs are identical in both vocabularies.
      • BOS is placed at the LLM's actual BOS frame (timing sync).
      • EOS is placed at the EAR TTS eval recipe formula position.
    """

    # Reported once at first call so the user can verify.
    _mapping_reported: bool = False

    def _build_tts_sequence(
        self,
        gen_text: "torch.Tensor",
        tts_tok,
        tts_pad_id: int,
        pad_factor: int = 5,
        eos_pct: float = 0.7,
    ) -> "tuple[torch.Tensor, int, list[int]]":
        """
        Timing-preserving vocab bridge using direct token-ID passthrough.

        Instead of decoding LLM token IDs to text and re-tokenising (v1), this
        method copies LLM content IDs directly into the TTS sequence.

        Unlike v2 (which skips LLM structural tokens), this version maps them
        to TTS PAD (id=12) so inter-word silence timing from the LLM is
        preserved:
          LLM PAD (id=0) → TTS PAD (id=12)  — inter-word silence / delay
          LLM BOS (id=1) → TTS BOS (id=1)   — passes through as TTS BOS
          LLM EOS (id=2) → TTS PAD (id=12)  — (rare mid-window) → silence
          IDs 10-13      → TTS PAD (id=12)  — code fill tokens
          All other IDs  → identity (131,068/131,072 vocab entries match)

        Runtime special-token IDs (from YAML bos/eos/pad_token settings):
          TTS BOS  = tts_tok.bos_id  = 1   (<s>)
          TTS EOS  = tts_tok.eos_id  = 2   (</s>)  — same as LLM EOS
          TTS PAD  = tts_pad_id      = 12  (<SPECIAL_12>)

        Layout (same as NemotronVoiceTranslateSync v1):

            PAD(12)… | TTS_BOS(1) | w1 … wN | PAD(12)×(seg×pad_factor) | TTS_EOS(2) | PAD(12)…
            ← bos_pos →↑ bos_pos                                          ↑ eos_pos

        EOS position = bos_pos + 1 + N + int((1+N) × pad_factor × eos_pct)

        Parameters
        ----------
        gen_text   : (B, T) LLM token IDs — full autoregressive output.
        tts_tok    : EAR TTS NeMo tokenizer (for bos_id / eos_id).
        tts_pad_id : TTS pad token id = 12 (tts_model.text_pad_id).
        pad_factor : PAD window multiplier (default 5).
        eos_pct    : EOS position within the PAD window (default 0.7 = 70%).

        Returns
        -------
        tts_seq       : (B, T_tts) TTS input token tensor.
        T_tts         : int — length of the TTS sequence.
        eos_positions : list[int] — per-sample TTS EOS frame indices.
        """
        B, T = gen_text.shape

        # LLM special IDs
        llm_bos_id = self.text_bos_id   # 1
        llm_eos_id = self.text_eos_id   # 2
        llm_pad_id = self.text_pad_id   # 0

        # TTS special IDs (resolved from YAML bos/eos/pad_token settings)
        tts_bos_id = tts_tok.bos_id     # 1  (same as LLM BOS)
        tts_eos_id = tts_tok.eos_id     # 2  (same as LLM EOS — both are </s>)
        # tts_pad_id is passed as parameter = 12 (<SPECIAL_12>)

        # IDs to exclude from content (LLM structural tokens)
        llm_special = {llm_pad_id, llm_bos_id, llm_eos_id}

        # Runtime remap: IDs 10-13 are code fill-in-the-middle tokens that
        # differ between LLM and TTS vocabularies.  Map them to TTS PAD (12).
        _REMAP = {10: tts_pad_id, 11: tts_pad_id, 12: tts_pad_id, 13: tts_pad_id}

        # One-time mapping report
        if not NemotronVoiceTranslateSyncV3._mapping_reported:
            NemotronVoiceTranslateSyncV3._mapping_reported = True
            logging.info(
                "NemotronVoiceTranslateSyncV3 — token ID mapping:\n"
                "  LLM BOS (id=%-4d)  → TTS BOS (id=%d)  [same]\n"
                "  LLM EOS (id=%-4d)  → TTS EOS (id=%d)  [DIFFERENT: marks content end; TTS EOS placed at formula pos]\n"
                "  LLM PAD (id=%-4d)  → TTS PAD (id=%d)  [same]\n"
                "  Differing special tokens (code fill-in-the-middle, never in translation output):\n"
                "    LLM id=10 (<pad>)    → TTS PAD (id=%d)\n"
                "    LLM id=11 ([PREFIX]) → TTS PAD (id=%d)\n"
                "    LLM id=12 ([MIDDLE]) → TTS PAD (id=%d)\n"
                "    LLM id=13 ([SUFFIX]) → TTS PAD (id=%d)\n"
                "  All other IDs (131,068/131,072): identical in both vocabularies.",
                llm_bos_id, tts_bos_id,
                llm_eos_id, tts_eos_id,
                llm_pad_id,
                tts_pad_id,
                tts_pad_id, tts_pad_id, tts_pad_id, tts_pad_id,
            )

        # ── Pass 1: extract per-sample content IDs and LLM timing ────────────
        per_tts_ids:    list[list[int]]  = []
        bos_frames:     list[int]        = []
        eos_frames_llm: list[int | None] = []

        for b in range(B):
            row = gen_text[b].cpu().tolist()

            # Find first LLM BOS and the first LLM EOS after it.
            bos_pos = None
            eos_pos = None
            for t, tok in enumerate(row):
                if tok == llm_bos_id and bos_pos is None:
                    bos_pos = t
                if tok == llm_eos_id and bos_pos is not None and eos_pos is None:
                    eos_pos = t
                    break

            if bos_pos is not None:
                content_slice = row[bos_pos + 1 : eos_pos]
            else:
                # No BOS found — fall back to all tokens before first EOS.
                first_eos = next(
                    (t for t, tok in enumerate(row) if tok == llm_eos_id), None
                )
                content_slice = row[:first_eos]
                bos_pos = 0
                logging.warning(
                    f"_build_tts_sequence_v3[{b}]: no LLM BOS — placing TTS BOS at frame 0."
                )

            # ── Direct ID passthrough — "freeze last word" on PAD frames ────────
            # Option B: during LLM PAD frames (silence/delay between words)
            # we repeat the last content token rather than sending TTS PAD or
            # EOS.  The acoustic decoder keeps running from its current state
            # (stretches the last phoneme, then advances when the next word
            # token arrives), without the hallucination risk of TTS PAD.
            #
            #   LLM:  w1 w2 PAD PAD BOS  w3 w4 PAD PAD …
            #   TTS:  w1 w2               w3 w4           …
            #              ↑↑↑ ↑↑↑            ↑↑↑
            #    PADs, intermediate BOS, EOS → all ignored; content only
            tts_ids: list[int] = []
            remapped_count = 0
            for tok in content_slice:
                if tok == llm_pad_id or tok == llm_eos_id:
                    pass  # all PADs/EOS ignored
                elif tok == llm_bos_id:
                    pass  # intermediate BOS ignored
                elif tok in _REMAP:
                    tts_ids.append(_REMAP[tok])   # code fill → TTS PAD
                    remapped_count += 1
                else:
                    tts_ids.append(tok)            # regular content: identity

            if remapped_count > 0:
                logging.debug(
                    f"_build_tts_sequence_v3[{b}]: remapped {remapped_count} "
                    f"special token(s) to TTS PAD."
                )

            if not tts_ids:
                logging.warning(
                    f"_build_tts_sequence_v3[{b}]: no TTS content tokens — no speech will be generated."
                )

            per_tts_ids.append(tts_ids)
            bos_frames.append(bos_pos)
            eos_frames_llm.append(eos_pos)

        # ── Compute T_tts ─────────────────────────────────────────────────────
        # Each sample needs: bos_pos + 1 (BOS) + N (content) + pad_len + 2 (EOS + slack).
        max_needed = T
        for bp, tts_ids in zip(bos_frames, per_tts_ids):
            N       = len(tts_ids)
            seg_len = 1 + N
            pad_len = seg_len * pad_factor
            need    = bp + 1 + N + pad_len + 2
            max_needed = max(max_needed, need)
        T_tts = max_needed

        logging.info(
            f"_build_tts_sequence_v3: T={T} T_tts={T_tts} "
            f"max_N={max((len(x) for x in per_tts_ids), default=0)} "
            f"pad_factor={pad_factor}"
        )

        # ── Pass 2: fill TTS sequence ─────────────────────────────────────────
        tts_seq = torch.full((B, T_tts), tts_pad_id, dtype=torch.long, device=gen_text.device)
        eos_positions: list[int] = []

        for b in range(B):
            tts_ids    = per_tts_ids[b]
            N          = len(tts_ids)
            bp         = bos_frames[b]
            ep         = eos_frames_llm[b]
            seg_len    = 1 + N
            pad_len    = seg_len * pad_factor
            eos_in_pad = int(pad_len * eos_pct)

            # TTS BOS at the LLM's actual BOS frame (timing sync).
            tts_seq[b, bp] = tts_bos_id

            # Content tokens packed immediately after BOS.
            for k, tid in enumerate(tts_ids):
                pos = bp + 1 + k
                if pos < T_tts:
                    tts_seq[b, pos] = tid

            # EOS at formula position, pushed out to match LLM EOS frame if later.
            eos_recipe = bp + 1 + N + eos_in_pad
            if ep is not None:
                actual_eos = max(eos_recipe, ep)
            else:
                actual_eos = eos_recipe
            actual_eos = min(actual_eos, T_tts - 1)

            tts_seq[b, actual_eos] = tts_eos_id   # id=2 (</s>)
            eos_positions.append(actual_eos)

            logging.debug(
                f"_build_tts_sequence_v3[{b}]: bos={bp} N={N} "
                f"pad_len={pad_len} eos_recipe={eos_recipe} "
                f"eos_llm={ep} actual_eos={actual_eos} T_tts={T_tts}"
            )

        return tts_seq, T_tts, eos_positions
