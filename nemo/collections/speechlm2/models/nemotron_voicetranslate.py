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
NemotronVoiceTranslate: Offline duplex speech-to-speech translation model.

Combines:
  - DuplexS2SSpeechDecoderModel2-style LLM backbone (perception + LLM + lm_head)
    for streaming STT and text generation (e.g. Riva-Translate or Qwen2.5 LLM).
  - DuplexEARTTS for autoregressive audio codec generation conditioned on
    speaker reference audio (no TitaNet speaker encoder).

Inference only — no training_step, no prepare_inputs, no forward.
Speaker identity is controlled via EAR TTS audio prompt (set_init_inputs /
get_init_inputs) rather than a separate speaker embedding network.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector as NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS, load_audio_librosa
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.utils import logging


class NemotronVoiceTranslate(LightningModule, HFHubMixin):
    """
    Offline duplex speech-to-speech translation model.

    Architecture
    ------------
    STT / text side:
        self.perception   — streaming speech encoder (e.g. FastConformer)
        self.llm          — causal LLM decoder (e.g. Riva-Translate-4B / Qwen2.5)
        self.lm_head      — LM projection head
        self.embed_tokens — token embedding table (pulled out of LLM for FSDP safety)

    TTS / audio side:
        self.tts_model    — DuplexEARTTS: EAR-based autoregressive codec model.
                            Speaker identity is controlled via audio prompt
                            (set_init_inputs / get_init_inputs) — no TitaNet.

    Configuration
    -------------
    Required cfg fields:

        model:
            pretrained_llm: str          # HF model id
            pretrained_weights: bool     # whether to load LLM weights
            speech_generation:           # DuplexEARTTS config sub-tree
                ...
            scoring_asr: str             # ASR model id for ASRBLEU metric

        data:
            source_sample_rate: int
            target_sample_rate: int
            frame_length: float

        exp_manager:
            explicit_log_dir: str
    """

    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "Pass config to NemotronVoiceTranslate as a Python dict "
            f"(got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()

        cfg = DictConfig(cfg)
        self.full_cfg = cfg
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(
            cfg.exp_manager.explicit_log_dir, "validation_logs"
        )

        # optional text-channel advance (mirrors duplex_s2s_speech_decoder_model2)
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # source fps (used for metrics logging)
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )

        # ------------------------------------------------------------------ #
        # Tokenizer                                                            #
        # ------------------------------------------------------------------ #
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)

        if 'Riva-Translate-4B-Instruct' in self.cfg.pretrained_llm:
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

        if 'Qwen2.5' in self.cfg.pretrained_llm:
            logging.warning("Tokenizer has no bos_token — setting to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

        # ------------------------------------------------------------------ #
        # LLM backbone                                                         #
        # ------------------------------------------------------------------ #
        llm = load_pretrained_hf(
            self.cfg.pretrained_llm,
            pretrained_weights=self.cfg.pretrained_weights,
        ).train()
        self.llm = llm.model          # base transformer (no LM head)
        self.lm_head = llm.lm_head
        # embed_tokens lives outside llm to avoid FSDP/TP hook issues
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens

        maybe_install_lora(self)

        # ------------------------------------------------------------------ #
        # Speech encoder (perception)                                          #
        # ------------------------------------------------------------------ #
        # pretrained_asr in config provides the full preprocessor/encoder arch.
        # Weights are overwritten by init_from_model_from_ckpt afterward.
        setup_speech_encoder(self, pretrained_weights=self.cfg.get("pretrained_asr", None) is not None)

        # ------------------------------------------------------------------ #
        # TTS model (EAR TTS with audio-prompt speaker conditioning)           #
        # ------------------------------------------------------------------ #
        # DuplexEARTTS.__init__ expects:
        #   cfg.model  — the TTS model config (= self.cfg.speech_generation)
        #   cfg.data   — {target_sample_rate, source_sample_rate, frame_length}
        #   cfg.exp_manager.explicit_log_dir — for saving validation logs
        tts_model_cfg = OmegaConf.to_container(self.cfg.speech_generation, resolve=True)
        top_data = self.full_cfg.get("data", {})
        tts_full_cfg = {
            "model": tts_model_cfg,
            "data": {
                "target_sample_rate":    OmegaConf.select(top_data, "target_sample_rate",    default=22050),
                "source_sample_rate":    OmegaConf.select(top_data, "target_sample_rate",    default=22050),
                "frame_length":          OmegaConf.select(top_data, "frame_length",          default=0.08),
                "audio_prompt_duration": OmegaConf.select(top_data, "audio_prompt_duration", default=3.0),
            },
            "exp_manager": {
                "explicit_log_dir": OmegaConf.select(
                    self.full_cfg, "exp_manager.explicit_log_dir", default="/tmp/tts_validation"
                ),
            },
        }
        self.tts_model = DuplexEARTTS(tts_full_cfg)
        # ensure silence tokens are ready for inference
        self.tts_model.codec_silence_tokens = self.tts_model.get_codec_silence_frame()

        self.target_fps = self.tts_model.target_fps

        self._use_fsdp = False
        self._use_tp = False

    # ------------------------------------------------------------------ #
    # Token-ID helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        return get_pad_id(self.tokenizer)

    # ------------------------------------------------------------------ #
    # BOS embedding for first input frame                                  #
    # ------------------------------------------------------------------ #

    def _get_bos_embedding(self) -> Tensor:
        """Embed text_pad_id as a neutral seed for the first audio frame."""
        bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        return self.embed_tokens(bos)

    # ------------------------------------------------------------------ #
    # Checkpoint loading                                                   #
    # ------------------------------------------------------------------ #

    def init_from_model_from_ckpt(self, checkpoint_path: str):
        """Load a full model checkpoint (state_dict) with partial-init support."""
        if checkpoint_path is None:
            return
        if '.nemo' in checkpoint_path:
            with tempfile.TemporaryDirectory() as tmpdir:
                NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            state_dict = torch.load(
                checkpoint_path, weights_only=False, map_location='cpu'
            )['state_dict']

        state_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
        self.load_state_dict(state_dict, strict=True)

    def init_tts_model_from_checkpoint(self, checkpoint_path: str):
        """Load only TTS weights from a standalone DuplexEARTTS checkpoint."""
        if checkpoint_path is None:
            return
        if '.nemo' in checkpoint_path:
            with tempfile.TemporaryDirectory() as tmpdir:
                NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            state_dict = torch.load(
                checkpoint_path, weights_only=False, map_location='cpu'
            )['state_dict']

        state_dict = set_model_dict_for_partial_init(state_dict, self.tts_model.state_dict())
        self.tts_model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------ #
    # Training — not implemented                                           #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            "NemotronVoiceTranslate is inference-only; training_step is not implemented."
        )

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def _build_tts_sequence(
        self,
        gen_text: "torch.Tensor",
        tts_tok,
        tts_pad_id: int,
        pad_factor: int = 3,
        eos_pct: float = 0.7,
        gen_text_len: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, int]":
        """
        Offline vocab bridge: convert (B, T) LLM token sequence → (B, T_tts) TTS sequence.

        Follows the EAR TTS eval recipe (duplex_eartts_eval.py):

            [BOS,  w1  w2 … wN,  PAD × (seg_len × pad_factor),  EOS at 70 %,  PAD …]
              0    1   2     N    ← padding window (pad_factor × seg_len) →

        where seg_len = 1 + N (BOS + content tokens).

        pad_factor controls the trailing PAD window.  The original EAR TTS eval
        recipe uses 10, but that produces T_tts ≈ 300+ frames for typical
        sentences, ballooning the TTS KV-cache to ~600 MB at B=8.
        pad_factor=3 keeps T_tts close to T (the LLM horizon) for most
        sentences while still giving the TTS enough runway to finish articulating.

        T_tts is chosen large enough to hold this structure and is returned so
        the caller can size its gen_codes buffer accordingly.  It may exceed T
        (the LLM horizon) when the translation is long.

        Only tokens strictly between the first LLM BOS and its following EOS
        are decoded and re-tokenised with the TTS tokeniser — post-EOS garbage
        from the continued LLM autoregressive loop is excluded.

        gen_text_len: per-sample length of the committed LLM output (EOS+1 when EOS
            was found, T when it was not).  When provided, the BOS/EOS scan is
            limited to [:gen_text_len[b]] rather than the full T-length buffer,
            preventing content from run-on generation beyond the valid window.

        Content is automatically capped so that T_tts ≤ T (= 1.3 × T_source).
        This ties the TTS budget to the LLM budget and prevents codec 32-bit
        overflow when EOS was never predicted.
        """
        B, T = gen_text.shape
        special = {self.text_pad_id, self.text_bos_id, self.text_eos_id}

        # --- Pass 1: extract per-sample tts_ids ----------------------------
        per_tts_ids: list[list[int]] = []

        for b in range(B):
            # Limit the scan to the committed LLM output window.
            # When EOS was found: gen_text_len[b] = eos_pos + 1.
            # When EOS was never found: gen_text_len[b] = T (full buffer), but we
            # still cap via max_content_tokens below to prevent codec overflow.
            committed_len = int(gen_text_len[b]) if gen_text_len is not None else T
            row = gen_text[b, :committed_len].cpu().tolist()

            # Find first BOS then first EOS that follows it.
            bos_pos = None
            eos_pos = None
            for t, tok in enumerate(row):
                if tok == self.text_bos_id and bos_pos is None:
                    bos_pos = t
                if tok == self.text_eos_id and bos_pos is not None and eos_pos is None:
                    eos_pos = t
                    break

            # Extract content tokens.
            # If the LLM generated BOS: use only tokens strictly between BOS and EOS.
            # If the LLM skipped BOS (translates directly): use all tokens up to the
            # first EOS — the LLM still produced a valid translation, just without <s>.
            if bos_pos is not None:
                content_slice = row[bos_pos + 1 : eos_pos]  # eos_pos=None → row[bos+1:]
            else:
                first_eos = next(
                    (t for t, tok in enumerate(row) if tok == self.text_eos_id), None
                )
                content_slice = row[:first_eos]
                logging.warning(
                    f"_build_tts_sequence[{b}]: no LLM BOS — using all content up to EOS."
                )

            content_ids = [tok for tok in content_slice if tok not in special]

            # Cap content to ensure T_tts ≤ T (= 1.3 × T_source).
            # Derivation: T_tts = (1 + pad_factor) × (1 + N) + 2
            # Setting T_tts ≤ T  →  N ≤ (T − 2) / (1 + pad_factor) − 1
            # This ties the TTS budget to the LLM budget, preventing runaway
            # growth (and codec 32-bit overflow) when EOS was never predicted.
            max_N_safe = max(0, int((T - 2) / (1 + pad_factor)) - 1)
            if len(content_ids) > max_N_safe:
                logging.warning(
                    f"_build_tts_sequence[{b}]: content truncated from "
                    f"{len(content_ids)} to {max_N_safe} LLM tokens "
                    f"(EOS {'found' if eos_pos is not None else 'NOT found'}, "
                    f"T={T}, pad_factor={pad_factor})."
                )
                content_ids = content_ids[:max_N_safe]

            tts_ids: list[int] = []
            if content_ids:
                text = self.tokenizer.ids_to_text(content_ids)
                if text.strip():
                    tts_ids = tts_tok.text_to_ids(text)

            if not tts_ids:
                logging.warning(
                    f"_build_tts_sequence[{b}]: no TTS tokens produced — no speech will be generated."
                )

            per_tts_ids.append(tts_ids)

        # --- Compute T_tts following EAR TTS eval recipe -------------------
        # seg_len = 1 (BOS) + N (content); pad_len = seg_len × pad_factor
        # minimum sequence: 1 + N + pad_len + 1 (EOS sentinel)
        max_N = max((len(ids) for ids in per_tts_ids), default=0)
        seg_len_max = 1 + max_N
        pad_len_max = seg_len_max * pad_factor
        T_tts = max(T, 1 + max_N + pad_len_max + 2)
        logging.info(f"_build_tts_sequence: max_N={max_N} T={T} T_tts={T_tts} (pad_factor={pad_factor})")

        tts_seq = torch.full((B, T_tts), tts_pad_id, dtype=torch.long, device=gen_text.device)

        # --- Pass 2: fill tts_seq per sample --------------------------------
        # Track EOS positions directly rather than searching by token ID later.
        # Searching is unreliable when the fast tokenizer returns unk_id for
        # newly-added special tokens (e.g. </s> on a Qwen-based vocabulary).
        eos_positions: list[int] = []

        for b in range(B):
            tts_ids = per_tts_ids[b]
            N = len(tts_ids)
            seg_len = 1 + N
            pad_len = seg_len * pad_factor

            # BOS at position 0 (EAR TTS eval recipe)
            tts_seq[b, 0] = tts_tok.bos_id

            # Content tokens at positions 1 … N
            for k, tid in enumerate(tts_ids):
                tts_seq[b, 1 + k] = tid

            # EOS at 70 % into the PAD window (EAR TTS eval recipe)
            eos_in_pad = int(pad_len * eos_pct)
            eos_pos_tts = min(1 + N + eos_in_pad, T_tts - 1)
            tts_seq[b, eos_pos_tts] = tts_tok.eos_id
            eos_positions.append(eos_pos_tts)

            logging.debug(
                f"_build_tts_sequence[{b}]: N_tts={N} seg_len={seg_len} "
                f"pad_len={pad_len} eos_pos={eos_pos_tts} T_tts={T_tts}"
            )

        return tts_seq, T_tts, eos_positions

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        speaker_audio: torch.Tensor = None,
        speaker_audio_lens: torch.Tensor = None,
        input_pad_len: int = 0,
        decode_audio: bool = True,
        incremental_audio_decoding: bool = None,   # None → read from cfg (default False)
        generation_config: dict = None,
        guidance_enabled: bool = None,    # None → read from cfg (default True)
    ) -> dict:
        """
        Full offline speech-to-speech translation inference.

        Steps
        -----
        1. Encode source audio with the speech encoder (perception).
        2. Determine decoding length T (with optional FSDP sync / config override).
        3. Bootstrap the LLM on frame 0 (BOS frame) → gen_text[:, 0].
        4. Initialize EAR TTS with speaker reference audio prompt.
        5. Warm up TTS transformer on the speaker prompt → KV cache.
        6. Autoregressive loop (t = 1 … T-1):
            a. Embed gen_text[:, t-1] → add into input_embeds[:, t] → LLM step.
            b. gen_text[:, t] ← argmax(lm_head(LLM output)).
            c. infer_codes_one_step → gen_codes[:, t].
        7. Decode codec tokens → waveform (non-incremental or incremental).

        Args
        ----
        input_signal : (B, T_source) user waveform at source_sample_rate.
        input_signal_lens : (B,) waveform lengths in samples.
        speaker_audio : (B, T_ref) speaker reference waveform, optional.
            If None, loaded from cfg.inference_speaker_reference or
            cfg.inference_speaker_name (pre-cached latent).
        speaker_audio_lens : (B,) lengths of speaker_audio, optional.
        input_pad_len : extra padding added to input_signal before encoding.
        decode_audio : if True, decode codec tokens to waveform.
        incremental_audio_decoding : if True, decode one chunk per step
            (lower latency, higher compute); otherwise decode all at end.
        generation_config : TTS sampling parameters; defaults from tts_model.
        guidance_enabled : enable classifier-free guidance in TTS.

        Returns
        -------
        dict with keys:
            "text"         : List[str]  — generated text per sample.
            "tokens_text"  : (B, T)    — generated text token ids.
            "tokens_len"   : (B,)      — valid length per sample.
            "audio"        : (B, T_wave) — generated waveform (if decode_audio).
            "audio_len"    : (B,)      — waveform lengths in samples (if decode_audio).
        """

        # Resolve incremental_audio_decoding: function arg overrides cfg.
        # Default is False (post-loop batch decode, matching model2 behaviour).
        if incremental_audio_decoding is None:
            incremental_audio_decoding = self.cfg.get("incremental_audio_decoding", False)

        # Resolve guidance_enabled: function arg overrides cfg.speech_generation.
        # Default is True (classifier-free guidance on), matching marianag's eval_config.yaml.
        if guidance_enabled is None:
            guidance_enabled = self.cfg.speech_generation.get("inference_guidance_enabled", True)

        # -------------------------------------------------------- #
        # 1. Encode source audio                                     #
        # -------------------------------------------------------- #
        if input_pad_len > 0:
            input_signal = torch.nn.functional.pad(
                input_signal, (0, input_pad_len), mode='constant', value=0
            )
            input_signal_lens = input_signal_lens + input_pad_len
       
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            source_encoded, lengths, asr_emb = self.perception(
                input_signal=input_signal,
                input_signal_length=input_signal_lens,
                return_encoder_emb=True,
            )
        B, T_local, H = source_encoded.shape

        # -------------------------------------------------------- #
        # 2. Determine T (FSDP sync or config override)             #
        # -------------------------------------------------------- #
        T_tensor = torch.tensor([T_local], device=source_encoded.device)
        if self._use_fsdp:
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)

        T = int(self.cfg.get("inference_tgt_len", 1.5 * T_tensor.item()))

        if T > T_local:
            last_frame = source_encoded[:, T_local - 1 : T_local, :]
            pad_frames = last_frame.repeat(1, T - T_local, 1)
            source_encoded = torch.cat([source_encoded, pad_frames], dim=1)

        # -------------------------------------------------------- #
        # 3. Build input embeddings and LLM cache                   #
        # -------------------------------------------------------- #
        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)
        # Cast to bf16 to match the LLM weights (on_validation_start cast LLM to bf16).
        input_embeds = input_embeds.to(self.llm.dtype)

        cache = DynamicCache()
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)

        # Step 0: BOS frame → first text prediction
        input_embeds[:, 0] = input_embeds[:, 0] + self._get_bos_embedding()
        out0 = self.llm(
            inputs_embeds=input_embeds[:, :1],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        text_logits_0 = self.lm_head(out0['last_hidden_state'])
        gen_text[:, 0] = text_logits_0[:, -1].argmax(dim=-1)
        llm_cache = out0['past_key_values']

        # -------------------------------------------------------- #
        # 4. Initialize EAR TTS with speaker audio prompt            #
        # -------------------------------------------------------- #
        if speaker_audio is None:
            speaker_name = self.cfg.get("inference_speaker_name", None)
            if speaker_name is not None:
                speaker_audio = None
                speaker_audio_lens = None
            else:
                spk_audio_raw, sr = load_audio_librosa(self.cfg.inference_speaker_reference)
                spk_audio = resample(spk_audio_raw, sr, self.tts_model.target_sample_rate)
                speaker_audio = spk_audio.repeat(B, 1).to(self.device)
                speaker_audio_lens = (
                    torch.tensor([speaker_audio.size(1)]).long().repeat(B).to(self.device)
                )
        else:
            speaker_name = None

        self.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            speaker_name=speaker_name,
        )
        init_inputs = self.tts_model.get_init_inputs(B=B)

        if generation_config is None:
            generation_config = self.tts_model._get_generation_config(guidance_enabled)

        # Output buffers
        gen_text_len = torch.full((B,), T, device=self.device, dtype=input_signal_lens.dtype)

        # -------------------------------------------------------- #
        # 6. Joint LLM + TTS autoregressive loop                    #
        #    Mirrors nemotron_voicechat.py: at each step t the LLM  #
        #    generates one text token and the TTS immediately        #
        #    generates one audio codec frame.                        #
        #                                                            #
        #    Vocab bridge: the LLM (Riva-Translate-4B, 13k) and     #
        #    EAR TTS (Nemotron-Nano-9B, ~128k) use different        #
        #    tokenizers. At each step we decode the accumulated LLM  #
        #    text and retokenize it with the TTS tokenizer, then     #
        #    feed the latest stable TTS token to infer_codes_one_step.
        # -------------------------------------------------------- #
        tts_tok    = self.tts_model.tokenizer
        tts_pad_id = self.tts_model.text_pad_id   # correct pad used during EAR TTS training

        # TTS warmup on speaker prompt — identical to nemotron_voicechat.py
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})
        outputs = self.tts_model.tts_model(**init_inputs)
        code    = init_inputs["code"][:, -1:]

        past_key_values = outputs.past_key_values
        num_quantizers  = self.tts_model.tts_model.config.num_quantizers
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        audio_pred      = None
        audio_pred_len  = torch.zeros(B, device=self.device, dtype=torch.long)

        # -------------------------------------------------------- #
        # Pass 1 — LLM-only autoregressive loop                  #
        #   Generate all T text tokens first, then bridge to TTS. #
        # -------------------------------------------------------- #
        # Track which samples have already emitted EOS so generation stops there
        # — mirrors duplex_s2s_speech_decoder_model2.py.
        eos_done = torch.zeros(B, device=self.device, dtype=torch.bool)
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] = input_embeds[:, t] + last_emb

            out_t = self.llm(
                inputs_embeds=input_embeds[:, t : t + 1],
                past_key_values=llm_cache,
                use_cache=True,
                return_dict=True,
            )
            text_logits_t  = self.lm_head(out_t['last_hidden_state'])
            gen_text[:, t] = text_logits_t[:, -1].argmax(dim=-1)
            llm_cache      = out_t['past_key_values']

            # Update gen_text_len for samples that just produced EOS.
            just_eos = (~eos_done) & (gen_text[:, t] == self.text_eos_id)
            gen_text_len[just_eos] = t + 1
            eos_done |= just_eos

            if eos_done.all():
                break  # all samples done — mirrors duplex_s2s_speech_decoder_model2.py

            logging.debug(f"LLM step {t}/{T}")

        # -------------------------------------------------------- #
        # Pass 2 — Offline vocab bridge                           #
        #   Only content between LLM BOS and EOS is decoded and   #
        #   re-tokenised.  T_tts follows the EAR TTS eval recipe: #
        #   BOS + content + pad_factor × seg_len PADs + EOS.      #
        # -------------------------------------------------------- #
        # Pass pad_factor from config only when explicitly set, so each subclass
        # can define its own default via the _build_tts_sequence signature.
        # (e.g. base class defaults to 3, sync class defaults to 5.)
        _pad_factor_cfg = self.cfg.get("tts_pad_factor", None)
        _tts_build_kwargs = {"pad_factor": _pad_factor_cfg} if _pad_factor_cfg is not None else {}
        tts_input, T_tts, tts_eos_positions = self._build_tts_sequence(
            gen_text, tts_tok, tts_pad_id,
            gen_text_len=gen_text_len,
            **_tts_build_kwargs,
        )

        # Allocate TTS buffers with the (potentially larger) T_tts.
        gen_codes    = torch.zeros(B, T_tts, num_quantizers, device=self.device, dtype=torch.long)
        subword_mask = torch.ones(B, T_tts, device=self.device, dtype=torch.bool)

        # -------------------------------------------------------- #
        # Pass 3 — TTS-only autoregressive loop                   #
        #   Matches duplex_ear_tts.py offline_inference exactly:  #
        #   loop starts at t=0 so BOS is processed as current     #
        #   before the first word (standalone TTS eval recipe).   #
        # -------------------------------------------------------- #
        for t in range(T_tts):
            current_subword_id   = tts_input[:, t : t + 1]
            prev_subword_id      = (
                first_context_subword_id if t == 0
                else tts_input[:, t - 1 : t]
            )
            current_subword_mask = subword_mask[:, t].unsqueeze(-1)

            code, past_key_values = self.tts_model.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
            )
            gen_codes[:, t] = code.squeeze(1)

            if decode_audio and incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.tts_model.decode_one_audio_step(
                    gen_codes[:, : t + 1],
                    number_prev_tokens=self.cfg.get(
                        "inference_codec_decoding_prev_tokens_number", None
                    ),
                )
                audio_pred = (
                    audio_pred_i if audio_pred is None
                    else torch.cat([audio_pred, audio_pred_i], dim=1)
                )
                audio_pred_len += audio_pred_i_len

            logging.debug(f"TTS step {t}/{T_tts}")

        # -------------------------------------------------------- #
        # 7. Compute per-sample audio length from TTS EOS position. #
        #    Use positions returned by _build_tts_sequence directly  #
        #    rather than searching by token ID — the fast tokenizer  #
        #    may return unk_id for </s> making the search unreliable. #
        # -------------------------------------------------------- #
        _AUDIO_TRAIL = 10
        gen_codes_lengths = torch.tensor(
            [min(eos_pos + 1 + _AUDIO_TRAIL, T_tts) for eos_pos in tts_eos_positions],
            device=self.device, dtype=torch.long,
        )

        if decode_audio:
            if not incremental_audio_decoding:
                with fp32_precision(), torch.no_grad():
                    audio_pred, audio_pred_len = self.tts_model.audio_codec.decode(
                        gen_codes, gen_codes_lengths
                    )

        # -------------------------------------------------------- #
        # 9. Post-process text                                       #
        #    Decode BEFORE trimming so that EOS tokens that fall in  #
        #    the extended window (T_local < t ≤ T) are honoured.     #
        #    gen_text_len correctly records the first EOS position   #
        #    up to T; clamping it to T_local BEFORE decoding would   #
        #    truncate long translations and include post-content      #
        #    garbage tokens for samples where EOS > T_local.         #
        # -------------------------------------------------------- #
        text_output = tokens_to_str(
            gen_text, gen_text_len, tokenizer=self.tokenizer, pad_id=self.text_pad_id
        )

        # -------------------------------------------------------- #
        # 8. Trim text to local length (FSDP padding removal).      #
        #    Done AFTER decoding so that EOS beyond T_local is still #
        #    used for text extraction above.                         #
        #    gen_codes / audio are NOT trimmed here.                 #
        # -------------------------------------------------------- #
        gen_text     = gen_text[:, :T_local]
        gen_text_len = gen_text_len.clamp(max=T_local)

        ans = {
            "text": text_output,
            "tokens_text": gen_text,
            "tokens_len": gen_text_len,
        }

        if decode_audio:
            ans["audio"]     = audio_pred.squeeze(1)
            ans["audio_len"] = audio_pred_len

        return ans

    # ------------------------------------------------------------------ #
    # Validation / Test plumbing                                           #
    # ------------------------------------------------------------------ #

    def on_validation_start(self) -> None:
        """Precision setup for validation.

        Both the LLM and the ASR speech encoder (perception) are cast to bf16 to
        match the bf16-true training distribution.

        With trainer.precision=bf16-true (recommended):
          - Lightning auto-casts all parameters to bf16 and activates autocast.
          - The explicit casts below are no-ops but kept for robustness.
          - The TTS codec is safe because offline_inference wraps all codec
            decode calls in fp32_precision(), which overrides the bf16 autocast.

        With trainer.precision=32 (fallback):
          - LLM and perception must be cast to bf16 manually to match the
            bf16-true training distribution.
          - The torch.autocast(bf16) wrapper around perception in offline_inference
            provides an additional layer of protection.

        Note: .to() calls on FSDP-wrapped modules corrupt FSDP's internal flat-
        parameter bookkeeping and cause segfaults on the first forward pass.
        Under FSDP, dtype is managed by the mixed_precision policy, so we skip
        the manual casts entirely.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        def _is_fsdp(module) -> bool:
            return module is not None and isinstance(module, FSDP)

        if any(_is_fsdp(getattr(self, m, None)) for m in ("llm", "lm_head", "embed_tokens", "perception")):
            logging.info(
                "NemotronVoiceTranslate: FSDP detected — skipping manual bf16 cast "
                "(dtype is managed by FSDP mixed_precision policy)."
            )
            return

        if hasattr(self, "llm") and self.llm is not None:
            self.llm.to(torch.bfloat16)
        if hasattr(self, "lm_head") and self.lm_head is not None:
            self.lm_head.to(torch.bfloat16)
        if hasattr(self, "embed_tokens") and self.embed_tokens is not None:
            self.embed_tokens.to(torch.bfloat16)
        if hasattr(self, "perception") and self.perception is not None:
            self.perception.to(torch.bfloat16)
        logging.info(
            "NemotronVoiceTranslate: LLM + perception → bf16 (matching training)."
        )

    def on_train_epoch_start(self) -> None:
        self.tts_model.on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        self.results_logger.compute_and_save()

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        """
        Runs one validation step.

        Called automatically by PTL as: validation_step(batch, batch_idx).
        PTL never injects extra arguments — speaker audio is sourced from inside
        the batch when ``model.inference_use_source_speaker=true``.

        The batch is a dict of dataset-name → sub-batch dicts (produced by
        CombinedLoader), each containing at minimum:
            "source_audio"      : (B, T)  @ source_sample_rate
            "source_audio_lens" : (B,)
            "target_texts"      : List[str]
            "sample_id"         : List[str]

        When ``model.inference_use_source_speaker=true`` the source audio for
        each sample is resampled to target_sample_rate and used as that sample's
        individual TTS speaker reference (per-sample voice cloning).
        """
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue

            # ── Determine speaker audio for this batch ──────────────────────
            # Source-speaker cloning: use first_turn_audio, which is already:
            #   • resampled to target_sample_rate (22050 Hz) by the dataloader
            #   • trimmed to training_speaker_duration (~3 s), matching the TTS
            #     audio_prompt_duration — no resampling or truncation needed here.
            #   • from the source/input-role speaker (same roles as input_roles)
            batch_speaker_audio = None
            batch_speaker_audio_lens = None

            if self.cfg.get("inference_use_source_speaker", False):
                batch_speaker_audio = dataset_batch["first_turn_audio"]
                batch_speaker_audio_lens = dataset_batch["first_turn_audio_lens"]

            results = self.offline_inference(
                input_signal=dataset_batch["source_audio"],
                input_signal_lens=dataset_batch["source_audio_lens"],
                speaker_audio=batch_speaker_audio,
                speaker_audio_lens=batch_speaker_audio_lens,
            )

            with fp32_precision():
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(
                        results["audio"], self.target_sample_rate, 16000
                    ),
                    pred_audio_lens=(
                        results["audio_len"] / self.target_sample_rate * 16000
                    ).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch["sample_id"],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                )

            self.bleu.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=results["text"],
            )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # FSDP / TP parallel configuration (mirrors model2)                   #
    # ------------------------------------------------------------------ #

    def configure_model(self) -> None:
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Shard(1),),
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                }
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} not divisible by "
                            f"{tp_mesh.size()=}: adjust tensor parallelism."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())
                parallelize_module(transformer_block, tp_mesh, plan)

            parallelize_module(
                self.lm_head,
                tp_mesh,
                ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
            )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True
            fsdp_config = {"mesh": dp_mesh}
            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.tts_model = fully_shard(self.tts_model, **fsdp_config)

    # ------------------------------------------------------------------ #
    # State dict loading                                                   #
    # ------------------------------------------------------------------ #

    def load_state_dict(self, state_dict, strict: bool = True):
        # recreate audio prompt latent entries if needed (EAR TTS compatibility)
        self.tts_model.maybe_recreate_cached_audio_prompt_latents_structure(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading state_dict — retrying with partial init.")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
