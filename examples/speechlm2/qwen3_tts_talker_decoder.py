#!/usr/bin/env python
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
"""Qwen3-TTS talker adapter for the duplex S2S speech-generation slot.

Speaker conditioning uses Qwen3-TTS's own ECAPA-TDNN speaker encoder
(1024-dim, 24 kHz) rather than TitaNet, enabling speaker cloning with the
same voice encoder that Qwen3-TTS was trained with.  The embedding is
projected additively into the talker's conditioning signal, matching the
additive approach used for other conditioning inputs (ASR emb, LLM latent).
"""

from __future__ import annotations

from typing import Any

import torch
import torchaudio
from torch import nn


def _as_dtype(name: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {name!r}")


class Qwen3TTSTalkerSpeechDecoder(nn.Module):
    """Expose the Qwen3-TTS talker with the NeMo duplex speech-decoder API.

    Architecture summary
    --------------------
    * **Codebook prediction**: The 12 Hz Qwen3 talker predicts the first
      codec codebook with its main Transformer; a lightweight sub-predictor
      handles codebooks 2..K autoregressively (teacher-forced during training,
      greedy during inference).
    * **Speaker conditioning**: Qwen3's own ECAPA-TDNN (``qwen.model.speaker_encoder``,
      1024-dim, 24 kHz input) replaces TitaNet.  A linear projection maps the
      1024-dim x-vector to the talker's hidden size and adds it to the text
      conditioning — the same additive pattern used for ASR / LLM latent.
    * **Classifier-Free Guidance (CFG)**: optionally drops the full condition
      vector with probability ``cfg_unconditional_prob`` during training; at
      inference scales logits by ``cfg_scale``.

    Parameters
    ----------
    model_path_or_name:
        Path to ``Qwen3-TTS-12Hz-0.6B-Base-hf`` (or HF hub ID).
    speech_decoder_parms:
        ``cfg.model.speech_decoder`` dict.  Keys consumed here are popped so
        the remainder is not silently ignored.
    lantent_dim:
        LLM hidden size (used for optional LLM-latent conditioning).
    num_audio_codebooks:
        Must match ``talker.config.num_code_groups`` (16 for 0.6B).
    speech_vocab_size:
        Effective vocab size seen by the duplex model (e.g. 3072).
    raw_codec_vocab_size:
        Number of raw audio codes below the special-token range (2048).
    """

    def __init__(
        self,
        model_path_or_name: str,
        speech_decoder_parms: dict[str, Any],
        lantent_dim: int,
        num_audio_codebooks: int,
        speech_vocab_size: int,
        raw_codec_vocab_size: int = 2048,
        device_map: str = "cuda:0",
        dtype: str | torch.dtype | None = torch.bfloat16,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        self.use_input_cache = False
        self.cache = self._init_cache()
        self.speech_decoder_parms = dict(speech_decoder_parms)
        self.lantent_dim = int(lantent_dim)
        self.num_audio_codebooks = int(num_audio_codebooks)
        self.num_audio_tokens_per_codebook = int(speech_vocab_size)
        self.raw_codec_vocab_size = int(raw_codec_vocab_size)

        # Pop all known keys so stray keys are caught early.
        self.cfg_unconditional_prob = self.speech_decoder_parms.pop("cfg_unconditional_prob", None)
        self.cfg_scale = self.speech_decoder_parms.pop("cfg_scale", 2.5)
        self.cond_on_prev_audio_tokens = self.speech_decoder_parms.pop("cond_on_prev_audio_tokens", True)
        self.detach_input = self.speech_decoder_parms.pop("detach_input", False)
        self.cond_on_text_tokens = self.speech_decoder_parms.pop("cond_on_text_tokens", False)
        self.cond_on_llm_latent = self.speech_decoder_parms.pop("cond_on_llm_latent", False)
        self.cond_on_asr_emb = self.speech_decoder_parms.pop("cond_on_asr_emb", False)
        self.drop_asr_emb_prob = self.speech_decoder_parms.pop("drop_asr_emb_prob", 0.0)
        self.asr_emb_dim = self.speech_decoder_parms.pop("asr_emb_dim", 512)
        self.cond_on_modality_adapter_emb = self.speech_decoder_parms.pop("cond_on_modality_adapter_emb", False)
        self.use_speaker_encoder = self.speech_decoder_parms.pop("use_speaker_encoder", True)
        # speaker_embedding_dim from config is a hint only; setup_speaker_encoder() overrides it
        # to the actual ECAPA-TDNN output dim (1024).
        self.speaker_embedding_dim = self.speech_decoder_parms.pop("speaker_embedding_dim", 192)
        self.inference_speaker_reference = self.speech_decoder_parms.pop("inference_speaker_reference", None)
        self.max_speaker_reference_len = self.speech_decoder_parms.pop("max_speaker_reference_len", 5)
        self.speaker_encoder_model_name = self.speech_decoder_parms.pop("speaker_encoder_model_name", "titanet_large")
        self.cond_on_char_embedding = self.speech_decoder_parms.pop("cond_on_char_embedding", True)
        self.use_random_spk_emb = False

        # ------------------------------------------------------------------
        # Load Qwen3-TTS and extract the talker + speaker encoder.
        # ------------------------------------------------------------------
        from qwen_tts import Qwen3TTSModel

        qwen = Qwen3TTSModel.from_pretrained(
            model_path_or_name,
            device_map=device_map,
            dtype=_as_dtype(dtype),
            attn_implementation=attn_implementation,
        )
        qwen_model = qwen.model

        # Talker (main transformer + sub-codebook predictor)
        self.talker = qwen_model.talker
        self.talker_config = self.talker.config
        self.talker_hidden_size = int(self.talker_config.hidden_size)
        self.first_codebook_vocab_size = int(self.talker_config.vocab_size)
        self.sub_codebook_vocab_size = int(self.talker_config.code_predictor_config.vocab_size)

        if self.num_audio_codebooks != int(self.talker_config.num_code_groups):
            raise ValueError(
                f"Qwen talker expects {self.talker_config.num_code_groups} codebooks "
                f"but duplex model has {self.num_audio_codebooks}."
            )
        if self.num_audio_tokens_per_codebook > self.first_codebook_vocab_size:
            raise ValueError(
                f"speech_vocab_size={self.num_audio_tokens_per_codebook} exceeds "
                f"Qwen first-codebook vocab={self.first_codebook_vocab_size}."
            )

        # Save the ECAPA-TDNN speaker encoder before releasing the wrapper.
        # It lives at qwen.model.speaker_encoder (Qwen3TTSSpeakerEncoder, 1024-dim, 24 kHz).
        self._qwen_speaker_encoder = qwen_model.speaker_encoder
        self._qwen_speaker_encoder_sr = int(getattr(qwen_model, "speaker_encoder_sample_rate", 24000))

        # Release the full wrapper — the duplex model owns the frozen codec
        # separately via Qwen3TTSCodecAdapter.
        del qwen
        del qwen_model

        # ------------------------------------------------------------------
        # Optional projection layers.
        # ------------------------------------------------------------------
        self.input_proj = None
        if self.cond_on_llm_latent:
            self.input_proj = nn.Linear(self.lantent_dim, self.talker_hidden_size)

        self.modality_adapter_emb_projection = None
        if self.cond_on_modality_adapter_emb:
            self.modality_adapter_emb_projection = nn.Linear(self.lantent_dim, self.talker_hidden_size)

        self.asr_emb_projection = None
        if self.cond_on_asr_emb:
            self.asr_emb_projection = nn.Linear(self.asr_emb_dim, self.talker_hidden_size)

        # ------------------------------------------------------------------
        # Speaker encoder setup.
        # setup_speaker_encoder() sets self.speaker_encoder and
        # overwrites self.speaker_embedding_dim to the actual output dim.
        # ------------------------------------------------------------------
        self.speaker_encoder = None
        self.speaker_encoder_emb_projection = None
        if self.use_speaker_encoder:
            self.setup_speaker_encoder()
            # speaker_embedding_dim is now 1024 (set inside setup_speaker_encoder)
            self.speaker_encoder_emb_projection = nn.Linear(self.speaker_embedding_dim, self.talker_hidden_size)
            inference_speaker_embedding = torch.randn([1, 1, self.speaker_embedding_dim])
            self.register_buffer("inference_speaker_embedding", inference_speaker_embedding, persistent=False)
            if self.inference_speaker_reference:
                self.update_inference_speaker_embedding(self.inference_speaker_reference)

        self.register_buffer(
            "_extra_logit_floor",
            torch.tensor(-1.0e4, dtype=torch.float32),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Speaker encoder
    # ------------------------------------------------------------------

    def setup_speaker_encoder(self):
        """Install (or re-freeze) the Qwen3 ECAPA-TDNN speaker encoder.

        Called once from ``__init__`` and again from ``on_train_epoch_start``
        (via the outer model) to ensure the encoder stays frozen.  After the
        first call the encoder is registered as ``self.speaker_encoder`` and
        subsequent calls just re-apply ``eval()`` + freeze.
        """
        if self.speaker_encoder is None:
            # First call from __init__: install the Qwen3 ECAPA-TDNN.
            self.speaker_encoder = self._qwen_speaker_encoder  # Qwen3TTSSpeakerEncoder
            # Determine output dimension from the final FC layer.
            fc = self._qwen_speaker_encoder.fc
            self.speaker_embedding_dim = (
                fc.out_channels if hasattr(fc, "out_channels") else fc.out_features
            )  # 1024 for 0.6B-Base

        # Always ensure frozen + eval (safe to call repeatedly).
        self.speaker_encoder.eval()
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

    def update_inference_speaker_embedding(self, audio_path: str):
        audio, sr = torchaudio.load(audio_path)
        audio_len = torch.tensor([audio.size(1)]).long()
        self.inference_speaker_embedding = self.get_speaker_embedding(
            audio.to(self.device), audio_len.to(self.device), sr
        )

    def update_inference_speaker_embedding_from_embedding(self, embedding):
        self.inference_speaker_embedding = embedding

    def get_speaker_embedding(self, audio: torch.Tensor, audio_len: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract a 1024-dim x-vector using Qwen3's ECAPA-TDNN.

        Args:
            audio:     ``(B, S)`` float waveform at sample rate ``sr``.
            audio_len: ``(B,)`` lengths in samples (used to trim to reference).
            sr:        Sample rate of ``audio``.

        Returns:
            ``(B, 1, 1024)`` speaker embedding, cast to ``inference_speaker_embedding`` dtype.
        """
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        audio = audio[:, : int(self.max_speaker_reference_len * sr)]
        with torch.no_grad():
            # Resample to the ECAPA-TDNN's expected 24 kHz.
            audio_24k = torchaudio.functional.resample(
                audio.float(), sr, self._qwen_speaker_encoder_sr
            )
            mels = mel_spectrogram(
                audio_24k,
                n_fft=1024,
                num_mels=128,
                sampling_rate=self._qwen_speaker_encoder_sr,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)  # (B, T_mel, 128)

            speaker_emb = self.speaker_encoder(
                mels.to(self.device).to(torch.float32)
            )  # (B, 1024)
            speaker_emb = speaker_emb.unsqueeze(1)  # (B, 1, 1024)

        return speaker_emb.to(self.inference_speaker_embedding.dtype)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_cache():
        return {
            "hidden_states": None,
            "speech_mask": None,
            "input_audio_tokens": None,
            "target_text_tokens": None,
            "modality_adapter_emb": None,
            "asr_emb": None,
        }

    def reset_input_and_kv_cache(self, use_cache: bool):
        if use_cache:
            print("Enabling Qwen3 talker input cache!", flush=True)
        else:
            print("Disabling Qwen3 talker input cache!", flush=True)
        self.use_input_cache = use_cache
        self.cache = self._init_cache()

    def _append_cache(self, name: str, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if self.cache[name] is None:
            self.cache[name] = value
        else:
            self.cache[name] = torch.cat([self.cache[name], value], dim=1)
        return self.cache[name]

    # ------------------------------------------------------------------
    # Token sanitisation helpers
    # ------------------------------------------------------------------

    def _sanitize_first_codebook(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.where(
            (tokens >= 0) & (tokens < self.first_codebook_vocab_size),
            tokens,
            torch.zeros_like(tokens),
        )

    def _sanitize_sub_codebook(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.where(
            (tokens >= 0) & (tokens < self.sub_codebook_vocab_size),
            tokens,
            torch.zeros_like(tokens),
        )

    # ------------------------------------------------------------------
    # Audio token embedding
    # ------------------------------------------------------------------

    def _embed_audio_tokens_btk(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Sum embeddings across codebooks: ``(B, T, K) → (B, T, H)``."""
        first = self._sanitize_first_codebook(audio_tokens[:, :, 0])
        audio_emb = self.talker.get_input_embeddings()(first)
        sub_embeddings = self.talker.code_predictor.get_input_embeddings()
        for k in range(1, self.num_audio_codebooks):
            sub_tok = self._sanitize_sub_codebook(audio_tokens[:, :, k])
            audio_emb = audio_emb + sub_embeddings[k - 1](sub_tok)
        return audio_emb

    def _pad_logits(self, logits: torch.Tensor, target_vocab_size: int) -> torch.Tensor:
        if logits.size(-1) == target_vocab_size:
            return logits
        output = logits.new_full(
            (*logits.shape[:-1], target_vocab_size),
            float(self._extra_logit_floor.item()),
        )
        width = min(logits.size(-1), target_vocab_size)
        output[..., :width] = logits[..., :width]
        return output

    # ------------------------------------------------------------------
    # Sub-codebook prediction
    # ------------------------------------------------------------------

    def _sub_logits_teacher_forced(
        self, hidden_states: torch.Tensor, target_audio_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return sub-codebook logits ``(B, T, K-1, V)`` using teacher forcing."""
        bsz, steps, hidden = hidden_states.shape
        flat_hidden = hidden_states.reshape(bsz * steps, hidden)
        flat_codes = target_audio_tokens.reshape(bsz * steps, self.num_audio_codebooks)
        first_codes = self._sanitize_first_codebook(flat_codes[:, :1])
        sub_codes = self._sanitize_sub_codebook(flat_codes[:, 1:])

        input_embeds = [flat_hidden.unsqueeze(1), self.talker.get_input_embeddings()(first_codes)]
        sub_embeddings = self.talker.code_predictor.get_input_embeddings()
        for k in range(1, self.num_audio_codebooks - 1):
            input_embeds.append(sub_embeddings[k - 1](sub_codes[:, k - 1 : k]))
        input_embeds = torch.cat(input_embeds, dim=1)

        sub_outputs = self.talker.code_predictor.forward_finetune(inputs_embeds=input_embeds, labels=None)
        return sub_outputs.logits.reshape(bsz, steps, self.num_audio_codebooks - 1, -1)

    @torch.no_grad()
    def _sub_logits_greedy(
        self, hidden_states: torch.Tensor, first_code_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy sub-codebook decoding for inference (single frame)."""
        bsz, hidden = hidden_states.shape
        first_code_ids = self._sanitize_first_codebook(first_code_ids)
        embeds = [
            hidden_states.unsqueeze(1),
            self.talker.get_input_embeddings()(first_code_ids[:, None]),
        ]
        logits_per_codebook, generated = [], []
        sub_embeddings = self.talker.code_predictor.get_input_embeddings()
        for idx in range(self.num_audio_codebooks - 1):
            inputs_embeds = torch.cat(embeds, dim=1)
            out = self.talker.code_predictor(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                generation_steps=idx,
            )
            logits = out.logits[:, -1, :]
            pred = torch.argmax(logits, dim=-1)
            logits_per_codebook.append(logits)
            generated.append(pred)
            embeds.append(sub_embeddings[idx](pred[:, None]))
        return torch.stack(logits_per_codebook, dim=1), torch.stack([first_code_ids] + generated, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states,
        speech_mask,
        input_audio_tokens=None,
        target_text_tokens=None,
        target_audio_tokens=None,
        modality_adapter_emb=None,
        asr_emb=None,
        speaker_encoder_emb=None,
        temperature: float = 0.7,
        topk: int = 80,
        greedy: bool = True,
    ):
        if hidden_states is not None:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            if self.detach_input:
                hidden_states = hidden_states.detach()

        if self.use_input_cache:
            hidden_states = self._append_cache("hidden_states", hidden_states)
            speech_mask = self._append_cache("speech_mask", speech_mask)
            input_audio_tokens = self._append_cache("input_audio_tokens", input_audio_tokens)
            target_text_tokens = self._append_cache("target_text_tokens", target_text_tokens)
            modality_adapter_emb = self._append_cache("modality_adapter_emb", modality_adapter_emb)
            asr_emb = self._append_cache("asr_emb", asr_emb)

        if target_text_tokens is None:
            raise ValueError("Qwen3TTSTalkerSpeechDecoder requires target_text_tokens.")
        if input_audio_tokens is None:
            raise ValueError("Qwen3TTSTalkerSpeechDecoder requires input_audio_tokens.")

        if speech_mask is None:
            speech_mask = torch.ones(
                target_text_tokens.shape,
                device=target_text_tokens.device,
                dtype=torch.bool,
            )
        else:
            speech_mask = speech_mask.bool()

        # ---- Build conditioning signal ----
        text_emb = self.talker.text_projection(self.talker.get_text_embeddings()(target_text_tokens))
        condition = text_emb

        if self.cond_on_llm_latent and hidden_states is not None:
            llm_emb = self.input_proj(hidden_states) if self.input_proj is not None else hidden_states
            condition = condition + llm_emb

        if self.cond_on_modality_adapter_emb and modality_adapter_emb is not None:
            if self.detach_input:
                modality_adapter_emb = modality_adapter_emb.detach()
            condition = condition + self.modality_adapter_emb_projection(modality_adapter_emb)

        if self.cond_on_asr_emb and asr_emb is not None:
            if self.detach_input:
                asr_emb = asr_emb.detach()
            projected_asr = self.asr_emb_projection(asr_emb)
            if self.training and self.drop_asr_emb_prob and torch.rand(1).item() < self.drop_asr_emb_prob:
                projected_asr = torch.zeros_like(projected_asr)
            condition = condition + projected_asr

        # ---- Additive speaker conditioning ----
        if self.use_speaker_encoder:
            if self.use_input_cache and not self.training:
                speaker_encoder_emb = self.inference_speaker_embedding
                if self.use_random_spk_emb:
                    speaker_encoder_emb = speaker_encoder_emb.repeat(condition.size(0), condition.size(1), 1)
                if speaker_encoder_emb.size(1) != condition.size(1):
                    speaker_encoder_emb = speaker_encoder_emb.repeat(1, condition.size(1), 1)
            elif speaker_encoder_emb is not None and speaker_encoder_emb.size(1) != condition.size(1):
                speaker_encoder_emb = speaker_encoder_emb.repeat(1, condition.size(1), 1)

            if speaker_encoder_emb is not None:
                condition = condition + self.speaker_encoder_emb_projection(speaker_encoder_emb)

        # ---- Classifier-Free Guidance ----
        use_cfg_inference = bool(
            self.cfg_unconditional_prob
            and not self.training
            and abs(float(self.cfg_scale) - 1.0) > 1.0e-6
        )
        if self.cfg_unconditional_prob:
            if self.training:
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    condition = torch.zeros_like(condition)
            elif use_cfg_inference:
                zeros = torch.zeros_like(condition)
                condition = torch.cat([condition, zeros], dim=0)
                speech_mask = torch.cat([speech_mask, speech_mask], dim=0)
                input_audio_tokens = torch.cat([input_audio_tokens, input_audio_tokens], dim=0)

        # ---- Talker forward ----
        if self.cond_on_prev_audio_tokens:
            if self.detach_input:
                input_audio_tokens = input_audio_tokens.detach()
            talker_input = condition + self._embed_audio_tokens_btk(input_audio_tokens)
        else:
            talker_input = condition

        outputs = self.talker.model(
            input_ids=None,
            inputs_embeds=talker_input,
            attention_mask=speech_mask.long(),
            use_cache=False,
            return_dict=True,
        )
        talker_hidden = outputs.last_hidden_state
        first_logits = self._pad_logits(self.talker.codec_head(talker_hidden), self.num_audio_tokens_per_codebook)

        # Apply CFG to first-codebook logits
        if use_cfg_inference:
            batch_size = first_logits.size(0) // 2
            cond_first = first_logits[:batch_size]
            uncond_first = first_logits[batch_size:]
            first_logits = (1 - self.cfg_scale) * uncond_first + self.cfg_scale * cond_first

        # ---- Sub-codebook prediction ----
        if target_audio_tokens is not None:
            # Training: teacher-forced sub-codebook prediction.
            if use_cfg_inference:
                talker_hidden = talker_hidden[:batch_size]
            sub_logits = self._sub_logits_teacher_forced(talker_hidden, target_audio_tokens)
            sub_logits = self._pad_logits(sub_logits, self.num_audio_tokens_per_codebook)
            all_logits = torch.cat([first_logits.unsqueeze(2), sub_logits], dim=2)
            sampled_audio_tokens = None

        elif self.use_input_cache and not self.training:
            # Inference: greedy per-frame sub-codebook decoding.
            first_last = first_logits[:, -1, :]
            first_pred = torch.argmax(first_last, dim=-1)
            if use_cfg_inference:
                cond_hidden = talker_hidden[:batch_size, -1, :]
                uncond_hidden = talker_hidden[batch_size:, -1, :]
                cond_sub_logits, cond_generated = self._sub_logits_greedy(cond_hidden, first_pred)
                uncond_sub_logits, _ = self._sub_logits_greedy(uncond_hidden, first_pred)
                sub_logits = (1 - self.cfg_scale) * uncond_sub_logits + self.cfg_scale * cond_sub_logits
                generated = cond_generated
            else:
                sub_logits, generated = self._sub_logits_greedy(talker_hidden[:, -1, :], first_pred)
            sub_logits = self._pad_logits(sub_logits[:, None], self.num_audio_tokens_per_codebook)
            all_logits = torch.cat([first_last[:, None, None, :], sub_logits], dim=2)
            sampled_audio_tokens = [generated[:, i : i + 1] for i in range(generated.size(1))]

        else:
            # Fallback: pseudo-teacher-forced using input tokens.
            if use_cfg_inference:
                talker_hidden = talker_hidden[:batch_size]
            sub_logits = self._sub_logits_teacher_forced(talker_hidden, input_audio_tokens)
            sub_logits = self._pad_logits(sub_logits, self.num_audio_tokens_per_codebook)
            all_logits = torch.cat([first_logits.unsqueeze(2), sub_logits], dim=2)
            sampled_audio_tokens = None

        return all_logits.flatten(2, 3), sampled_audio_tokens
