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
"""S2S duplex training entry-point using Qwen3-TTS as codec + talker.

This script is the NeMo_S2S counterpart of the HUMAIN_NeMo Qwen3-codec training
script. It follows the same dataset pipeline as the base recipe
``s2s_duplex_speech_decoder_st_train_concat_v2.py`` (``DuplexS2SDatasetConcatV``),
while replacing:

* **Audio codec**  → ``Qwen3TTSCodecAdapter``  (12 Hz Qwen3 tokenizer)
* **Speech decoder** → ``Qwen3TTSTalkerSpeechDecoder`` (0.6B talker with
  Qwen3's own ECAPA-TDNN speaker encoder instead of TitaNet)

Relevant config overrides (set in the launch script):
  ++model.pretrained_audio_codec=<path/to/Qwen3-TTS-Tokenizer-12Hz>
  ++model.pretrained_qwen3_tts=<path/to/Qwen3-TTS-12Hz-0.6B-Base-hf>
  ++model.qwen3_talker_speech_vocab_size=3072
  ++model.qwen3_talker_raw_codec_vocab_size=2048
  ++model.custom_speech_bos_id=2149
  ++model.custom_speech_eos_id=2150
  ++model.custom_speech_delay_id=2148
  ++model.qwen3_clamp_target_token_lens=true
  ++model.speech_decoder.cond_on_llm_latent=false
"""

from __future__ import annotations

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

import nemo.collections.speechlm2.models.duplex_s2s_speech_decoder_model2 as speech_decoder_module
from nemo.collections.speechlm2 import DataModule, DuplexS2SDatasetConcatV
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from qwen3_tts_codec_adapter import Qwen3TTSCodecAdapter
from qwen3_tts_talker_decoder import Qwen3TTSTalkerSpeechDecoder

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# ---------------------------------------------------------------------------
# Codec patch — replace EnCodec with the Qwen3 12 Hz tokenizer.
# ---------------------------------------------------------------------------

def setup_qwen3_audio_codec(model: torch.nn.Module) -> None:
    """Install a frozen Qwen3-TTS tokenizer adapter as ``model.audio_codec``."""
    device_map = f"cuda:{torch.cuda.current_device()}"
    device = torch.device("cuda", torch.cuda.current_device())
    input_sr = int(model.cfg.get("qwen3_input_audio_sample_rate", model.target_sample_rate))
    output_sr = int(model.cfg.get("qwen3_output_audio_sample_rate", model.target_sample_rate))
    with fp32_precision():
        model.audio_codec = Qwen3TTSCodecAdapter(
            model_path_or_name=model.cfg.pretrained_audio_codec,
            device_map=device_map,
            audio_sample_rate=input_sr,
            output_audio_sample_rate=output_sr,
        ).to(device).float().eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False


# Monkey-patch only this entry-point; the default NeMo entry-point is unchanged.
speech_decoder_module.setup_audio_codec = setup_qwen3_audio_codec


# ---------------------------------------------------------------------------
# Qwen3 model subclass
# ---------------------------------------------------------------------------

class Qwen3CodecDuplexS2SSpeechDecoderModel(speech_decoder_module.DuplexS2SSpeechDecoderModel2):
    """``DuplexS2SSpeechDecoderModel2`` adapted for the Qwen3-TTS 12 Hz codec.

    Changes vs. the base class
    --------------------------
    1. **``speech_vocab_size``** — returns ``qwen3_talker_speech_vocab_size``
       (3072) instead of ``codebook_size + 3``.
    2. **``speech_generation``** — initialised as ``Qwen3TTSTalkerSpeechDecoder``
       (Qwen3 talker + ECAPA-TDNN speaker encoder) via
       ``init_speech_generation_from_qwen3_tts``.
    3. **``prepare_inputs``** — two extra post-processing steps:
       * *Frame-rate clamping*: ``target_token_lens`` is clamped to the minimum
         of the dataloader length, the actual Qwen 12 Hz codec output length,
         and the ASR encoder frame count.  This prevents loss masks from
         extending beyond valid Qwen codec frames.
       * *Sub-codebook masking*: special control tokens (BOS/EOS/delay) that
         appear in sub-codebooks (where they are out of range) are replaced
         with token-0 and their ``loss_scale`` is zeroed.
    4. **``forward``** — adds ``target_audio_tokens`` argument and passes it to
       ``speech_generation`` for teacher-forced hierarchical codebook training.
    5. **``training_step``** — passes ``audio_labels`` as ``target_audio_tokens``.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        if self.cfg.get("pretrained_qwen3_tts", None):
            self.init_speech_generation_from_qwen3_tts(self.cfg.pretrained_qwen3_tts)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def speech_vocab_size(self) -> int:
        if hasattr(self, "cfg") and self.cfg.get("qwen3_talker_speech_vocab_size", None):
            return int(self.cfg.qwen3_talker_speech_vocab_size)
        return super().speech_vocab_size

    # ------------------------------------------------------------------
    # Qwen3 speech-generation initialisation
    # ------------------------------------------------------------------

    def init_speech_generation_from_qwen3_tts(self, model_path_or_name: str) -> None:
        """Replace ``self.speech_generation`` with ``Qwen3TTSTalkerSpeechDecoder``."""
        device_map = f"cuda:{torch.cuda.current_device()}"
        self.speech_generation = Qwen3TTSTalkerSpeechDecoder(
            model_path_or_name=model_path_or_name,
            speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder, resolve=True),
            lantent_dim=self.llm.config.hidden_size,
            num_audio_codebooks=self._num_codebooks,
            speech_vocab_size=self.speech_vocab_size,
            raw_codec_vocab_size=int(self.cfg.get("qwen3_talker_raw_codec_vocab_size", self._codebook_size)),
            device_map=device_map,
            dtype=self.cfg.get("qwen3_talker_dtype", "bfloat16"),
            attn_implementation=self.cfg.get("qwen3_talker_attn_implementation", "eager"),
        )
        print(
            "QWEN3_TALKER_INIT "
            f"path={model_path_or_name} "
            f"speech_vocab_size={self.speech_vocab_size} "
            f"raw_codec_vocab_size={self.cfg.get('qwen3_talker_raw_codec_vocab_size', self._codebook_size)} "
            f"first_vocab={self.speech_generation.first_codebook_vocab_size} "
            f"sub_vocab={self.speech_generation.sub_codebook_vocab_size} "
            f"num_codebooks={self._num_codebooks} "
            f"speaker_embedding_dim={self.speech_generation.speaker_embedding_dim}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _ceil_div(num: torch.Tensor, den: int) -> torch.Tensor:
        return torch.div(num + den - 1, den, rounding_mode="floor")

    # ------------------------------------------------------------------
    # prepare_inputs
    # ------------------------------------------------------------------

    def prepare_inputs(self, batch: dict):
        """Extend the base ``prepare_inputs`` with two Qwen3-specific steps.

        Step 1 — Frame-rate clamping
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The Qwen3 tokenizer operates at ~12.5 Hz (not 50 Hz like EnCodec).
        The dataloader's ``target_token_lens`` is computed from audio duration
        and may slightly exceed the actual codec output length due to rounding.
        Clamping to ``min(dataloader_len, codec_len, source_encoder_len)``
        prevents out-of-bounds loss masks.

        Step 2 — Sub-codebook special-token masking
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Qwen3 uses BOS/EOS/delay IDs ≥ 2048 (the raw codec vocab size).
        These special tokens are valid in the *first* codebook but are
        *out of range* in sub-codebooks (which have vocab 0…2047).
        We zero them out in ``input_audio_tokens`` and ``audio_labels``,
        and also zero the corresponding ``loss_scale`` entries so that these
        positions do not contribute to the sub-codebook loss.
        """
        if self.cfg.get("qwen3_clamp_target_token_lens", True):
            batch = dict(batch)
            source_samples_per_frame = int(round(self.source_sample_rate / self.source_fps))

            # Compute actual Qwen codec frame count from raw audio lengths.
            if hasattr(self.audio_codec, "audio_lens_to_codec_frames"):
                qwen_target_lens = self.audio_codec.audio_lens_to_codec_frames(
                    batch["target_audio_lens"].long()
                )
            else:
                target_spf = int(self.audio_codec.samples_per_frame)
                qwen_target_lens = self._ceil_div(batch["target_audio_lens"].long(), target_spf)

            source_lens = self._ceil_div(batch["source_audio_lens"].long(), source_samples_per_frame)

            clamped = torch.minimum(
                batch["target_token_lens"].long(),
                qwen_target_lens.to(batch["target_token_lens"].device),
            )
            clamped = torch.minimum(clamped, source_lens.to(clamped.device))

            if (
                os.environ.get("DEBUG_QWEN3_CODEC_LENGTHS", "0") == "1"
                and os.environ.get("RANK", "0") == "0"
                and not getattr(self, "_printed_qwen3_len_debug", False)
            ):
                print(
                    "QWEN3_LENGTH_DEBUG "
                    f"target_token_lens={batch['target_token_lens'].detach().cpu().tolist()} "
                    f"qwen_target_lens={qwen_target_lens.detach().cpu().tolist()} "
                    f"source_lens={source_lens.detach().cpu().tolist()} "
                    f"clamped_lens={clamped.detach().cpu().tolist()}",
                    flush=True,
                )
                self._printed_qwen3_len_debug = True

            batch["target_token_lens"] = clamped

        prepared = super().prepare_inputs(batch)

        # Sub-codebook masking: zero any token ≥ raw_codec_vocab_size in
        # codebooks 2..K and suppress the corresponding loss.
        if self.cfg.get(
            "qwen3_talker_control_first_codebook_only",
            bool(self.cfg.get("pretrained_qwen3_tts", None)),
        ):
            raw_vocab = int(self.cfg.get("qwen3_talker_raw_codec_vocab_size", self._codebook_size))
            for key in ("input_audio_tokens", "audio_labels"):
                tokens = prepared[key].clone()
                invalid_sub = tokens[:, :, 1:] >= raw_vocab
                if invalid_sub.any():
                    tokens[:, :, 1:] = torch.where(
                        invalid_sub,
                        torch.zeros_like(tokens[:, :, 1:]),
                        tokens[:, :, 1:],
                    )
                    if key == "audio_labels":
                        loss_scale = prepared["loss_scale"].clone()
                        loss_scale[:, :, 2:] = torch.where(
                            invalid_sub,
                            torch.zeros_like(loss_scale[:, :, 2:]),
                            loss_scale[:, :, 2:],
                        )
                        prepared["loss_scale"] = loss_scale
                prepared[key] = tokens

        return prepared

    # ------------------------------------------------------------------
    # forward  (adds target_audio_tokens for teacher-forced sub-codebooks)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_embeds,
        cache=None,
        input_audio_tokens=None,
        seq_mask=None,
        target_text_tokens=None,
        target_audio_tokens=None,       # NEW — needed for Qwen3 teacher forcing
        modality_adapter_emb=None,
        asr_emb=None,
        speaker_encoder_emb=None,
    ) -> dict[str, torch.Tensor]:
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out["last_hidden_state"])

        if seq_mask is not None:
            seq_mask = seq_mask[:, :, -1].reshape(seq_mask.size(0), seq_mask.size(1))
            if self.speech_generation.use_input_cache:
                self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if self.speech_generation.use_input_cache and not self.training:
            if self.cfg.get("inference_pad_boost", None):
                text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
            if self.cfg.get("inference_bos_boost", None):
                text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost
            if self.cfg.get("inference_eos_boost", None):
                text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost

            target_text_tokens = torch.argmax(text_logits, dim=-1).view(B, T).contiguous()

            if self.cfg.get("convert_pad_to_extra_id_on_speech_decoder", None):
                target_text_tokens[target_text_tokens == self.text_pad_id] = (
                    self.tokenizer.tokenizer._tokenizer.token_to_id("<|endoftext|>")
                )
        else:
            drop_bos_prob = getattr(self.cfg, "drop_text_bos_prob", 0.0)
            if drop_bos_prob > 0.0:
                bos_mask = target_text_tokens == self.text_bos_id
                drop_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_bos_prob
                target_text_tokens = torch.where(bos_mask & drop_mask, self.text_pad_id, target_text_tokens)

            drop_eos_prob = getattr(self.cfg, "drop_text_eos_prob", 0.0)
            if drop_eos_prob > 0.0:
                eos_mask = target_text_tokens == self.text_eos_id
                drop_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_eos_prob
                target_text_tokens = torch.where(eos_mask & drop_mask, self.text_pad_id, target_text_tokens)

        audio_logits, _ = self.speech_generation(
            out["last_hidden_state"].transpose(0, 1),
            seq_mask,
            input_audio_tokens=input_audio_tokens,
            target_text_tokens=target_text_tokens,
            target_audio_tokens=target_audio_tokens,   # teacher-forcing for K>1 codebooks
            modality_adapter_emb=modality_adapter_emb,
            asr_emb=asr_emb,
            speaker_encoder_emb=speaker_encoder_emb,
        )

        audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)

        ans = {"text_logits": text_logits, "audio_logits": audio_logits}
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    # ------------------------------------------------------------------
    # training_step  (passes audio_labels as target_audio_tokens)
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
            if speech_decoder_module.is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            seq_mask=inputs["seq_mask"],
            target_text_tokens=inputs["text_labels"],
            target_audio_tokens=inputs["audio_labels"],     # teacher-forced sub-codebooks
            modality_adapter_emb=inputs["perception_emb"],
            asr_emb=inputs["asr_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )

        num_frames = inputs["input_lens"].sum()
        with speech_decoder_module.loss_parallel():
            text_logits = forward_outputs["text_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

            text_loss = (
                torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1),
                    inputs["text_labels"].flatten(0, 1),
                    reduction="none",
                )
                * inputs["loss_scale"][:, :, 0].flatten(0, 1)
            ).sum(-1) / num_frames

            audio_logits = forward_outputs["audio_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                audio_logits = audio_logits * inputs["seq_mask"][:, :, -1].unsqueeze(-1).unsqueeze(-1)

            audio_loss = (
                torch.nn.functional.cross_entropy(
                    audio_logits.flatten(0, 2),
                    inputs["audio_labels"].flatten(0, 2),
                    reduction="none",
                )
                * inputs["loss_scale"][:, :, 1:].flatten(0, 2)
            ).sum(-1) / (num_frames * self._num_codebooks)

        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "num_frames": num_frames.to(torch.float32),
            "padding_ratio": num_frames / (B * T),
        }
        self.log("batch_size", B, on_step=True, prog_bar=True, logger=True)
        self.log("sequence_length", T, on_step=True, prog_bar=True, logger=True)
        self.log_dict(ans, on_step=True)
        return ans


# Alias so checkpoints saved with the base class name can be loaded here.
DuplexS2SSpeechDecoderModel2 = Qwen3CodecDuplexS2SSpeechDecoderModel


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = Qwen3CodecDuplexS2SSpeechDecoderModel(OmegaConf.to_container(cfg, resolve=True))

    dataset = DuplexS2SDatasetConcatV(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
