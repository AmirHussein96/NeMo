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
"""Qwen3-TTS standalone duplex S2S speech decoder model.

A self-contained model class that does not inherit from
DuplexS2SSpeechDecoderModel2.

Differences from DuplexS2SSpeechDecoderModel2
----------------------------------------------
* **Codec**       – ``Qwen3TTSCodecAdapter`` (12 Hz Qwen3 tokenizer) replaces EnCodec.
* **Speech gen.** – ``Qwen3TTSTalkerSpeechDecoder`` (Qwen3 0.6B talker +
  ECAPA-TDNN speaker encoder) replaces ``TransformerARSpeechDecoder``.
* **prepare_inputs** – frame-rate clamping + sub-codebook special-token masking.
* **forward / training_step** – ``target_audio_tokens`` passed for teacher-forced
  hierarchical codebook prediction.
"""

from __future__ import annotations

import os
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules.qwen3_tts_codec_adapter import Qwen3TTSCodecAdapter
from nemo.collections.speechlm2.modules.qwen3_tts_talker_decoder import Qwen3TTSTalkerSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
    Safe for GPU execution.
    """
    B, T = tokens.shape
    tokens = tokens.clone()

    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]
    eos_pos = eos_indices[:, 1]
    new_pos = eos_pos + shift

    valid = new_pos < T
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        target_vals = tokens[b_idx, new_pos]
        safe = target_vals != eos_token_id

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


def setup_qwen3_audio_codec(model: torch.nn.Module) -> None:
    """Install a frozen Qwen3-TTS tokenizer adapter as ``model.audio_codec``.

    Intended to monkey-patch ``nemo.collections.speechlm2.parts.pretrained.setup_audio_codec``
    at the training entry-point *before* the model is constructed so that the
    standard ``__init__`` picks up the Qwen3 codec.
    """
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


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Qwen3CodecDuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    """Standalone duplex S2S model using the Qwen3-TTS 12 Hz codec.

    Config keys (set via Hydra overrides in the launch script)
    ----------------------------------------------------------
    model.pretrained_audio_codec           Path to Qwen3-TTS-Tokenizer-12Hz
    model.pretrained_qwen3_tts             Path to Qwen3-TTS-12Hz-0.6B-Base-hf
    model.qwen3_talker_speech_vocab_size   Effective vocab (default 3072)
    model.qwen3_talker_raw_codec_vocab_size  Raw codec vocab below special tokens (default 2048)
    model.qwen3_talker_dtype               Talker weight dtype (default "bfloat16")
    model.qwen3_talker_attn_implementation Attention backend (default "eager")
    model.qwen3_input_audio_sample_rate    Input audio SR for codec (default = target_sample_rate)
    model.qwen3_output_audio_sample_rate   Output audio SR for codec (default = target_sample_rate)
    model.qwen3_clamp_target_token_lens    Enable frame-rate clamping (default True)
    model.custom_speech_bos_id             2149
    model.custom_speech_eos_id             2150
    model.custom_speech_delay_id           2148
    """

    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to Qwen3CodecDuplexS2SSpeechDecoderModel as a Python dict to support "
            f"hyperparameter serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        self.source_fps = self.source_sample_rate / (self.source_sample_rate * cfg.data.frame_length)

        setup_qwen3_audio_codec(self)
        self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self.audio_codec.vector_quantizer.num_groups
        if self.cfg.get("custom_codebook_size", None):
            self._codebook_size = self.cfg.get("custom_codebook_size")

        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame
        self.interpolation_factor = self.target_fps / self.source_fps

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        if 'Qwen2.5' in self.cfg.pretrained_llm:
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.llm = llm.model
        self.lm_head = llm.lm_head
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        setup_speech_encoder(self)

        self.use_random_spk_emb = self.cfg.get("use_random_spk_emb", False)
        self.speech_generation = self.init_speech_generation_from_qwen3_tts(self.cfg.pretrained_qwen3_tts)

        self.embed_audio_tokens = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                for _ in range(self._num_codebooks)
            ]
        )
        self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)
        if self.cfg.get("pretrained_s2s_model", None):
            self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )
        self._use_fsdp = False
        self._use_tp = False

    # ------------------------------------------------------------------
    # Checkpoint initialisation helpers
    # ------------------------------------------------------------------

    def init_speech_generation_from_tts_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_speech_generation_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = {
                k.replace("model.speech_decoder.", "").replace("speech_generation.", ""): v
                for k, v in checkpoint_state.items()
                if "model.speech_decoder." in k or "speech_generation." in k
            }
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    def init_speech_generation_from_qwen3_tts(self, model_path_or_name: str) -> Qwen3TTSTalkerSpeechDecoder:
        """Build and return a ``Qwen3TTSTalkerSpeechDecoder`` for ``self.speech_generation``."""
        device_map = f"cuda:{torch.cuda.current_device()}"
        speech_generation = Qwen3TTSTalkerSpeechDecoder(
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
            f"first_vocab={speech_generation.first_codebook_vocab_size} "
            f"sub_vocab={speech_generation.sub_codebook_vocab_size} "
            f"num_codebooks={self._num_codebooks} "
            f"speaker_embedding_dim={speech_generation.speaker_embedding_dim}",
            flush=True,
        )
        return speech_generation

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def speech_vocab_size(self) -> int:
        """Return the effective speech vocab size.

        Uses ``qwen3_talker_speech_vocab_size`` from config when available;
        otherwise falls back to ``_codebook_size + 3`` (BOS/EOS/delay).
        """
        if hasattr(self, "cfg") and self.cfg.get("qwen3_talker_speech_vocab_size", None):
            return int(self.cfg.qwen3_talker_speech_vocab_size)
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_delay_id", None):
            return self.cfg.get("custom_speech_delay_id")
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        return get_pad_id(self.tokenizer)

    # ------------------------------------------------------------------
    # Noise augmentation
    # ------------------------------------------------------------------

    def add_noise_to_batch(
        self,
        batch_audio,
        noise_folder,
        snr_db=20,
        noise_prob_scale_user=0.3,
        noise_prob_scale_user_min_snr=-15,
        noise_prob_scale_user_max_snr=24,
        snr_measure_dur=0.0,
        noise_resample=True,
        noise_prob_low_pass=0.1,
    ):
        batch_size, audio_length = batch_audio.shape

        import glob

        import librosa
        import numpy as np
        import soundfile as sf
        from scipy.signal import butter, lfilter

        noise_files = [f for f in glob.glob(noise_folder + "/*.wav")]
        if not noise_files:
            raise ValueError(f"No noise files found in {noise_folder}")

        for i in range(batch_size):

            def get_scale_factor(signal, noise, snr_db):
                if snr_measure_dur > 0:
                    signal = signal[: int(snr_measure_dur * self.source_sample_rate)]
                    noise = noise[: int(snr_measure_dur * self.source_sample_rate)]
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8
                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(noise_prob_scale_user_min_snr, noise_prob_scale_user_max_snr),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor

            def get_noise(noise_files):
                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')
                if noise_resample and sr != self.source_sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.source_sample_rate)
                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)
                noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)
                scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
                noise_tensor = noise_tensor * scaling_factor
                return noise_tensor

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise3 = get_noise(noise_files)
            noise = torch.cat([noise, noise2, noise3], axis=0)

            if noise.size(0) < audio_length:
                repeat_times = (audio_length // noise.size(0)) + 1
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                noise = noise[start_idx : start_idx + audio_length]

            def butter_lowpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            def lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                b = torch.tensor(b, dtype=torch.float32).cuda()
                a = torch.tensor(a, dtype=torch.float32).cuda()
                y_cpu = lfilter(b.cpu().numpy(), a.cpu().numpy(), data.cpu().numpy())
                y_gpu = torch.tensor(y_cpu, dtype=torch.float32).cuda()
                return y_gpu

            if random.random() < noise_prob_low_pass:
                cutoff = 1000.0
                noise = lowpass_filter(noise, cutoff, self.source_sample_rate)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio

    # ------------------------------------------------------------------
    # prepare_inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _ceil_div(num: torch.Tensor, den: int) -> torch.Tensor:
        return torch.div(num + den - 1, den, rounding_mode="floor")

    def prepare_inputs(self, batch: dict):
        """Extend the base ``prepare_inputs`` with two Qwen3-specific steps.

        Step 1 — Frame-rate clamping
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Clamp ``target_token_lens`` to
        ``min(dataloader_len, qwen_codec_len, asr_encoder_len)`` so that loss
        masks never extend beyond valid Qwen 12 Hz codec frames.

        Step 2 — Sub-codebook special-token masking
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        BOS/EOS/delay token IDs (≥ raw_codec_vocab_size = 2048) are valid only
        in codebook 1.  Any such token appearing in codebooks 2..K is replaced
        with 0 and its ``loss_scale`` is set to 0 so it does not contribute to
        the sub-codebook loss.
        """
        if self.cfg.get("qwen3_clamp_target_token_lens", True):
            batch = dict(batch)
            source_samples_per_frame = int(round(self.source_sample_rate / self.source_fps))

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

        prepared = self._base_prepare_inputs(batch)

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

    def _base_prepare_inputs(self, batch: dict):
        """Core input preparation logic (ported from DuplexS2SSpeechDecoderModel2)."""
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["first_turn_audio"].size(0) == batch["target_audio"].size(0)

        if self.cfg.get('use_old_noise_aug', None):
            noise_prob = 0.99
            noise_min_snr = 20
            noise_max_snr = 50
            noise_path = self.cfg.get('old_noise_aug_path', None)
            noise_path_name = "*"
            no_noise_audio = batch["source_audio"].clone()
            if (
                self.training
                and batch["formatter"][0] != 's2s_duplex_overlap_as_s2s_duplex'
                and noise_prob
                and random.random() < noise_prob
            ):
                batch["source_audio"] = self.add_noise_to_batch(
                    batch["source_audio"],
                    os.path.join(noise_path, noise_path_name),
                    snr_db=random.randint(noise_min_snr, noise_max_snr),
                    noise_prob_scale_user=0.3,
                    noise_prob_scale_user_min_snr=-15,
                    noise_prob_scale_user_max_snr=24,
                    snr_measure_dur=0.0,
                    noise_resample=True,
                    noise_prob_low_pass=0.1,
                )
        else:
            if self.training and random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
                min_scale_val = self.cfg.get('noise_scale_user_min', 0.0631)
                max_scale_val = self.cfg.get('noise_scale_user_min', 5.6234)
                scaling_factor = (
                    torch.rand(batch["source_audio"].size(0), device=batch["source_audio"].device)
                    * (max_scale_val - min_scale_val)
                    + min_scale_val
                )
                batch["source_audio"] = batch["source_audio"] * scaling_factor.unsqueeze(-1)

            if self.training and random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
                cutoff_freq = self.cfg.get('noise_low_pass_cutoff_freq', 1000.0)
                batch["source_audio"] = torchaudio.functional.lowpass_biquad(
                    waveform=batch["source_audio"], sample_rate=self.source_sample_rate, cutoff_freq=cutoff_freq
                )

        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        if not self.training:
            speaker_encoder_emb = None
        else:
            if self.speech_generation.use_speaker_encoder:
                first_turn_audio = batch["first_turn_audio"]
                first_turn_audio_lens = batch["first_turn_audio_lens"]
                speaker_encoder_emb = self.speech_generation.get_speaker_embedding(
                    first_turn_audio, first_turn_audio_lens, self.target_sample_rate
                )
            else:
                speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device)
                        * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
        target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                asr_emb = asr_emb[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        btt = target_tokens[..., None]
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        if self.advance_text_channel_by:
            pad = torch.full(
                (target_tokens.shape[0], self.advance_text_channel_by),
                fill_value=self.text_pad_id,
                device=target_tokens.device,
                dtype=torch.long,
            )
            target_tokens = torch.cat([target_tokens[:, self.advance_text_channel_by :], pad], dim=-1)

        if self.cfg.get("delay_text_eos_by", None):
            target_tokens = delay_eos(
                target_tokens, self.text_eos_id, self.text_pad_id, shift=self.cfg.delay_text_eos_by
            )

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]
                asr_emb = asr_emb[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]
        text_labels = input_ids[:, 1:, -1]
        audio_inputs = input_ids[:, :-1, :-1]
        audio_labels = input_ids[:, 1:, :-1]

        input_embeds = self.embed_tokens(text_inputs)
        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

        seq_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        if self.cfg.get("mask_sequence_loss", True):
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                seq_mask[i, speech_end_idx:, :] = 0

            mask_lengths = seq_mask[:, :, 0].sum(-1)
            assert torch.allclose(batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0)

        loss_scale = seq_mask.clone().float()

        if self.cfg.get("scale_loss_by") == 'non_sil_t':
            loss_scale[:, :, :1] = torch.where(
                text_labels.unsqueeze(-1) != self.text_pad_id,
                self.cfg.get("scale_loss_mask", self.cfg.get("nonsil_weight", 4.0)),
                loss_scale[:, :, :1],
            )

        if (
            self.cfg.get("debug_dataloader_audios_path", None)
            and self.training
            and "s2s_duplex_overlap_as_s2s_duplex" not in batch["formatter"][0]
        ):

            def count_leading_silence_tokens(tensor: torch.Tensor, silence_token: int = 0) -> int:
                if tensor.ndim != 1:
                    raise ValueError("Input tensor must be 1D.")
                count = 0
                for token in tensor:
                    if token.item() == silence_token:
                        count += 1
                    else:
                        break
                return count

            def write_wave(one_audio_signal, file_name, sr=None):
                import numpy as np
                import soundfile as sf

                one_audio_signal = one_audio_signal.cpu().numpy()
                one_audio_signal = one_audio_signal.astype(np.float32)
                if sr is None:
                    sr = self.target_sample_rate
                sf.write(file_name, one_audio_signal, sr)

            with fp32_precision(), torch.no_grad():
                lengths = torch.tensor([batch["target_audio"].shape[1]] * batch["target_audio"].shape[0]).to(
                    self.audio_codec.device
                )
                reconstructed_audio_from_wav, _ = self.audio_codec(audio=batch["target_audio"], audio_len=lengths)
                audio_labels_ = replace_control_speech_codes(audio_labels, self._control_codes)
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([audio_labels_.shape[1]] * audio_labels_.shape[0]).to(
                        self.audio_codec.device
                    )
                    reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                        tokens=audio_labels_.transpose(1, 2), tokens_len=lengths
                    )

            for i in range(audio_labels_.shape[0]):
                write_wave(
                    batch["target_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["first_turn_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"speaker_ref_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["source_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"source_audio_{i}.wav"),
                    sr=self.source_sample_rate,
                )
                write_wave(
                    reconstructed_audio_from_tokens[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"),
                        f"target_audio_reconstructed_from_tokens_{i}.wav",
                    ),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    reconstructed_audio_from_wav[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"),
                        f"target_audio_reconstructed_from_waveform_{i}.wav",
                    ),
                    sr=self.target_sample_rate,
                )

            num_bos_tokens = (text_labels.unsqueeze(-1) == self.text_bos_id).flatten(1, 2).sum(-1)
            num_eos_tokens = (text_labels.unsqueeze(-1) == self.text_eos_id).flatten(1, 2).sum(-1)
            print("Num eos:", num_eos_tokens, "num bos:", num_bos_tokens)
            print(
                "text_labels decoded:",
                tokens_to_str(
                    text_labels[-1:], target_codes_lens - 1, tokenizer=self.tokenizer, pad_id=self.text_pad_id
                ),
            )
            print(
                "target labels from dataloader decoded:",
                tokens_to_str(
                    batch["target_tokens"][-1:],
                    target_codes_lens - 1,
                    tokenizer=self.tokenizer,
                    pad_id=self.text_pad_id,
                ),
            )
            print(
                "Number of padding tokens on the begining:",
                count_leading_silence_tokens(text_labels[-1:].squeeze(), self.text_pad_id),
            )
            print(batch["formatter"])
            if audio_labels_.shape[0] > 1:
                exit()

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "seq_mask": seq_mask,
            "loss_scale": loss_scale,
            "perception_emb": source_encoded[:, :-1],
            "asr_emb": asr_emb[:, :-1],
            "speaker_encoder_emb": speaker_encoder_emb,
        }

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_embeds,
        cache=None,
        input_audio_tokens=None,
        seq_mask=None,
        target_text_tokens=None,
        target_audio_tokens=None,
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
            target_audio_tokens=target_audio_tokens,
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
    # training_step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            seq_mask=inputs["seq_mask"],
            target_text_tokens=inputs["text_labels"],
            target_audio_tokens=inputs["audio_labels"],
            modality_adapter_emb=inputs["perception_emb"],
            asr_emb=inputs["asr_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )

        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
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

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        # Keep frozen modules in fp32 eval mode.
        # We avoid reloading from disk here because Lustre I/O completes at
        # different speeds per rank, causing NCCL BROADCAST timeouts on multi-node jobs.
        self.audio_codec.float().eval()
        for p in self.audio_codec.parameters():
            p.requires_grad = False

        if hasattr(self.speech_generation, "use_speaker_encoder") and self.speech_generation.use_speaker_encoder:
            # Lightweight fp32 eval guard for ALL speaker encoder types.
            # We never reload from disk here — even a single .nemo file read on Lustre
            # has variable latency per node, causing the first validation ALLREDUCE to
            # time out while slow ranks are still in the load call.
            # fp32 is guaranteed during inference by torch.autocast(dtype=float32) in
            # _get_titanet_speaker_embedding, and by .float() called below for both types.
            speaker_enc = getattr(self.speech_generation, "speaker_encoder", None)
            if speaker_enc is not None:
                speaker_enc.float().eval()
                for p in speaker_enc.parameters():
                    p.requires_grad = False

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()

        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()
        tolerance = int(self.cfg.get("val_acc_tolerance", 160) / (1000 / self.target_fps))
        self.text_bos_acc = TokenAccuracy(
            token_name="text_bos", token_id=self.text_bos_id, tolerance=tolerance
        ).reset()
        self.text_eos_acc = TokenAccuracy(
            token_name="text_eos", token_id=self.text_eos_id, tolerance=tolerance
        ).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_bos_acc = self.text_bos_acc.compute()
        for k, m in text_bos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_eos_acc = self.text_eos_acc.compute()
        for k, m in text_eos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):
        if self.speech_generation.use_speaker_encoder and self.use_random_spk_emb:
            self.speech_generation.update_inference_speaker_embedding(
                self.speech_generation.inference_speaker_reference
            )
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue
            if self.speech_generation.use_speaker_encoder and not self.use_random_spk_emb:
                first_turn_audio = dataset_batch["first_turn_audio"]
                first_turn_audio_lens = dataset_batch["first_turn_audio_lens"]
                speaker_encoder_emb = self.speech_generation.get_speaker_embedding(
                    first_turn_audio, first_turn_audio_lens, self.target_sample_rate
                )
                self.speech_generation.update_inference_speaker_embedding_from_embedding(speaker_encoder_emb)

            results = self.offline_inference(dataset_batch, speaker_encoder_emb=speaker_encoder_emb)

            with fp32_precision():
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            self.text_bos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])
            self.text_eos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_bos_embedding(self) -> torch.Tensor:
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        dataset_batch: dict,
        speaker_encoder_emb: torch.Tensor = None,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive prediction."""
        input_signal = dataset_batch["source_audio"]
        input_signal_lens = dataset_batch["source_audio_lens"]
        if self.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            input_signal, sr = torchaudio.load(self.cfg.custom_sample_inference)
            input_signal = input_signal.to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
        B, T_local, H = source_encoded.shape

        T_tensor = torch.tensor([T_local], device=source_encoded.device)
        if self._use_fsdp:
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)

        T_config = self.cfg.get("inference_tgt_len", 1.2 * T_tensor.item())
        T = int(T_config)

        if T > T_local:
            last_frame_source = source_encoded[:, T_local - 1 : T_local, :]
            pad_source = last_frame_source.repeat(1, T - T_local, 1)
            source_encoded = torch.cat([source_encoded, pad_source], dim=1)

            last_frame_asr = asr_emb[:, T_local - 1 : T_local, :]
            pad_asr = last_frame_asr.repeat(1, T - T_local, 1)
            asr_emb = torch.cat([asr_emb, pad_asr], dim=1)

        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        cache = DynamicCache()
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(
            input_embeds[:, :1],
            cache=cache,
            input_audio_tokens=first_audio,
            seq_mask=None,
            target_text_tokens=None,
            modality_adapter_emb=source_encoded[:, :1],
            asr_emb=asr_emb[:, :1],
            speaker_encoder_emb=speaker_encoder_emb,
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        speech_state = torch.zeros(B, device=self.device, dtype=torch.long)
        gen_audio_len = torch.full((B,), T, device=self.device, dtype=input_signal_lens.dtype)
        gen_text_len = torch.full((B,), T, device=self.device, dtype=input_signal_lens.dtype)
        audio_done = torch.zeros(B, dtype=torch.bool, device=self.device)
        txt_done = torch.zeros(B, dtype=torch.bool, device=self.device)

        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb

            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(
                input_embeds[:, t : t + 1],
                cache=ans["cache"],
                input_audio_tokens=current_audio,
                seq_mask=None,
                target_text_tokens=None,
                modality_adapter_emb=source_encoded[:, t : t + 1],
                asr_emb=asr_emb[:, t : t + 1],
                speaker_encoder_emb=speaker_encoder_emb,
            )
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)
            if self.cfg.get('inference_force_speech_state', None):
                speech_state = torch.where(
                    gen_text[:, t] == self.text_bos_id, torch.ones_like(speech_state), speech_state
                )
                speech_state = torch.where(
                    gen_text[:, t] == self.text_eos_id, torch.zeros_like(speech_state), speech_state
                )
                gen_audio[:, t] = torch.where(
                    speech_state.unsqueeze(-1) == 0,
                    gen_audio[:, 0],
                    gen_audio[:, t],
                )
            num_speech_delay = 1
            if self.cfg.get('inference_force_speech_bos', None) and num_speech_delay < gen_text.shape[1]:
                gen_audio[:, t] = torch.where(
                    (gen_text[:, t - num_speech_delay].unsqueeze(-1) == self.text_bos_id)
                    * (torch.sum(gen_audio[:, t - num_speech_delay :] == self.speech_bos_id, 1) == 0),
                    self.speech_bos_id,
                    gen_audio[:, t],
                )

            if self.cfg.get('inference_force_speech_eos', None) and gen_text.shape[1] > num_speech_delay + self.cfg.get("advance_text_channel_by", 0):
                gen_audio[:, t] = torch.where(
                    (
                        gen_text[:, t - num_speech_delay - self.cfg.get("advance_text_channel_by", 0)].unsqueeze(-1)
                        == self.text_eos_id
                    ),
                    self.speech_eos_id,
                    gen_audio[:, t],
                )

            speech_done = (gen_audio[:, t] == self.speech_eos_id).any(dim=1)
            text_done = gen_text[:, t] == self.text_eos_id
            newly_speech_done = (~audio_done) & speech_done
            newly_text_done = (~txt_done) & text_done
            gen_audio_len[newly_speech_done] = t + 1
            gen_text_len[newly_text_done] = t + 1
            audio_done |= newly_speech_done
            txt_done |= newly_text_done
            if audio_done.all() and txt_done.all():
                break

        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, gen_text_len, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": dataset_batch["decode_source_audio_lens"],
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=gen_audio_len
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if self.cfg.get("custom_sample_inference", None):
            print(ans["audio"].shape, input_signal.shape)
            self.results_logger.merge_and_save_audio(
                self.cfg.custom_sample_inference + "inf.wav",
                pred_audio=ans["audio"][0],
                pred_audio_sr=self.target_sample_rate,
                user_audio=input_signal[0],
                user_audio_sr=self.source_sample_rate,
            )
            exit()
        return ans

    # ------------------------------------------------------------------
    # Optimizer / parallelism
    # ------------------------------------------------------------------

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

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
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
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
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
