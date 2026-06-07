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
"""Small NeMo-compatible adapter around the Qwen3-TTS 12 Hz tokenizer codec."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def _read_nested(obj, names: Sequence[str], default=None):
    cur = obj
    for name in names:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(name, default)
        else:
            cur = getattr(cur, name, default)
    return cur


class Qwen3TTSCodecAdapter(torch.nn.Module):
    """Expose Qwen3TTSTokenizer with the minimal codec API used by S2S training.

    The adapter returns encoded tokens in NeMo codec layout: ``(B, K, T)``.
    Qwen's tokenizer internally emits per-sample tensors in layout ``(T, K)``.
    """

    def __init__(
        self,
        model_path_or_name: str,
        device_map: str = "cuda:0",
        audio_sample_rate: int | None = None,
        output_audio_sample_rate: int | None = None,
    ):
        super().__init__()
        from qwen_tts import Qwen3TTSTokenizer

        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(model_path_or_name, device_map=device_map)
        self.model = self.tokenizer.model
        self.input_sample_rate = int(self.model.get_input_sample_rate())
        self.output_sample_rate = int(self.model.get_output_sample_rate())
        self.audio_sample_rate = int(audio_sample_rate or self.input_sample_rate)
        self.output_audio_sample_rate = int(output_audio_sample_rate or self.output_sample_rate)
        self.codec_samples_per_frame = int(self.model.get_encode_downsample_rate())
        self.codec_frame_rate = self.input_sample_rate / self.codec_samples_per_frame
        self.samples_per_frame = int(round(self.output_audio_sample_rate / self.codec_frame_rate))

        config = getattr(self.model, "config", None)
        num_groups = _read_nested(config, ["encoder_valid_num_quantizers"], None)
        if num_groups is None:
            num_groups = _read_nested(config, ["decoder_config", "num_quantizers"], None)
        codebook_size = _read_nested(config, ["encoder_config", "codebook_size"], None)
        if codebook_size is None:
            codebook_size = _read_nested(config, ["decoder_config", "codebook_size"], None)

        self.vector_quantizer = SimpleNamespace(
            num_groups=int(num_groups),
            codebook_size_per_group=int(codebook_size),
        )
        self.discriminator = None
        self._dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def _resample_if_needed(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.input_sample_rate:
            return audio
        return torchaudio.functional.resample(audio.float(), sr, self.input_sample_rate)

    def _expected_num_frames(self, num_samples: int) -> int:
        return max(1, (int(num_samples) + self.codec_samples_per_frame - 1) // self.codec_samples_per_frame)

    def audio_lens_to_codec_frames(self, audio_len: torch.Tensor) -> torch.Tensor:
        """Map input-audio sample lengths to Qwen codec frame lengths."""
        audio_len = audio_len.long()
        if self.audio_sample_rate == self.input_sample_rate:
            resampled_lens = audio_len
        else:
            resampled_lens = torch.div(
                audio_len * self.input_sample_rate + self.audio_sample_rate - 1,
                self.audio_sample_rate,
                rounding_mode="floor",
            )
        frame_lens = torch.div(
            resampled_lens + self.codec_samples_per_frame - 1,
            self.codec_samples_per_frame,
            rounding_mode="floor",
        )
        return torch.clamp(frame_lens, min=1)

    @staticmethod
    def _match_num_frames(codes: torch.Tensor, expected_len: int) -> torch.Tensor:
        if codes.shape[0] == expected_len:
            return codes
        if codes.shape[0] > expected_len:
            return codes[:expected_len]
        pad_len = expected_len - codes.shape[0]
        if codes.shape[0] == 0:
            pad = torch.zeros(pad_len, codes.shape[1], dtype=codes.dtype, device=codes.device)
        else:
            pad = codes[-1:].repeat(pad_len, 1)
        return torch.cat([codes, pad], dim=0)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, audio_len: torch.Tensor):
        """Encode waveform batch.

        Args:
            audio:     Float tensor ``(B, S)`` sampled at ``self.audio_sample_rate``.
            audio_len: Long tensor  ``(B,)``  lengths in input samples.

        Returns:
            tokens:     Long tensor ``(B, K, T)``.
            tokens_len: Long tensor ``(B,)`` lengths in codec frames.
        """
        output_device = audio.device
        audio = audio.detach().float().cpu()
        audio_len = audio_len.detach().long().cpu()
        wavs, expected_lens = [], []
        for wav, length in zip(audio, audio_len):
            wav = wav[: int(length)]
            wav = self._resample_if_needed(wav.unsqueeze(0), self.audio_sample_rate).squeeze(0)
            wavs.append(wav.numpy().astype(np.float32))
            expected_lens.append(self._expected_num_frames(wav.numel()))

        encoded = self.tokenizer.encode(wavs, sr=self.input_sample_rate, return_dict=True)
        codes_list = [
            self._match_num_frames(x.detach().long(), el)
            for x, el in zip(encoded.audio_codes, expected_lens)
        ]
        tokens_len = torch.tensor(expected_lens, dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(codes_list, batch_first=True, padding_value=0)
        tokens = padded.transpose(1, 2).contiguous()
        return tokens.to(output_device), tokens_len.to(output_device)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, tokens_len: torch.Tensor):
        """Decode NeMo-layout codec tokens ``(B, K, T)`` to a waveform batch."""
        tokens = tokens.detach().long()
        tokens_len = tokens_len.detach().long().cpu()
        codes_list = []
        codebook_size = int(self.vector_quantizer.codebook_size_per_group)
        for code, length in zip(tokens, tokens_len):
            code = code[:, : int(length)].transpose(0, 1).contiguous()
            code = torch.where(
                (code >= 0) & (code < codebook_size),
                code,
                torch.zeros_like(code),
            )
            codes_list.append(code.to(self.model.device if hasattr(self.model, "device") else tokens.device))

        wavs, sr = self.tokenizer.decode({"audio_codes": codes_list})
        wav_tensors = [torch.from_numpy(np.asarray(wav, dtype=np.float32)) for wav in wavs]
        audio_lens = torch.tensor([wav.numel() for wav in wav_tensors], dtype=torch.long)
        max_len = int(audio_lens.max()) if wav_tensors else 0
        padded = torch.stack([F.pad(wav, (0, max_len - wav.numel())) for wav in wav_tensors])

        if sr != self.output_sample_rate:
            raise RuntimeError(f"Unexpected Qwen decoded sample rate: {sr}, expected {self.output_sample_rate}")
        if sr != self.output_audio_sample_rate:
            padded = torchaudio.functional.resample(padded.float(), sr, self.output_audio_sample_rate)
            audio_lens = torch.div(
                audio_lens * self.output_audio_sample_rate + sr - 1,
                sr,
                rounding_mode="floor",
            )
        return padded.to(tokens.device), audio_lens.to(tokens.device)
