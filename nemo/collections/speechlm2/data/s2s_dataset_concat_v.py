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

import torch
import torch.utils.data

from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging

_DEFAULT_LANG_MAP = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
}


class DuplexS2SDatasetConcatV(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

        add_lang_prompt (bool, optional):
            If True, prepend a language-direction instruction token sequence before the
            audio stream for each sample. The prompt is built from ``cut.custom["lang_src"]``
            and ``cut.custom["lang_tgt"]`` (e.g. ``"de"`` and ``"es"``) and tokenized as:
            ``<bos> System\\nTranslate from {src} to {tgt}. <eos>``.
            Silence audio is inserted in both channels to keep frame alignment.
            Default: ``False``.

        lang_map (dict, optional):
            Mapping from 2-letter ISO code to full language name used when building the
            prompt text. Defaults to ``{"en": "English", "de": "German",
            "es": "Spanish", "fr": "French"}``.

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]
            - lang_prompt: List of raw lang_pair strings per sample (empty string if disabled) [B]
            - prompt_lens: List of prompt lengths in tokens per sample (0 if disabled) [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        training_speaker_reference: str = None,
        training_speaker_duration: float = 3.0,
        add_lang_prompt: bool = False,
        lang_map: dict = None,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.training_speaker_duration = training_speaker_duration
        self.add_lang_prompt = add_lang_prompt
        self.lang_map = lang_map if lang_map is not None else _DEFAULT_LANG_MAP
        self.source_samples_per_frame = int(source_sample_rate * frame_length)
        self.target_samples_per_frame = int(target_sample_rate * frame_length)
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

        if training_speaker_reference is not None:
            import torchaudio  # optional dependency, only needed for this feature

            audio, sr = torchaudio.load(training_speaker_reference)
            if audio.shape[0] > 1:
                audio = audio[0:1, :]
            if sr != target_sample_rate:
                audio = torchaudio.functional.resample(audio, sr, target_sample_rate)
            max_samples = int(self.training_speaker_duration * target_sample_rate)
            audio = audio[:, :max_samples]
            self._fixed_spk_audio = audio.squeeze(0)
            logging.info(
                "Fixed training speaker reference loaded: %s (%d samples @ %dHz)",
                training_speaker_reference, self._fixed_spk_audio.shape[0], target_sample_rate,
            )
        else:
            self._fixed_spk_audio = None

    def _build_lang_prompt_tokens(self, src_lang: str, tgt_lang: str, device: torch.device) -> torch.Tensor:
        """
        Build a token sequence for a language direction prompt.

        Looks up ``src_lang`` and ``tgt_lang`` (e.g. ``"de"``, ``"es"``) in
        ``self.lang_map`` and tokenizes:

            <bos> System\\nTranslate from {src_name} to {tgt_name}. <eos>

        Falls back to ``<eos>`` only when a code is unknown or missing.
        """
        try:
            src_name = self.lang_map.get(src_lang.lower())
            tgt_name = self.lang_map.get(tgt_lang.lower())
            if src_name is None or tgt_name is None:
                raise ValueError(f"Unknown language code(s): {src_lang!r}, {tgt_lang!r}")
            prompt_text = f"System\nTranslate from {src_name} to {tgt_name}."
            ids = [self.tokenizer.bos] + self.tokenizer.text_to_ids(prompt_text) + [self.tokenizer.eos]
        except Exception as e:
            logging.warning(
                f"[DuplexS2SDatasetConcatV] Could not build lang prompt for {src_lang!r}->{tgt_lang!r}: {e}. Using EOS only."
            )
            ids = [self.tokenizer.eos]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)
        source_audio, decode_source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        vals = [float(c.custom['src_duration'])*self.source_sample_rate for c in cuts]
        source_audio_lens = torch.tensor(vals, dtype=decode_source_audio_lens.dtype, device=decode_source_audio_lens.device)
        if cuts[0].custom.get('target_audio') is not None:
            target_audio, target_audio_lens = collate_audio(
                cuts.resample(self.target_sample_rate), recording_field="target_audio"
            )
            
        else:
            target_audio, target_audio_lens = None, None
            
        target_tokens, target_token_lens = collate_token_channel(
                                            cuts, self.tokenizer, self.frame_length, roles=self.output_roles)
        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.input_roles
        )
        # extract speaker first turn audio for speaker conditioning
        if self._fixed_spk_audio is not None:
            batch_size = len(cuts)
            fixed_len = self._fixed_spk_audio.shape[0]
            first_turn_audio = (
                self._fixed_spk_audio.unsqueeze(0).expand(batch_size, -1).clone()
                .to(dtype=source_audio.dtype, device=source_audio.device)
            )
            first_turn_audio_lens = torch.full((batch_size,), fixed_len, dtype=torch.long, device=source_audio.device)
        else:
            first_turn_audio, first_turn_audio_lens = collate_first_turn_audio_source(
                cuts.resample(self.target_sample_rate), roles=self.input_roles, duration=self.training_speaker_duration
            )

        # --- optional language-direction prompt ---
        # Prompt tokens are returned as a SEPARATE field (DuplexSTT-style).
        # The model's prepare_inputs inserts them into source_encoded at the feature level,
        # so target_tokens/audio are NOT modified here — no loss masking needed.
        lang_prompts_raw = []
        prompt_token_lens_list = []
        prompt_ids_list = []
        if self.add_lang_prompt:
            pad_id = get_pad_id(self.tokenizer)
            for i, cut in enumerate(cuts):
                custom = cut.custom or {}
                src_lang = custom.get("lang_src", "")
                tgt_lang = custom.get("lang_tgt", "")
                prompt_ids = self._build_lang_prompt_tokens(src_lang, tgt_lang, device=target_tokens.device)
                prompt_ids_list.append(prompt_ids)
                prompt_token_lens_list.append(len(prompt_ids))
                lang_prompts_raw.append(f"{src_lang}-{tgt_lang}")
            prompt_tokens = collate_vectors(prompt_ids_list, padding_value=pad_id)
            prompt_token_lens = torch.tensor(prompt_token_lens_list, dtype=torch.long)
        else:
            lang_prompts_raw = [""] * len(cuts)
            prompt_tokens = None
            prompt_token_lens = torch.zeros(len(cuts), dtype=torch.long)

        return {
            "sample_id": [str(cut.id) for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "decode_source_audio_lens": decode_source_audio_lens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "first_turn_audio": first_turn_audio,
            "first_turn_audio_lens": first_turn_audio_lens,
            "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in cuts],
            "lang_prompt": lang_prompts_raw,
            "prompt_token_lens": prompt_token_lens,
            **( {"prompt_tokens": prompt_tokens} if prompt_tokens is not None else {} ),
        }


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_first_turn_audio_source(
    cuts: CutSet,
    roles: set[str],
    duration: float = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        # truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_audio()
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=duration).load_audio()
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
        cut: Cut,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
        pad_id: int = -1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"
    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    count = 0
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            if count == 0:
                text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))
                count += 1
            else:
                text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(" " + supervision.text))
                #text_ids = torch.as_tensor(tokenizer.text_to_ids(" " + supervision.text))
            start_pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if start_pos >= len(tokens):  # Changed from > to >= for robustness
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {start_pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue


            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)


            available_frames_for_text = eospos - start_pos


            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = start_pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - start_pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = start_pos + len(text_ids)  

            try:
                tokens[start_pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {start_pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # if eospos < len(tokens):
            #     tokens[eospos] = tokenizer.eos
    # eospos = compute_num_frames(supervision.end, ...) gives the *count* of complete frames
    # up to the end of the last agent supervision, so valid frame indices are 0..eospos-1.
    # Placing EOS at eospos (one-past-end) would put it exactly at the truncation boundary
    # in the training step (target_tokens is cut to target_codes.shape[1] == eospos frames),
    # so it would never appear in training labels.  Use eospos-1 (the last valid frame).
    eos_frame = min(max(eospos - 1, 0), len(tokens) - 1)
    tokens[eos_frame] = tokenizer.eos
    return tokens

def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces
