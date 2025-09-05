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
import torchaudio

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import fastcopy
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging


class DuplexS2SDatasetST(torch.utils.data.Dataset):
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
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, cuts: CutSet) -> dict:

        cuts = cuts.map(set_target_duration)
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        
        target_audio, target_audio_lens = collate_audio(
            cuts.resample(self.target_sample_rate), recording_field="target_recording"
        )

        target_tokens, target_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.output_roles, target_sample_rate=self.target_sample_rate, source_sample_rate=self.source_sample_rate
        )
        # source_tokens, source_token_lens = collate_token_channel(
        #     cuts, self.tokenizer, self.frame_length, roles=self.input_roles
        # )
        # extract target speaker first turn audio to uses for speaker conditioning
        # target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
        #     cuts.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
        # )

        return {
            "sample_id": [str(cut.id) for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            # "source_tokens": source_tokens,
            # "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(cut.custom['tgt_traj']) for cut in cuts
            ],
            #     "target_first_turn_audio": target_first_turn_audio,
            #     "target_first_turn_audio_lens": target_first_turn_audio_lens,
            "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in cuts],
        }

def set_target_duration(cut):
    if cut.custom['target_recording'].duration != cut.custom['tgt_duration']:
        cut.custom['target_recording'].duration = cut.custom['tgt_duration']
    return fastcopy(cut, custom={**cut.custom})

def set_cuts_duration(cut):
    cut.duration = cut.custom['tgt_duration']
    return fastcopy(cut, custom={**cut.custom})

def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_recording",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    target_sample_rate: int = 22050,
    source_sample_rate: int = 16000,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, pad_id=pad_id, roles=roles, target_sample_rate=target_sample_rate, source_sample_rate=source_sample_rate)
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
        target_sample_rate: int = 22050,
        source_sample_rate: int = 16000,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"
    # src_chunk_size = compute_num_frames(cut.custom['chunk_size_ms'], frame_length, cut.sampling_rate)
    # tgt_chunk_size = compute_num_frames(cut.custom['chunk_size_ms'], frame_length, self.target_sample_rate)
    src = compute_num_frames(cut.duration, frame_length, source_sample_rate)
    tgt = compute_num_frames(cut.custom['tgt_duration'], frame_length, target_sample_rate)
    
    
    src_alignments = cut.custom['src_merged_alignments']
    tgt_alignments = cut.custom['tgt_merged_alignments']
    total = src + tgt + len(src_alignments) + 1# to ensure sufficient buffer space 
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    assert len(src_alignments) == len(tgt_alignments), f"src_alignments and tgt_alignments have different lengths: {len(src_alignments)} != {len(tgt_alignments)}"
    offset = compute_num_frames(cut.supervisions[0].start, frame_length, source_sample_rate) # offset points to the latest next frame in the total buffer 
    prev_src_end_frame = offset - 1
    prev_tgt_end_frame = -1
    
    for i, (src_alignment, tgt_alignment) in enumerate(zip(src_alignments, tgt_alignments)):


        if src_alignment and tgt_alignment:
            # if both available update offset and add tgt text after src
            
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(tgt_alignment[0]))
            tgt_alig_frames = [tgt_alignment[1], tgt_alignment[1]+tgt_alignment[2]]
            src_alig_frames = [src_alignment[1], src_alignment[1]+src_alignment[2]]
            tgt_alig_frames = compute_num_frames(tgt_alig_frames[0], frame_length, target_sample_rate), compute_num_frames(tgt_alig_frames[1], frame_length, target_sample_rate)
            src_alig_frames = compute_num_frames(src_alig_frames[0], frame_length, source_sample_rate), compute_num_frames(src_alig_frames[1], frame_length, source_sample_rate)

            tgt_start_pos = offset + (src_alig_frames[1] - prev_src_end_frame)  # start from the next frame
            tgt_end_pos = tgt_start_pos + (tgt_alig_frames[1] - prev_tgt_end_frame) - 1 
        #tgt_txt_start_pos = tgt_start_pos + tgt_alig_frames[0] # start of the target text (explore this later)
        # this depends if we later want the target audio conditionally dependent on the text
            tgt_txt_start_pos = tgt_start_pos
            offset = tgt_end_pos + 1
            prev_src_end_frame = src_alig_frames[1]  
            prev_tgt_end_frame = tgt_alig_frames[1]
        elif src_alignment:
            # if only src available, update offset 
            src_alig_frames = [src_alignment[1], src_alignment[1]+src_alignment[2]]
            src_alig_frames = compute_num_frames(src_alig_frames[0], frame_length, source_sample_rate), compute_num_frames(src_alig_frames[1], frame_length, source_sample_rate)
            offset = offset + (src_alig_frames[1] - prev_src_end_frame) + 1
            prev_src_end_frame = src_alig_frames[1]
            continue
        
        elif tgt_alignment:
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(tgt_alignment[0]))
            tgt_alig_frames = [tgt_alignment[1], tgt_alignment[1]+tgt_alignment[2]]
            tgt_alig_frames = compute_num_frames(tgt_alig_frames[0], frame_length, target_sample_rate), compute_num_frames(tgt_alig_frames[1], frame_length, target_sample_rate)
            tgt_start_pos = offset
            tgt_end_pos = tgt_start_pos + (tgt_alig_frames[1] - prev_tgt_end_frame) - 1 

            tgt_txt_start_pos = tgt_start_pos
            prev_tgt_end_frame = tgt_alig_frames[1]
            offset = tgt_end_pos + 1
        else:
            continue
            

        if tgt_txt_start_pos >= len(tokens):  # Changed from > to >= for robustness
            logging.warning(
                f"Ill-constructed example: the beginning offset of a supervision {tgt_txt_start_pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
            )
            continue

        available_frames_for_text = tgt_end_pos - tgt_txt_start_pos + 1


        if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
            # Truncate text_ids to fit before the eos position.
            text_ids = text_ids[:available_frames_for_text]
        elif available_frames_for_text <= 0:
            # If there's no space for text (e.g., start >= end), use an empty sequence.
            text_ids = torch.tensor([], dtype=torch.long)

        endpos = tgt_txt_start_pos + len(text_ids)
        if endpos > len(tokens):
            trunc_len = len(tokens) - tgt_txt_start_pos
            logging.warning(
                f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
            )
            text_ids = text_ids[:trunc_len]
            endpos = tgt_txt_start_pos + len(text_ids)  

        try:
            tokens[tgt_txt_start_pos:endpos] = text_ids
        except Exception as e:
            raise RuntimeError(f"{tokens.shape=} {tgt_txt_start_pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

        if tgt_end_pos < len(tokens):
            tokens[tgt_end_pos] = tokenizer.eos
        else:
            tokens = torch.cat([tokens, torch.tensor([tokenizer.eos], dtype=torch.long)])
            
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
