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
import os
import random
import tempfile

import torch
import torch.distributed as dist
import torch.utils.checkpoint  # used by the optional RNNT-loss activation checkpoint path
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
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector as NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.label_prep import maybe_prepend_prompt_tokens
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.optim_setup import (
    configure_optimizers,
    is_frozen,
    snapshot_frozen_param_fingerprints,
    verify_frozen_params_unchanged,
)
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_rnnt_decoder_joint,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def _get_rnnt_loss_class():
    """Lazy import for the standard NeMo RNNT loss wrapper (warprnnt/numba/pytorch backends
    auto-resolved). Keeps this module importable on environments without an RNNT loss backend
    when ``model.use_rnnt_loss`` is unset/false."""
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    return RNNTLoss


def _get_rnnt_decoding_module():
    """Lazy import for RNNTDecoding and RNNTBPEDecoding (same stack as speech_to_text_streaming_infer)."""
    from nemo.collections.asr.parts.submodules.rnnt_decoding import (
        RNNTBPEDecoding,
        RNNTBPEDecodingConfig,
        RNNTDecoding,
        RNNTDecodingConfig,
    )

    return RNNTDecoding, RNNTDecodingConfig, RNNTBPEDecoding, RNNTBPEDecodingConfig


class NemotronVoiceTranslateSTT(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to NemotronVoiceTranslateSTT as a Python dict to support hyperparameter "
            f"serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )
        self.target_fps = self.source_fps

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        # Handle different model types with all their specific configurations
        if 'Riva-Translate-4B-Instruct' in self.cfg.pretrained_llm:
            # ====== Riva-Translate-4B-Instruct-SPECIFIC HANDLING ======
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'
            elif self.tokenizer.pad_token == self.tokenizer.eos_token:
                # The checkpoint's own tokenizer_config.json is inconsistent across
                # Riva-Translate releases: v1.1 has no pad_token set at all (HF leaves it
                # None -> get_pad_id() safely falls back to <unk>, distinct from bos/eos), but
                # v2 explicitly sets pad_token="</s>", i.e. identical to eos_token. When
                # pad_token == eos_token, every silent/non-speaking frame in the duplex
                # sequence (the majority of frames) gets labeled and embedded as EOS, so the
                # model learns to predict EOS almost everywhere. Loss still looks fine
                # (dominated by the audio-token loss) but generated text collapses to
                # near-immediate EOS, causing BLEU == 0. Only override in this specific
                # colliding case (leaving v1.1 and any other well-behaved checkpoint
                # untouched, to stay compatible with already-trained checkpoints); use the
                # shared vocab's dedicated, otherwise-unused '<pad>' token instead.
                self.tokenizer.pad_token = '<pad>'
        
        if 'Qwen2.5' in self.cfg.pretrained_llm:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder(self)
        # Optionally load a pretrained RNNT decoder + joint (set cfg.pretrained_rnnt_asr, or
        # cfg.pretrained_rnnt_weights for the pre-extracted-bundle fast path). No-op (all three
        # attributes set to None) when neither config key is set.
        setup_rnnt_decoder_joint(self)

        # Optional auxiliary RNNT-ASR objective: an additive loss (total = text_loss + rnnt_loss)
        # trained on source-language transcripts, alongside (not instead of) the existing
        # translation/text loss. Gated on `use_rnnt_loss` (default False) so this block is a
        # complete no-op -- no extra module, no extra import -- for every recipe that doesn't
        # set it.
        self.rnnt_loss = None
        if self.cfg.get("use_rnnt_loss", False):
            if self.rnnt_joint is None or self.rnnt_decoder is None:
                raise ValueError(
                    "model.use_rnnt_loss=true requires both rnnt_decoder and rnnt_joint to be "
                    "loaded; set model.pretrained_rnnt_asr to a NeMo ASR checkpoint exposing "
                    "decoder+joint (or model.pretrained_rnnt_weights for the pre-extracted bundle)."
                )
            if self.rnnt_tokenizer is None:
                raise ValueError(
                    "model.use_rnnt_loss=true requires the pretrained ASR checkpoint to ship a "
                    "tokenizer (needed to BPE-encode batch['source_texts'] into RNNT joint targets)."
                )
            # NeMo convention: RNNTLoss(num_classes=...) expects the BLANK index, which equals
            # vocab_size_without_blank, i.e. num_classes_with_blank - 1 (blank is the last id).
            blank_idx = self.rnnt_joint.num_classes_with_blank - 1
            self.rnnt_loss = _get_rnnt_loss_class()(num_classes=blank_idx, reduction='mean_batch')
            # The joint loaded from the pretrained checkpoint defaults to `_fuse_loss_wer=True`
            # (NeMo's standard RNNT recipe co-fuses joint+loss+WER for memory). We use the
            # non-fused path: rnnt_joint(encoder_outputs, decoder_outputs) -> logits, then apply
            # RNNTLoss separately. Without this, joint.forward() would take the fused branch and
            # raise "`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided!".
            if hasattr(self.rnnt_joint, "set_fuse_loss_wer"):
                self.rnnt_joint.set_fuse_loss_wer(False)
            else:  # very old NeMo with no public setter; fall back to private attribute
                self.rnnt_joint._fuse_loss_wer = False
            logging.info(
                "use_rnnt_loss=True -> constructed RNNTLoss(blank_idx=%d, vocab_with_blank=%d) "
                "and forced rnnt_joint._fuse_loss_wer=False for non-fused training. "
                "Set model.rnnt_loss_weight to scale its contribution in training_step.",
                blank_idx,
                self.rnnt_joint.num_classes_with_blank,
            )

        if self.cfg.get("pretrained_s2s_model", None):
            self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        self._use_fsdp = False
        self._use_tp = False

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

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
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        seq_mask=None,
    ) -> dict[str, Tensor]:
        """
        Text-only forward pass through the LLM.
        Returns text_logits (B, T, vocab_size) and optionally KV-cache.
        """
       
        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)
        ans = {
            "text_logits": text_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]

        return ans

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

                # resample noise from sr to self.cfg.data.train_ds.sample_rate
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
                # For a 1D tensor, we want to repeat its elements.
                # If noise has other dimensions, adjust the repeat_times_tuple accordingly.
                # e.g., if noise is (C, L), and we want to repeat along L,
                # repeat_times_tuple = (1, repeat_times)
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                # If noise is a PyTorch tensor
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                # Or if noise was originally a list/numpy array and you want to keep Python's random
                # start_idx = random.randint(0, len(noise) - audio_length)
                noise = noise[start_idx : start_idx + audio_length]

            # Function to create a low-pass filter
            def butter_lowpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            # Function to apply the low-pass filter to data (tmp impl on cpu)
            def lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                b = torch.tensor(b, dtype=torch.float32).cuda()
                a = torch.tensor(a, dtype=torch.float32).cuda()
                # Apply the filter using lfilter function from scipy..numpysig.numpynal (CPU)
                y_cpu = lfilter(b.cpu().numpy(), a.cpu().numpy(), data.cpu().numpy())
                # Convert the filtered data back to torch tensor and move to GPU.numpy
                y_gpu = torch.tensor(y_cpu, dtype=torch.float32).cuda()
                return y_gpu

            if random.random() < noise_prob_low_pass:
                # Define the desired cutoff frequency (in Hz)
                cutoff = 1000.0
                # Apply low-pass filter to the WAV data
                noise = lowpass_filter(noise, cutoff, self.source_sample_rate)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio

    def prepare_inputs(self, batch: dict):
        """
        Encode source audio, optionally prepend language-direction prompt embeddings into
        source_encoded (DuplexSTT-style), then build text inputs/labels for text-only loss.
        Audio codec and speech generation are not used.
        """
        assert batch["source_audio"].size(0) == batch["target_tokens"].size(0)

        if self.cfg.get('use_old_noise_aug', None):
            # ToDo we are applying it in all datasets, old codebase does not applied in real conv data
            noise_prob = 0.99
            noise_min_snr = 20
            noise_max_snr = 50
            noise_path = self.cfg.get(
                'old_noise_aug_path',
                None
            )
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
            # change audio volume randomly
            if self.training and random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                min_scale_val = self.cfg.get('noise_scale_user_min', 0.0631)  # -15 snr
                max_scale_val = self.cfg.get('noise_scale_user_min', 5.6234)  # 24 snr

                # get a random float value between min and max
                scaling_factor = (
                    torch.rand(batch["source_audio"].size(0), device=batch["source_audio"].device)
                    * (max_scale_val - min_scale_val)
                    + min_scale_val
                )
                batch["source_audio"] = batch["source_audio"] * scaling_factor.unsqueeze(-1)

            # apply low pass filter
            if self.training and random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                cutoff_freq = self.cfg.get('noise_low_pass_cutoff_freq', 1000.0)
                # note here we are using a biquad filter, older codebase we are using a filter of order 5

                import torchaudio as _ta
                batch["source_audio"] = _ta.functional.lowpass_biquad(
                    waveform=batch["source_audio"], sample_rate=self.source_sample_rate, cutoff_freq=cutoff_freq
                )
        
        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )
        # asr_emb (fed to the auxiliary RNNT ASR head, see training_step) is captured BEFORE
        # language-prompt embeddings are prepended to source_encoded below: the RNNT objective
        # transcribes raw source-language acoustic frames only and must not see prompt frames.
        # asr_emb_lens is cloned from the pre-prepend source_encoded_lens for the same reason.
        asr_emb_lens = source_encoded_lens.clone()

        # Insert language-direction prompt embeddings at the front of source_encoded.
        # Prompt tokens from the dataset are embedded and prepended; actual audio features
        # shift right.  target_tokens receives matching pad frames at the front so the
        # sequence stays aligned — no extra loss masking is needed.
        source_encoded, source_encoded_lens, target_tokens = maybe_prepend_prompt_tokens(
            batch=batch,
            embed_fn=self.embed_tokens,
            source_encoded=source_encoded,
            source_encoded_lens=source_encoded_lens,
            text_pad_id=self.text_pad_id,
        )

        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (target_tokens.shape[1] - 1) % tp_world_size) != 0:
                target_tokens = target_tokens[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = target_tokens[:, :-1]   # (B, T-1): shifted input
        text_labels = target_tokens[:, 1:]    # (B, T-1): shifted labels

        input_embeds = self.embed_tokens(text_inputs)
        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

        # Sequence mask: zero-out frames beyond each example's (prompt+content) length.
        # batch["target_token_lens"] was updated in-place by maybe_prepend_prompt_tokens.
        seq_mask = torch.ones(
            text_labels.shape[0], text_labels.shape[1], 1,
            device=self.device, dtype=torch.bool,
        )
        if self.cfg.get("mask_sequence_loss", True):
            for i in range(batch["target_token_lens"].size(0)):
                seq_mask[i, batch["target_token_lens"][i]:, :] = 0
            mask_lengths = seq_mask[:, :, 0].sum(-1)
            assert torch.allclose(batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0)

        loss_scale = seq_mask.clone().float()
        if self.cfg.get("scale_loss_by") == 'non_sil_t':
            loss_scale[:, :, :1] = torch.where(
                text_labels.unsqueeze(-1) != self.text_pad_id,
                self.cfg.get("scale_loss_mask", self.cfg.get("nonsil_weight", 4.0)),
                loss_scale[:, :, :1],
            )

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "text_labels": text_labels,
            "seq_mask": seq_mask,
            "loss_scale": loss_scale,
            "asr_emb": asr_emb,
            "asr_emb_lens": asr_emb_lens,
        }


    def _rnnt_joint_and_loss(
        self,
        encoder_outputs,
        decoder_outputs,
        targets,
        encoder_lengths,
        target_lengths,
        use_ckpt: bool = False,
    ):
        """Compute one rnnt_joint + rnnt_loss pair, with optional activation checkpointing.

        The joint produces a (B, T, U+1, V+1) tensor that RNNTLoss internally upcasts to fp32
        and whose gradient buffer the numba kernel also allocates at the same shape -- these
        can be large for multilingual BPE vocabularies. When ``use_ckpt=True``
        (``model.rnnt_loss_use_checkpoint``), the body runs under
        ``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` so those tensors are not kept
        alive by the autograd graph across the rest of ``training_step``.
        """
        if use_ckpt:
            return torch.utils.checkpoint.checkpoint(
                self._rnnt_joint_and_loss_inner,
                encoder_outputs,
                decoder_outputs,
                targets,
                encoder_lengths,
                target_lengths,
                use_reentrant=False,
            )
        return self._rnnt_joint_and_loss_inner(
            encoder_outputs,
            decoder_outputs,
            targets,
            encoder_lengths,
            target_lengths,
        )

    def _rnnt_joint_and_loss_inner(
        self,
        encoder_outputs,
        decoder_outputs,
        targets,
        encoder_lengths,
        target_lengths,
    ):
        """Inner body of the RNNT joint+loss path; kept separate so it can be passed
        to `torch.utils.checkpoint.checkpoint` as a plain callable."""
        joint_out = self.rnnt_joint(encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs)
        return self.rnnt_loss(
            log_probs=joint_out,
            targets=targets,
            input_lengths=encoder_lengths,
            target_lengths=target_lengths,
        )

    def _compute_rnnt_loss(self, batch: dict, inputs: dict):
        """Auxiliary RNNT-ASR loss on source-language transcripts (`batch["source_texts"]`),
        fed from `inputs["asr_emb"]` (the perception module's ASR-adapter output -- a path
        separate from the AST `modality_adapter`/LLM path). Returns ``None`` when there is
        nothing to train on in this batch (e.g. all transcripts empty)."""
        source_texts = batch.get("source_texts", None)
        if source_texts is None:
            raise KeyError(
                "model.use_rnnt_loss=true but batch is missing 'source_texts'; the data pipeline "
                "must provide per-cut source-language transcriptions (e.g. s2s_dataset_concat_v.py)."
            )
        src_for_rnnt = [(t or "").strip() for t in source_texts]
        # BPE-encode each source utterance through the RNNT's own tokenizer. Empty transcripts
        # are excluded -- RNNTLoss requires U >= 1.
        ids_lists = [self.rnnt_tokenizer.text_to_ids(s) for s in src_for_rnnt]
        keep_idx = [i for i, ids in enumerate(ids_lists) if len(ids) > 0]
        if len(keep_idx) == 0:
            return None

        kept_ids = [ids_lists[i] for i in keep_idx]
        target_lengths = torch.tensor([len(ids) for ids in kept_ids], device=self.device, dtype=torch.long)
        max_u = int(target_lengths.max().item())
        targets = torch.zeros((len(kept_ids), max_u), device=self.device, dtype=torch.long)
        for i, ids in enumerate(kept_ids):
            targets[i, : len(ids)] = torch.tensor(ids, device=self.device, dtype=torch.long)
        keep_t = torch.tensor(keep_idx, device=self.device, dtype=torch.long)

        # inputs["asr_emb"] is (B, T, D); rnnt_joint expects (B, D, T). Pick the non-empty
        # subset and truncate the time dimension to this sub-batch's actual max length before
        # the joint (memory optimization: keeps the (B, T, U+1, V+1) joint tensor as small as
        # the data allows instead of the full padded batch width).
        sub_emb = inputs["asr_emb"].index_select(0, keep_t)
        sub_lens = inputs["asr_emb_lens"].index_select(0, keep_t).clamp(min=1, max=sub_emb.shape[1])
        max_enc_len = int(sub_lens.max().item())
        if max_enc_len < sub_emb.shape[1]:
            sub_emb = sub_emb[:, :max_enc_len, :].contiguous()
        encoder_outputs = sub_emb.transpose(1, 2).contiguous()
        encoder_lengths = sub_lens

        use_ckpt = bool(self.cfg.get("rnnt_loss_use_checkpoint", False))
        chunk_size = int(self.cfg.get("rnnt_loss_chunk_size", 0) or 0)
        n_kept = encoder_outputs.shape[0]
        if chunk_size <= 0 or chunk_size >= n_kept:
            decoder_outputs, _, _ = self.rnnt_decoder(targets=targets, target_length=target_lengths)
            return self._rnnt_joint_and_loss(
                encoder_outputs=encoder_outputs,
                decoder_outputs=decoder_outputs,
                targets=targets,
                encoder_lengths=encoder_lengths,
                target_lengths=target_lengths,
                use_ckpt=use_ckpt,
            )

        # Chunked path (large batches / long vocab): process the keep-set in chunks and average
        # per-sample losses so the weight stays comparable to the un-chunked case (RNNTLoss
        # reduction='mean_batch'). Each chunk is re-truncated to its OWN max lengths (required
        # for RNNTLoss.certify_inputs correctness, and helpful for memory).
        chunk_losses = []
        chunk_sample_counts = []
        for start in range(0, n_kept, chunk_size):
            end = min(start + chunk_size, n_kept)
            c_tgt = targets[start:end]
            c_tgt_lens = target_lengths[start:end]
            c_enc = encoder_outputs[start:end]
            c_enc_lens = encoder_lengths[start:end]
            c_max_u = int(c_tgt_lens.max().item())
            if c_max_u < c_tgt.shape[1]:
                c_tgt = c_tgt[:, :c_max_u].contiguous()
            c_max_t = int(c_enc_lens.max().item())
            if c_max_t < c_enc.shape[2]:
                c_enc = c_enc[:, :, :c_max_t].contiguous()
            c_dec, _, _ = self.rnnt_decoder(targets=c_tgt, target_length=c_tgt_lens)
            c_loss = self._rnnt_joint_and_loss(
                encoder_outputs=c_enc,
                decoder_outputs=c_dec,
                targets=c_tgt,
                encoder_lengths=c_enc_lens,
                target_lengths=c_tgt_lens,
                use_ckpt=use_ckpt,
            )
            chunk_losses.append(c_loss * (end - start))
            chunk_sample_counts.append(end - start)
        return torch.stack(chunk_losses).sum() / float(sum(chunk_sample_counts))

    def training_step(self, batch: dict, batch_idx: int):
        modules_to_check = [self.perception.preprocessor, self.perception.encoder, self.llm]
        if self.rnnt_decoder is not None:
            modules_to_check.append(self.rnnt_decoder)
        if self.rnnt_joint is not None:
            modules_to_check.append(self.rnnt_joint)
        for m in modules_to_check:
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], seq_mask=inputs["seq_mask"])

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

        loss = self.cfg.text_loss_weight * text_loss

        # ---- Optional auxiliary RNNT-ASR loss --------------------------------------------
        # Additive: total_loss = text_loss_weight * text_loss + rnnt_loss_weight * rnnt_loss.
        # Gated on `use_rnnt_loss` (default False): when unset this block is a complete no-op,
        # so existing translation-only recipes are unaffected. Preserves the existing
        # translation/text loss and language-prompting behavior -- this does not replace them.
        rnnt_loss_val = None
        if self.cfg.get("use_rnnt_loss", False) and self.rnnt_loss is not None:
            rnnt_loss_val = self._compute_rnnt_loss(batch, inputs)
            if rnnt_loss_val is not None:
                loss = loss + self.cfg.get("rnnt_loss_weight", 1.0) * rnnt_loss_val
        # -----------------------------------------------------------------------------------

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "num_frames": num_frames.to(torch.float32),
            "padding_ratio": num_frames / (B * T),
        }
        if rnnt_loss_val is not None:
            ans["rnnt_loss"] = rnnt_loss_val

        self.log("batch_size", B, on_step=True, prog_bar=True, logger=True)
        self.log("sequence_length", T, on_step=True, prog_bar=True, logger=True)

        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        pass

    def _get_rnnt_decoding(self):
        """Lazy-build and cache NeMo RNNT greedy decoding for validation-time transcription
        of the source-language audio (asr_adapter -> rnnt_decoder -> rnnt_joint). Purely an
        eval-time utility: does not affect training_step, the RNNT loss, or any trainable
        parameter. Returns ``None`` when the RNNT branch isn't configured
        (``model.use_rnnt_loss``/``pretrained_rnnt_asr`` unset), matching training's own gating.
        """
        if getattr(self, "_rnnt_decoding", None) is not None:
            return self._rnnt_decoding
        if getattr(self, "rnnt_decoder", None) is None or getattr(self, "rnnt_joint", None) is None:
            return None
        (
            RNNTDecoding,
            RNNTDecodingConfig,
            RNNTBPEDecoding,
            RNNTBPEDecodingConfig,
        ) = _get_rnnt_decoding_module()
        decoding_cfg = OmegaConf.structured(RNNTBPEDecodingConfig())
        decoding_cfg.strategy = "greedy"
        decoding_cfg.greedy.max_symbols_per_step = self.cfg.get("rnnt_max_symbols_per_step", 10)
        tokenizer = getattr(self, "rnnt_tokenizer", None)
        if tokenizer is not None:
            self._rnnt_decoding = RNNTBPEDecoding(
                decoding_cfg=decoding_cfg,
                decoder=self.rnnt_decoder,
                joint=self.rnnt_joint,
                tokenizer=tokenizer,
            )
            logging.info("Using RNNTBPEDecoding with ASR tokenizer (ids_to_text) for validation-time RNNT WER.")
        else:
            vocab = getattr(self.rnnt_joint, "vocabulary", None)
            if vocab is None:
                logging.warning("rnnt_joint has no vocabulary; cannot build RNNT decoding for validation.")
                return None
            decoding_cfg_v = OmegaConf.structured(RNNTDecodingConfig())
            decoding_cfg_v.strategy = "greedy"
            decoding_cfg_v.greedy.max_symbols_per_step = self.cfg.get("rnnt_max_symbols_per_step", 10)
            self._rnnt_decoding = RNNTDecoding(
                decoding_cfg=decoding_cfg_v,
                decoder=self.rnnt_decoder,
                joint=self.rnnt_joint,
                vocabulary=list(vocab),
            )
            logging.info("Using RNNTDecoding with vocabulary (no tokenizer) for validation-time RNNT WER.")
        return self._rnnt_decoding

    @staticmethod
    def _rnnt_hypotheses_to_src_text(hypotheses: list) -> list:
        """Convert RNNT Hypothesis objects to transcript strings. Normalizes the SentencePiece
        ``▁`` (U+2581) word-boundary marker to a space; BCP-47 language tags the multilingual
        RNNT vocabulary may emit (e.g. ``<fr-FR>``) are intentionally kept for observability --
        WER scoring strips them via the metric's own text normalizer."""
        import re

        texts = []
        for hyp in hypotheses:
            raw = str(getattr(hyp, "text", "") or "").strip()
            raw = raw.replace("▁", " ")
            raw = re.sub(r" +", " ", raw).strip()
            raw = re.sub(r"(\s+)([.,?])", r"\2", raw)
            texts.append(raw)
        return texts

    def _decode_rnnt_offline(self, asr_emb: torch.Tensor, asr_emb_lens: torch.Tensor):
        """Full-utterance (non-streaming) greedy RNNT decode of ``asr_emb`` -> list[str], one
        transcript per batch item. Returns ``None`` when the RNNT branch isn't configured."""
        decoding = self._get_rnnt_decoding()
        if decoding is None:
            return None
        encoder_output = asr_emb.transpose(1, 2).contiguous()  # (B, D, T)
        with torch.inference_mode():
            hypotheses = decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoder_output,
                encoded_lengths=asr_emb_lens,
                return_hypotheses=True,
            )
        return self._rnnt_hypotheses_to_src_text(hypotheses)

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        # Val sees full data on every DDP rank (DataModule); only rank 0 should write shared
        # metadata/WAVs to avoid duplicate rows and parallel truncation on shared filesystems.
        if self.trainer.is_global_zero:
            self.results_logger = ResultsLogger(self.validation_save_path).reset()
        else:
            self.results_logger = None
        self.bleu = BLEU().reset()
        tolerance = int(
            self.cfg.get("val_acc_tolerance", 160) / (1000 / self.source_fps)
        )  # 160 ms as default tolerance
        self.text_bos_acc = TokenAccuracy(
            token_name="text_bos", token_id=self.text_bos_id, tolerance=tolerance
        ).reset()
        self.text_eos_acc = TokenAccuracy(
            token_name="text_eos", token_id=self.text_eos_id, tolerance=tolerance
        ).reset()
        if self.rnnt_decoder is not None:
            self.src_wer = WER(verbose=True).reset()
            self.src_bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_bos_acc = self.text_bos_acc.compute()
        for k, m in text_bos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_eos_acc = self.text_eos_acc.compute()
        for k, m in text_eos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        if self.rnnt_decoder is not None:
            src_wer = self.src_wer.compute()
            for k, m in src_wer.items():
                self.log(f"{prefix}_src_rnnt_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
            src_bleu = self.src_bleu.compute()
            for k, m in src_bleu.items():
                self.log(f"{prefix}_src_rnnt_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        if self.results_logger is not None:
            self.results_logger.compute_and_save()

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_inference(dataset_batch)
            src_hyps_rnnt = results.get("src_text_rnnt")

            if self.results_logger is not None:
                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=None,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=None,
                    pred_audio_sr=self.source_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                    src_refs=dataset_batch.get("source_texts") if src_hyps_rnnt is not None else None,
                    src_hyps=src_hyps_rnnt,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            self.text_bos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])
            self.text_eos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])

            if src_hyps_rnnt is not None:
                self.src_wer.update(name=name, refs=dataset_batch["source_texts"], hyps=src_hyps_rnnt)
                self.src_bleu.update(name=name, refs=dataset_batch["source_texts"], hyps=src_hyps_rnnt)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """Return embed(pad_id) — the 'nothing before sequence' marker added to frame 0."""
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        return self.embed_tokens(text_bos)  # (1, H)

    @torch.no_grad()
    def offline_inference(
        self,
        dataset_batch: dict,
        speaker_encoder_emb=None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive text-only prediction (speech-to-text translation).

        Mirrors DuplexSTT offline_inference: prompt embeddings are inserted into
        source_encoded at the feature level, the AR loop runs over all frames
        (prompt + audio), prompt positions are pre-filled with pad and not
        overwritten, and prompt frames are stripped from the output.

        Returns:
            * "text":        decoded text strings, list of length B.
            * "tokens_text": generated token ids (B, T_out).
            * "tokens_len":  output lengths (B,).
        """
        input_signal = dataset_batch["source_audio"]
        input_signal_lens = dataset_batch["source_audio_lens"]
        if self.cfg.get("custom_sample_inference", None):
            import torchaudio as _ta
            from nemo.collections.audio.parts.utils.transforms import resample as _resample
            device = input_signal.device
            input_signal, sr = _ta.load(self.cfg.custom_sample_inference)
            input_signal = input_signal.to(device)[:1, :]
            input_signal = _resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
        # asr_emb_lens is captured BEFORE the prompt-prefix loop below mutates `lengths` in
        # place (same reasoning as prepare_inputs/training_step): the RNNT branch transcribes
        # raw source-language acoustic frames only and must not see prompt frames.
        asr_emb_lens = lengths.clone()
        B, T_local, H = source_encoded.shape

        # ── Insert prompt embeddings (DuplexSTT-style) ──────────────────────────
        # Matches training: maybe_prepend_prompt_tokens inserts prompt embeddings
        # at the front of source_encoded and shifts audio features to the right.
        prompt_tokens = dataset_batch.get("prompt_tokens", None)
        prompt_token_lens = dataset_batch.get("prompt_token_lens", None)
        max_P = 0
        if prompt_tokens is not None and prompt_token_lens is not None:
            if not isinstance(prompt_token_lens, torch.Tensor):
                prompt_token_lens = torch.tensor(prompt_token_lens, device=self.device)
            else:
                prompt_token_lens = prompt_token_lens.to(self.device)
            max_P = int(prompt_token_lens.max().item())
            if max_P > 0:
                prompt_embedded = self.embed_tokens(prompt_tokens.to(self.device))  # (B, max_P, H)
                new_source = torch.zeros(
                    B, max_P + T_local, H, dtype=source_encoded.dtype, device=source_encoded.device
                )
                for i, pl in enumerate(prompt_token_lens):
                    pl = int(pl.item())
                    if pl > 0:
                        new_source[i, :pl] = prompt_embedded[i, :pl]
                    src_len = int(lengths[i].item())
                    new_source[i, pl : pl + src_len] = source_encoded[i, :src_len]
                    lengths[i] = pl + src_len
                source_encoded = new_source
                T_local = source_encoded.shape[1]

        # ── Determine decoding horizon T ────────────────────────────────────────
        T_tensor = torch.tensor([T_local], device=source_encoded.device)
        if self._use_fsdp:
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
        T = int(self.cfg.get("inference_tgt_len", 1.5 * T_tensor.item()))

        if T > T_local:
            last_frame = source_encoded[:, T_local - 1 : T_local, :]
            source_encoded = torch.cat([source_encoded, last_frame.repeat(1, T - T_local, 1)], dim=1)

        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # gen_text pre-filled with pad; prompt positions will stay as pad
        gen_text = torch.full((B, T), self.text_pad_id, device=self.device, dtype=torch.long)

        # Mark which frames are prompt frames so we skip overwriting them
        is_prompt = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        if max_P > 0:
            for i, pl in enumerate(prompt_token_lens):
                pl = int(pl.item())
                if pl > 0:
                    is_prompt[i, :pl] = True

        # ── Frame 0: add pad embedding (before-sequence marker) ─────────────────
        cache = DynamicCache()
        input_embeds[:, 0] += self._get_bos_embedding()
        ans = self(input_embeds[:, :1], cache=cache, seq_mask=None)
        if not is_prompt[:, 0].all():
            gen = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_text[:, 0] = torch.where(is_prompt[:, 0], gen_text[:, 0], gen)

        # ── AR loop ──────────────────────────────────────────────────────────────
        gen_text_len = torch.full((B,), T, device=self.device, dtype=input_signal_lens.dtype)
        txt_done = torch.zeros(B, dtype=torch.bool, device=self.device)

        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            ans = self(input_embeds[:, t : t + 1], cache=ans["cache"], seq_mask=None)

            is_prompt_t = is_prompt[:, t]
            if not is_prompt_t.all():
                gen = ans["text_logits"][:, -1].argmax(dim=-1)
                gen_text[:, t] = torch.where(is_prompt_t, gen_text[:, t], gen)

            text_done_t = (gen_text[:, t] == self.text_eos_id) & ~is_prompt_t
            newly_done = (~txt_done) & text_done_t
            gen_text_len[newly_done] = t + 1
            txt_done |= newly_done
            if txt_done.all():
                break

        # ── Trim FSDP padding ────────────────────────────────────────────────────
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_text_len = gen_text_len.clamp(max=T_local)

        # ── Strip prompt frames from output (DuplexSTT _post_inference style) ───
        if max_P > 0:
            current_T = gen_text.shape[1]
            gen_text_out = torch.zeros(B, current_T - max_P, dtype=torch.long, device=self.device)
            gen_text_len_out = torch.zeros(B, dtype=gen_text_len.dtype, device=self.device)
            for i, pl in enumerate(prompt_token_lens):
                pl_val = int(pl.item())
                raw_len = int(gen_text_len[i].item())
                actual_len = max(raw_len - pl_val, 0)
                if actual_len > 0:
                    gen_text_out[i, :actual_len] = gen_text[i, pl_val : pl_val + actual_len]
                gen_text_len_out[i] = actual_len
            gen_text = gen_text_out
            gen_text_len = gen_text_len_out

        result = {
            "text": tokens_to_str(gen_text, gen_text_len, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_len": dataset_batch["decode_source_audio_lens"],
            # Validation-time-only RNNT greedy decode (asr_adapter -> rnnt_decoder -> rnnt_joint)
            # of the source-language audio. None when the RNNT branch isn't configured. This does
            # not affect the AST/translation prediction above or any training behavior.
            "src_text_rnnt": self._decode_rnnt_offline(asr_emb, asr_emb_lens),
        }

        if self.cfg.get("custom_sample_inference", None):
            exit()
        return result

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        result = configure_optimizers(self)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        logging.info(
            "Parameter budget — trainable: %s (%.1f M)  frozen: %s (%.1f M)",
            f"{n_trainable:,}",
            n_trainable / 1e6,
            f"{n_frozen:,}",
            n_frozen / 1e6,
        )
        return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Snapshot frozen-parameter fingerprints for `on_save_checkpoint`'s integrity check,
        once (after the first optimizer step of this run/resume). Not done in
        `configure_optimizers()`: the strategy (DDP/FSDP) may still perform one-time
        parameter/buffer broadcasts or device/precision materialization after that point,
        which would make an earlier snapshot look like spurious "drift" at the first save.
        """
        if getattr(self, "_frozen_param_fingerprints", None) is None and self.trainer is not None:
            self._frozen_param_fingerprints = snapshot_frozen_param_fingerprints(self)
            logging.info(
                f"Frozen-parameter integrity snapshot recorded for {len(self._frozen_param_fingerprints)} "
                f"parameter tensor(s) after step {self.trainer.global_step}; will be re-checked before "
                "every checkpoint save."
            )

    def on_save_checkpoint(self, checkpoint):
        """
        Integrity check: every parameter marked frozen (requires_grad=False) by
        `configure_optimizers` must still be bit-for-bit identical to its value at the start
        of this run/resume. Runs on every checkpoint save. Raises loudly rather than silently
        persisting a checkpoint with corrupted "frozen" weights (see model.freeze_params).
        """
        fingerprints = getattr(self, "_frozen_param_fingerprints", None)
        if not fingerprints:
            return  # configure_optimizers()/first training step haven't run yet.
        drifted = verify_frozen_params_unchanged(self, fingerprints)
        if drifted:
            msg = (
                f"Frozen-parameter integrity check FAILED before checkpoint save at "
                f"global_step={self.trainer.global_step if self.trainer is not None else '?'}: "
                f"{len(drifted)} parameter(s) that should be frozen (per model.freeze_params) "
                f"have changed value since training started. Refusing to save -- this checkpoint's "
                f"frozen submodules (e.g. llm, perception.encoder) can no longer be trusted to "
                f"match their intended pretrained/frozen state."
            )
            if os.environ.get("FROZEN_PARAM_CHECK_NONFATAL", "0") == "1":
                # DIAGNOSTIC-ONLY escape hatch: log the FULL drifted list and continue instead
                # of raising, so a live repro run can keep going past the first checkpoint save.
                logging.error(msg)
                logging.error(f"[frozen-param-check] FULL drifted list ({len(drifted)}): {drifted}")
            else:
                raise RuntimeError(msg)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
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
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
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
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
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

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)


# Backward-compatibility alias so existing configs that reference
# DuplexS2SSpeechDecoderModel2 still work.
DuplexS2SSpeechDecoderModel2 = NemotronVoiceTranslateSTT