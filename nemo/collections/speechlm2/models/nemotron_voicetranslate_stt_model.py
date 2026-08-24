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
from lightning import LightningModule
from omegaconf import DictConfig
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
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


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
        
        source_encoded, source_encoded_lens, _ = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

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
        }


    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
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

        self.log("batch_size", B, on_step=True, prog_bar=True, logger=True)
        self.log("sequence_length", T, on_step=True, prog_bar=True, logger=True)

        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
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
        self.results_logger.compute_and_save()

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_inference(dataset_batch)

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

        source_encoded, lengths, _ = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
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
        }

        if self.cfg.get("custom_sample_inference", None):
            exit()
        return result

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

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