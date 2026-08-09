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
Evaluation script for NemotronVoiceTranslateSync.

Identical to nemotron_voicetranslate_eval.py except it instantiates
NemotronVoiceTranslateSync, which prepends source-duration silence to the
generated audio so the stereo WAV plays as:

  Channel 1 (source)     : source speech  →  silence
  Channel 2 (translation): silence        →  translated speech

This simulates a streaming S2S interpreter: you hear source language first,
then translation begins right after the source utterance ends.

See nemotron_voicetranslate_sync.py and conf/nemotron_voicetranslate_sync.yaml
for details.
"""

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexS2SDatasetConcatV
from nemo.collections.speechlm2.models.nemotron_voicetranslate_sync import NemotronVoiceTranslateSync
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="nemotron_voicetranslate_sync")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    with trainer.init_module():
        model = NemotronVoiceTranslateSync(OmegaConf.to_container(cfg, resolve=True))

    # Load full model checkpoint (LLM + TTS weights) when provided.
    if cfg.get("checkpoint_path", None):
        model.init_from_model_from_ckpt(cfg.checkpoint_path)

    # Optionally load a standalone DuplexEARTTS checkpoint for the TTS sub-module.
    if cfg.get("pretrained_tts", None):
        model.init_tts_model_from_checkpoint(cfg.pretrained_tts)

    model.eval()

    # Merge any inference-time config overrides back into the model.
    model.full_cfg.merge_with(cfg)
    model.cfg.merge_with(cfg.model)
    OmegaConf.save(model.full_cfg, log_dir / "exp_config.yaml")
    model.validation_save_path = os.path.join(log_dir, "validation_logs")

    # Build the translation dataset (returns source_audio, target_texts, etc.)
    dataset = DuplexS2SDatasetConcatV(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        training_speaker_reference=cfg.data.get("training_speaker_reference", None),
        training_speaker_duration=cfg.data.get("training_speaker_duration", 3.0),
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    inference()
