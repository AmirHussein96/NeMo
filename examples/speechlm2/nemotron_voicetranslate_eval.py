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
Evaluation script for NemotronVoiceTranslate models.

This script runs validation for a NemotronVoiceTranslate checkpoint using a
Duplex S2S translation-style Lhotse dataset.  It evaluates the full
speech-to-speech translation pipeline:
  1. Speech encoder (FastConformer) encodes the source speech.
  2. Riva-Translate LLM backbone generates target text tokens autoregressively.
  3. DuplexEARTTS synthesises the translated speech from those tokens,
     conditioned on a speaker reference audio.

Metrics
-------
During validation the script computes:
  - Text BLEU score  (reference target text vs. predicted text)
  - ASR-BLEU score   (reference target text vs. ASR transcription of
                      generated speech)

The ASR model used for scoring is defined by:
    model.scoring_asr

Arguments
---------
For a complete configuration reference see:
    examples/speechlm2/conf/nemotron_voicetranslate.yaml

cfg : omegaconf.DictConfig
    Top-level Hydra configuration object.  Expected top-level sections:

    checkpoint_path (str | null)
        Path to the full NemotronVoiceTranslate checkpoint (.ckpt).
        If null the model runs with randomly-initialised weights (useful
        only for architecture debugging).

    pretrained_tts (str | null)
        Optional path to a standalone DuplexEARTTS checkpoint.  When set,
        only the TTS sub-module weights are loaded from this file (instead
        of — or in addition to — checkpoint_path).  Ignored if null.

    model (DictConfig)
        Model settings.  Key parameters:
        * pretrained_llm (str): HuggingFace model-id for the LLM backbone.
        * pretrained_weights (bool): Whether to load pretrained LLM weights.
        * scoring_asr (str): ASR model id for ASRBLEU evaluation.
        * inference_speaker_reference (str | null): Path to speaker reference
          audio.  Set to null to use a named preset.
        * inference_speaker_name (str | null): Named speaker preset; overrides
          inference_speaker_reference.
        * speech_generation (DictConfig): Full DuplexEARTTS config sub-tree
          (codec_config, tts_config, inference params, …).

    data (DictConfig)
        Data pipeline configuration.  Key parameters:
        * source_sample_rate (int): Sample rate of the input source audio.
        * target_sample_rate (int): Sample rate of the generated output audio.
        * frame_length (float): Audio frame duration in seconds (e.g. 0.08).
        * input_roles (list[str]): Conversation roles mapped to input.
        * output_roles (list[str]): Conversation roles targeted for generation.
        * validation_ds (DictConfig): Lhotse shard paths and batch settings.

    exp_manager (DictConfig)
        Experiment manager settings.  Must include:
        * name (str): Experiment name.
        * explicit_log_dir (str): Root directory for output artefacts.

    trainer (DictConfig)
        PyTorch Lightning Trainer parameters (devices, num_nodes, precision,
        limit_val_batches, …).

Example Run
-----------
    python examples/speechlm2/nemotron_voicetranslate_eval.py \\
        --config-path=examples/speechlm2/conf/ \\
        --config-name=nemotron_voicetranslate \\
        exp_manager.name="NemotronVoiceTranslate_Eval" \\
        ++checkpoint_path="/path/to/checkpoint.ckpt" \\
        ++model.inference_speaker_reference=null \\
        ++model.inference_speaker_name=null \\
        ++model.speech_generation.inference_guidance_scale=0.2 \\
        ++model.speech_generation.inference_guidance_enabled=true \\
        ++model.speech_generation.inference_top_p_or_k=0.95 \\
        ++model.speech_generation.inference_noise_scale=0.001 \\
        trainer.num_nodes=1 \\
        exp_manager.explicit_log_dir="/path/to/results/" \\
        data.validation_ds.batch_size=4 \\
        "data.validation_ds.datasets.cvss_test.shar_path=/path/to/shard/" \\
        ++trainer.limit_val_batches=1.0 \\
        ++trainer.precision=32 \\
        data.validation_ds.seed=42

Outputs
-------
All generated artefacts are saved under:
    exp_manager.explicit_log_dir + "/validation_logs"

The script:
  - Saves generated audio files (.wav).
  - Saves per-utterance logs in JSON format via ResultsLogger.
  - Records predicted text, reference text, and ASR transcription of speech.

Each JSON entry has the format:
{
    "target_text": "...",
    "pred_text": "...",
    "speech_pred_transcribed": "...",
    "audio_path": "pred_wavs/example.wav"
}
"""

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexS2SDatasetConcatV
from nemo.collections.speechlm2.models.nemotron_voicetranslate import NemotronVoiceTranslate
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="nemotron_voicetranslate")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    with trainer.init_module():
        model = NemotronVoiceTranslate(OmegaConf.to_container(cfg, resolve=True))

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
