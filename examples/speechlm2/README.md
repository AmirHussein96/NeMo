 # English Duplex Speech-to-Speech Training Recipe

## Clone the repositories

```bash
git clone https://github.com/AmirHussein96/NeMo.git
cd NeMo
git checkout ara_duplex
pip install -e '.[all]'
```

```bash
git clone https://github.com/lhotse-speech/lhotse
cd lhotse
pip install -e '.[dev]'
```

### Pretrained model locations

- English streaming ASR: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_multi?version=1.20.0
- TTS and Nano Codec: https://huggingface.co/AmirHussein/nemo_models/tree/main
- Qwen: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct


### Required configuration changes

Before running the recipe, update the following paths:

1. In `NeMo/examples/speechlm2/train_qwen_1.5b_encoder_70_0.sh`, set the paths for:
    - pretrained ASR model
    - pretrained TTS model
    - pretrained codec model
    - HF_CACHE
    - CODE_DIR
    - RESULTS_DIR
2. In `NeMo/examples/speechlm2/conf/data/en.yaml`, set the path to the Lhotse Shar training dataset.
3. In `NeMo/examples/speechlm2/conf/train/qwen_1b.yaml`, set the path to the validation Lhotse Shar dataset.
4. To generate `bucket_duration_bins` for `NeMo/examples/speechlm2/conf/train/qwen_1b.yaml` run:
```bash
python NeMo/scripts/speech_recognition/estimate_buckets.py data/en.yaml --buckets 30 --min_duration 30 --max_duration 120
``` 


### Launch the English Duplex experiment:

Submit the following Slurm job:

```bash
sbatch NeMo/examples/speechlm2/train_qwen_1.5b_encoder_70_0.sh
```