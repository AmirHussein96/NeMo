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

### Convert raw data to Lhotse Shar format

This section provides a small toy example showing how to convert raw data from the NeMo manifest format into Lhotse Shar format.

1. Download the raw `{src|tgt}` data from `data_sample`:
   `https://huggingface.co/AmirHussein/nemo_models/tree/main/data_sample` 
2. Convert the raw data from the NVIDIA manifest format into Lhotse Shars:

   ```bash
    src_manifest=src/sharded_manifests/manifest_0.jsonl
    src_tar=src/audio_0.tar
    tgt_manifest=tgt/sharded_manifests/manifest_0.jsonl
    tgt_tar=tgt/audio_0.tar

    python NeMo/nemo/collections/speechlm2/data/combine_info_concat_v.py \
        --src_tar $src_tar \
        --tgt_tar $tgt_tar \
        --src_manifest $src_manifest \
        --tgt_manifest $tgt_manifest \
        --output_dir output_shar_dir
    ```
3. Compare the generated shars with `https://huggingface.co/AmirHussein/nemo_models/tree/main/data_sample/debug`
4. Set the train data path in `NeMo/examples/speechlm2/conf/data/en.yaml` to point to `outpur_shar_dir`
5. Repeat the same process for the validation subset, then set `validation_ds` in  `NeMo/examples/speechlm2/conf/train/qwen_1b.yaml` to point to `validation_shar_dir`. 


### Launch the English Duplex experiment:

Submit the following Slurm job:

```bash
sbatch NeMo/examples/speechlm2/train_qwen_1.5b_encoder_70_0.sh
```