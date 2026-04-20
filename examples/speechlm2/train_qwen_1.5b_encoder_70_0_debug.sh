#!/bin/bash
#SBATCH -J "s2s_duplex_hf_st"
#SBATCH --partition=polar,polar3,polar4
#SBATCH -N 4  # number of nodes
#SBATCH --cpus-per-task=8
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0

set -x

SEED=42
# SEED=$((SLURM_JOB_ID % 2147483647))

CODE_DIR=/export/fs06/ahussei6/nvidia/github/NeMo

CONFIG_PATH=conf/train
CONFIG_NAME="qwen_1b"

EXP_NAME="${CONFIG_NAME}" 
RESULTS_DIR="/export/fs06/ahussei6/nvidia/ara_duplex_test/${EXP_NAME}"
mkdir -p ${RESULTS_DIR}

PROJECT_NAME="duplex_s2s_st" 

pretrained_asr="/export/fs06/ahussei6/nvidia/pretrained_models/asr/stt_en_fastconformer_hybrid_large_streaming_multi_v1.20.0/stt_en_fastconformer_hybrid_large_streaming_multi.nemo" # replace with your pretrained ASR model path
pretrained_tts="/export/fs06/ahussei6/nvidia/pretrained_models/magpie_tts/tts-pretraining_qwnen_2.5_81007_steps.ckpt" # replace with your pretrained TTS model path
pretrained_codec="/export/fs06/ahussei6/nvidia/pretrained_models/nano_codec/Low_Frame-rate_Speech_Codec++.nemo" # replace with your pretrained codec model path

conda activate nemo 
echo "Using Python at: $(which python)"
echo "Config path: $CONFIG_PATH"
echo "Config name: $CONFIG_NAME"

chmod -R 777 "${RESULTS_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
export HF_HOME="/export/fs06/ahussei6/nvidia/cache/HFCACHE"
export TORCH_HOME="/export/fs06/ahussei6/nvidia/cache/HFCACHE"
export NEMO_CACHE_DIR="/export/fs06/ahussei6/nvidia/cache/HFCACHE"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3

HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
python ${CODE_DIR}/examples/speechlm2/s2s_duplex_speech_decoder_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    ++exp_manager.checkpoint_callback_params.save_top_k=3 \
    exp_manager.name=${EXP_NAME} \
    ++model.pretrained_audio_codec="${pretrained_codec}" \
    ++model.pretrained_tts_from_s2s="${pretrained_tts}" \
    ++model.pretrained_asr="${pretrained_asr}" \
    ++model.mask_sequence_loss=True \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES:-1} \
    exp_manager.explicit_log_dir=${RESULTS_DIR} \
    data.train_ds.seed=$SEED \
    ++model.audio_loss_weight=20 \
    ++model.speech_decoder.cond_on_prev_audio_tokens=True \
    ++model.speech_decoder.use_speaker_encoder=True \
    ++model.speech_decoder.cond_on_char_embedding=True \
    ++model.speech_decoder.cond_on_asr_emb=False \
    ++model.speech_decoder.cond_on_llm_latent=False \
    ++model.speech_decoder.cond_on_modality_adapter_emb=False \
    ++model.speech_decoder.cond_on_text_tokens=False \
    ++model.speech_decoder.cfg_scale=2.5 \
    ++model.speech_decoder.kernel_size=3 \
    ++model.speech_decoder.cfg_unconditional_prob=0.2 \
    ++model.custom_codebook_size=2045 \
    ++model.custom_speech_bos_id=2019 \
    ++model.custom_speech_eos_id=2020 \
    ++model.custom_speech_delay_id=2018 \
    model.perception.encoder.att_context_size=[70,0] \
    model.perception.modality_adapter.att_context_size=[70,0] \
    ++model.pretrained_llm="/export/fs06/ahussei6/nvidia/cache/HFCACHE/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" \
    ++trainer.limit_val_batches=1 \
    ++trainer.val_check_interval=1000 \
    ++model.scale_loss_by="non_sil_t" \
    ++model.scale_loss_mask=10 \
    ++model.val_acc_tolerance=480 \
    data.validation_ds.seed=$SEED 
    # ++model.use_old_noise_aug=True 
    # ++model.old_noise_aug_path='/lustre/fsw/portfolios/edgeai/users/amhussein/data/dns5_demand_noise_night'




