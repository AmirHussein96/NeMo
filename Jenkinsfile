pipeline {
  agent {
        docker {
      image 'gitlab-master.nvidia.com/sandeepsub/nemo_containers:nemo-180-2203-ci-apex-8cc91ceaa8faa64451d90e11b8ad4732393b32aa'
      args '--device=/dev/nvidia0 --gpus all -e TRANSFORMERS_OFFLINE=1 --user 0:128 -v /home/TestData:/home/TestData -v $HOME/.cache:/root/.cache --shm-size=8g'
        }
  }
  options {
    timeout(time: 2, unit: 'HOURS')
    disableConcurrentBuilds()
  }
  stages {

    stage('Transformers Offline') {
      steps{
        sh 'echo "TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}'
      }
    }

    stage('PyTorch version') {
      steps {
        sh 'python -c "import torch; print(torch.__version__)"'
        sh 'python -c "import torchvision; print(torchvision.__version__)"'
      }
    }

    stage('Install test requirements') {
      steps {
        sh 'apt-get update && apt-get install -y bc && pip install -r requirements/requirements_test.txt'
      }
    }

    stage('Code formatting checks') {
      steps {
        sh 'python setup.py style'
      }
    }

    stage('Copyright Headers check') {
      steps {
        sh 'python tests/check_copyright_header.py --dir .'
      }
    }

    // Removed `torch_tts` install option from NeMo>=1.7.0
    // Will add test back if/when we decide to support it again
    // stage('Torch TTS unit tests') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   steps {
    //     sh 'pip install ".[torch_tts]"'
    //     sh 'pip list'
    //     sh 'test $(pip list | grep -c lightning) -eq 0'
    //     sh 'test $(pip list | grep -c omegaconf) -eq 0'
    //     sh 'test $(pip list | grep -c hydra) -eq 0'
    //     sh 'pytest -m "torch_tts" --cpu tests/collections/tts/test_torch_tts.py --relax_numba_compat'
    //   }
    // }

    stage('NeMo Installation') {
      steps {
        sh './reinstall.sh release'
      }
    }

    // TODO: remove this when PTL updates their torchtext import logic
    stage('Remove torchtext from PTL Imports') {
      steps {
        sh "sed -i 's/_module_available(\"torchtext\")/False/g' /opt/conda/lib/python3.8/site-packages/pytorch_lightning/utilities/imports.py"
        sh "cat /opt/conda/lib/python3.8/site-packages/pytorch_lightning/utilities/imports.py"
      }
    }

    stage('PyTorch Lightning version') {
      steps {
        sh 'python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"'
      }
    }

    stage('PyTorch Lightning DDP Checks') {
      steps {
        sh 'CUDA_VISIBLE_DEVICES="0,1" python "tests/core_ptl/check_for_ranks.py"'
      }
    }

    stage('Basic Import Checks') {
      steps {
        sh 'python -c "import nemo.collections.asr as nemo_asr"'
        sh 'python -c "import nemo.collections.nlp as nemo_nlp"'
        sh 'python -c "import nemo.collections.tts as nemo_tts"'
      }
    }

    stage('L0: Unit Tests GPU') {
      steps {
        sh 'NEMO_NUMBA_MINVER=0.53 pytest -m "not pleasefixme and not torch_tts" --with_downloads'
      }
    }

    stage('L0: Unit Tests CPU') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      steps {
        sh 'CUDA_VISIBLE_DEVICES="" NEMO_NUMBA_MINVER=0.53 pytest -m "not pleasefixme" --cpu --with_downloads --relax_numba_compat'
      }
    }

    stage('L0: TN/ITN Tests CPU') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('En TN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize.py "1" --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14'
          }
        }
        stage('En ITN grammars') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/inverse_text_normalization/inverse_normalize.py --language en "twenty" --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14'
          }
        }
        stage('Test En non-deterministic TN & Run all En TN/ITN tests (restore grammars from cache)') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES="" python nemo_text_processing/text_normalization/normalize_with_audio.py --text "\$.01" --n_tagged 2 --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14'
            sh 'CUDA_VISIBLE_DEVICES="" pytest tests/nemo_text_processing/en/ -m "not pleasefixme" --cpu --tn_cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14'
          }
        }
      }
    }

    stage('L2: NeMo text processing') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Eng TN') {
          steps {
            sh 'cd tools/text_processing_deployment && python pynini_export.py --output=/home/TestData/nlp/text_norm/output/ --grammars=tn_grammars --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14 --language=en && ls -R /home/TestData/nlp/text_norm/output/ && echo ".far files created "|| exit 1'
            sh 'cd nemo_text_processing/text_normalization/ &&  python run_predict.py --input=/home/TestData/nlp/text_norm/ci/test.txt --input_case="lower_cased" --language=en --output=/home/TestData/nlp/text_norm/output/test.pynini.txt --verbose'
            sh 'cat /home/TestData/nlp/text_norm/output/test.pynini.txt'
            sh 'cmp --silent /home/TestData/nlp/text_norm/output/test.pynini.txt /home/TestData/nlp/text_norm/ci/test_goal_py_12-10.txt || exit 1'
            sh 'rm -rf /home/TestData/nlp/text_norm/output/*'
          }
        }

        stage('L2: Eng ITN export') {
          steps {
            sh 'cd tools/text_processing_deployment && python pynini_export.py --output=/home/TestData/nlp/text_denorm/output/ --grammars=itn_grammars --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14 --language=en && ls -R /home/TestData/nlp/text_denorm/output/ && echo ".far files created "|| exit 1'
            sh 'cd nemo_text_processing/inverse_text_normalization/ &&  python run_predict.py --input=/home/TestData/nlp/text_denorm/ci/test.txt --language=en --output=/home/TestData/nlp/text_denorm/output/test.pynini.txt --verbose'
            sh 'cmp --silent /home/TestData/nlp/text_denorm/output/test.pynini.txt /home/TestData/nlp/text_denorm/ci/test_goal_py.txt || exit 1'
            sh 'rm -rf /home/TestData/nlp/text_denorm/output/*'
          }
        }
        stage('L2: TN with Audio (audio and raw text)') {
          steps {
            sh 'cd nemo_text_processing/text_normalization && \
            python normalize_with_audio.py --language=en --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14 --text "The total amounts to \\$4.76." \
            --audio_data /home/TestData/nlp/text_norm/audio_based/audio.wav | tail -n2 | head -n1 > /tmp/out_raw.txt 2>&1 && \
            cmp --silent /tmp/out_raw.txt /home/TestData/nlp/text_norm/audio_based/result.txt || exit 1'
          }
        }
        stage('L2: TN with Audio (audio and text file)') {
          steps {
            sh 'cd nemo_text_processing/text_normalization && \
            python normalize_with_audio.py --language=en --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14 --text /home/TestData/nlp/text_norm/audio_based/text.txt \
            --audio_data /home/TestData/nlp/text_norm/audio_based/audio.wav | tail -n2 | head -n1 > /tmp/out_file.txt 2>&1 && \
            cmp --silent /tmp/out_file.txt /home/TestData/nlp/text_norm/audio_based/result.txt || exit 1'
          }
        }
        stage('L2: TN with Audio (manifest)') {
          steps {
            sh 'cd nemo_text_processing/text_normalization && \
            python normalize_with_audio.py --language=en --audio_data /home/TestData/nlp/text_norm/audio_based/manifest.json --n_tagged=120 --cache_dir /home/TestData/nlp/text_norm/ci/grammars/4-14'
          }
        }
      }
    }

    stage('L0: Computer Vision Integration') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('MNIST image classification with LeNet-5 Integration Test - on CPU') {
          steps {
            sh 'cd examples/cv && \
            python mnist_lenet5_image_classification_pure_lightning.py trainer.devices=1 \
            trainer.accelerator="cpu" \
            trainer.fast_dev_run=true model.dataset.data_folder=/home/TestData \
            && rm -rf outputs'
          }
        }
      }
    }

    // We have no integration tests, please enable this when one is added
    // stage('L0: Integration Tests GPU') {
    //   steps {
    //     sh 'pytest -s -m "integration and not skipduringci and not pleasefixme"'
    //   }
    // }

    // stage('L0: Integration Tests CPU') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   steps {
    //     sh 'pytest -s -m "integration and not pleasefixme" --cpu'
    //   }
    // }

    // We have no system tests, please enable this when one is added
    // stage('L1: System Tests GPU') {
    //   steps {
    //     sh 'pytest -m "system and not skipduringci and not pleasefixme"'
    //   }
    // }

    // stage('L1: System Tests CPU') {
    //   when {
    //     anyOf{
    //       branch 'dev
    //       changeRequest target: 'main'
    //     }
    //   }
    //   steps {
    //     sh 'pytest -m "system and not pleasefixme" --cpu'
    //   }
    // }

    stage('L2: ASR dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }

        stage('L2: Speech to Text WPE - CitriNet') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
            --config-path="../conf/citrinet/" --config-name="config_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_results'
          }
        }

        stage('L2: Speech Pre-training - CitriNet') {
          steps {
            sh 'python examples/asr/speech_pretraining/speech_pre_training.py \
            --config-path="../conf/ssl/citrinet/" --config-name="citrinet_ssl_ci" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_pre_training_results'
            sh 'rm -rf examples/asr/speech_pre_training_results'
          }
        }

        stage('L2: Speech to Text WPE - Conformer') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
            --config-path="../conf/conformer" --config-name="conformer_ctc_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            model.train_ds.batch_size=4 \
            model.validation_ds.batch_size=4 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_conformer_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_conformer_results'
          }
        }
      }
    }

    stage('L2: Speaker dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {

        stage('Speaker Recognition') {
          steps {
            sh 'python examples/speaker_tasks/recognition/speaker_reco.py \
            model.train_ds.batch_size=10 \
            model.validation_ds.batch_size=2 \
            model.train_ds.manifest_filepath=/home/TestData/an4_speaker/train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_speaker/dev.json \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/speaker_tasks/recognition/speaker_recognition_results'
            sh 'rm -rf examples/speaker_tasks/recognition/speaker_recognition_results'
          }
        }

        stage('Speech to Label') {
          steps {
            sh 'python examples/asr/speech_classification/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            model.test_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            ~model.preprocessor.window_size \
            ~model.preprocessor.window_stride \
            ~model.preprocessor.window \
            ~model.preprocessor.n_mels \
            ~model.preprocessor.n_mfcc \
            ~model.preprocessor.n_fft \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
          }
        }


        stage('Speaker Diarization with ASR Inference') {
          steps {
            sh 'python examples/speaker_tasks/diarization/offline_diarization_with_asr.py \
	    diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
            diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
            diarizer.speaker_embeddings.parameters.save_embeddings=True \
            diarizer.speaker_embeddings.parameters.window_length_in_sec=[1.5] \
            diarizer.speaker_embeddings.parameters.shift_length_in_sec=[0.75] \
            diarizer.speaker_embeddings.parameters.multiscale_weights=[1.0] \
            diarizer.asr.model_path=QuartzNet15x5Base-En \
            diarizer.asr.parameters.asr_based_vad=True \
            diarizer.out_dir=examples/speaker_tasks/diarization/speaker_diarization_asr_results'
            sh 'rm -rf examples/speaker_tasks/diarization/speaker_diarization_asr_results'
          }
        }

        stage('Speaker Diarization Inference') {
          steps {
            sh 'python examples/speaker_tasks/diarization/offline_diarization.py \
	    diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
            diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
            diarizer.speaker_embeddings.parameters.save_embeddings=True \
            diarizer.speaker_embeddings.parameters.window_length_in_sec=1.5 \
            diarizer.speaker_embeddings.parameters.shift_length_in_sec=0.75 \
            diarizer.speaker_embeddings.parameters.multiscale_weights=null \
            diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
            diarizer.out_dir=examples/speaker_tasks/diarization/speaker_diarization_results'
            sh 'rm -rf examples/speaker_tasks/diarization/speaker_diarization_results'
          }
        }
      }
    }
    // TODO: Enable test after 21.08 container is used.
    // stage('L2: ASR DALI dev run') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Speech to Text - DALI AudioToMelSpectrogramPreprocessor') {
    //       steps {
    //         sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         +model.train_ds.use_dali=True \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         +model.validation_ds.use_dali=True \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_results'
    //         sh 'rm -rf examples/asr/speech_to_text_results'
    //       }
    //     }
    //    stage('Speech to Text BPE - DALI AudioToMelSpectrogramPreprocessor') {
    //       steps {
    //         sh 'python examples/asr/asr_ctc/speech_to_text_bpe.py \
    //         --config-path="../conf/citrinet/" --config-name="config_bpe" \
    //         model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
    //         model.tokenizer.type="wpe" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         +model.train_ds.use_dali=True \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         +model.validation_ds.use_dali=True \
    // 	       trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_wpe_results'
    //         sh 'rm -rf examples/asr/speech_to_text_wpe_results'
    //       }
    //     }
    //     // TODO: This would fail due to an unnecessary torchaudio import.
    //     //       To be enabled once torchaudio is available in the container used for CI
    //     // stage('Speech to Text - DALI AudioToMFCCPreprocessor') {
    //     //   steps {
    //     //     sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
    //     //     model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //     //     +model.train_ds.use_dali=True \
    //     //     model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //     //     +model.validation_ds.use_dali=True \
    //     //     model.preprocessor._target_=nemo.collections.asr.modules.AudioToMFCCPreprocessor \
    //     //     ~model.preprocessor.normalize \
    //     //     ~model.preprocessor.features \
    //     //     ~model.preprocessor.frame_splicing \
    //     //     ~model.preprocessor.dither \
    //     //     ~model.preprocessor.stft_conv \
    //     //     +model.n_mels=64 \
    //     //     +model.n_mfcc=64 \
    //     //     trainer.devices=[1] \
    //     //     trainer.accelerator="gpu" \
    //     //     +trainer.fast_dev_run=True \
    //     //     exp_manager.exp_dir=examples/asr/speech_to_text_results'
    //     //     sh 'rm -rf examples/asr/speech_to_text_results'
    //     //   }
    //     // }
    //   }
    // }

    // TODO: Add back once CI is updated
    // stage('L2: ASR RNNT dev run') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Speech to Text - RNNT') {
    //       steps {
    //         sh 'STRICT_NUMBA_COMPAT_CHECK=false python examples/asr/asr_transducer/speech_to_text_rnnt.py \
    //         --config-path="../conf/contextnet_rnnt/" --config-name="config_rnnt.yaml" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         model.train_ds.batch_size=2 \
    //         model.validation_ds.batch_size=2 \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_rnnt_results'
    //         sh 'rm -rf examples/asr/speech_to_text_rnnt_results'
    //       }
    //     }
    //     stage('L2: Speech to Text RNNT WPE') {
    //       steps {
    //         sh 'STRICT_NUMBA_COMPAT_CHECK=false python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    //         --config-path="../conf/contextnet_rnnt/" --config-name="config_rnnt_bpe.yaml" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         model.train_ds.batch_size=2 \
    //         model.validation_ds.batch_size=2 \
    //         model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
    //         model.tokenizer.type="wpe" \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_rnnt_wpe_results'
    //         sh 'rm -rf examples/asr/speech_to_text_rnnt_wpe_results'
    //       }
    //     }
    //   }
    // }

    stage('L2: ASR Multi-dataloader dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text multi-dataloader') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=[/home/TestData/an4_dataset/an4_val.json,/home/TestData/an4_dataset/an4_val.json] \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            +trainer.num_sanity_val_steps=1 \
            exp_manager.exp_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }

        stage('Speech to Label multi-dataloader') {
          steps {
            sh 'python examples/asr/speech_classification/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=[/home/TestData/speech_commands/test_manifest.json,/home/TestData/speech_commands/test_manifest.json] \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            +trainer.num_sanity_val_steps=1 \
            model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            ~model.preprocessor.window_size \
            ~model.preprocessor.window_stride \
            ~model.preprocessor.window \
            ~model.preprocessor.n_mels \
            ~model.preprocessor.n_mfcc \
            ~model.preprocessor.n_fft \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
          }
        }
      }
    }

    stage('L2: Speech Transcription') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text Transcribe') {
          steps {
            sh 'python examples/asr/transcribe_speech.py \
            pretrained_name="QuartzNet15x5Base-En" \
            audio_dir="/home/TestData/an4_transcribe/test_subset/" \
            output_filename="stt_test_res.json" \
            amp=true'
            sh 'rm -rf stt_test_res.json'
          }
        }
      }
    }

    stage('L2: Segmentation Tool') {
      when {
            anyOf {
              branch 'main'
              changeRequest target: 'main'
            }
      }
      stages {
        stage('Install ctc_segmentation requirements') {
            steps {
            sh 'cd tools/ctc_segmentation && \
            pip install -r requirements.txt && \
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg'
            }
        }

        stage('Parallel ctc_segmentation test') {
          failFast true
          parallel {
            stage('L2: Eng CitriNet with .wav') {
              steps {
                sh 'cd tools/ctc_segmentation && \
            TIME=`date +"%Y-%m-%d-%T"` && \
            /bin/bash run_segmentation.sh \
            --MODEL_NAME_OR_PATH="stt_en_citrinet_512_gamma_0_25" \
            --DATA_DIR=/home/TestData/ctc_segmentation/eng \
            --OUTPUT_DIR=/home/TestData/ctc_segmentation/eng/output${TIME} \
            --LANGUAGE=en \
            --USE_NEMO_NORMALIZATION="TRUE" && \
            python /home/TestData/ctc_segmentation/verify_alignment.py \
            -r /home/TestData/ctc_segmentation/eng/eng_valid_segments_1.7.txt \
            -g /home/TestData/ctc_segmentation/eng/output${TIME}/verified_segments/nv_test_segments.txt && \
            rm -rf /home/TestData/ctc_segmentation/eng/output${TIME}'
              }
            }
            stage('L2: Ru QN with mp3') {
              steps {
                sh 'cd tools/ctc_segmentation && \
            TIME=`date +"%Y-%m-%d-%T"` && \
            /bin/bash run_segmentation.sh \
            --MODEL_NAME_OR_PATH=/home/TestData/ctc_segmentation/QuartzNet15x5-Ru-e512-wer14.45.nemo \
            --DATA_DIR=/home/TestData/ctc_segmentation/ru \
            --OUTPUT_DIR=/home/TestData/ctc_segmentation/ru/output${TIME} \
            --LANGUAGE=ru \
            --ADDITIONAL_SPLIT_SYMBOLS=";" && \
            python /home/TestData/ctc_segmentation/verify_alignment.py \
            -r /home/TestData/ctc_segmentation/ru/valid_ru_segments_1.7.txt \
            -g /home/TestData/ctc_segmentation/ru/output${TIME}/verified_segments/ru_segments.txt && \
            rm -rf /home/TestData/ctc_segmentation/ru/output${TIME}'
              }
            }
          }
        }
      }
    }

    // TODO: add test once megatron-bert is supported again
    // stage('L2: Multi-GPU Megatron finetuning') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('L2: Cased Megatron finetuning on MRPC') {
    //       steps {
    //         sh 'cd examples/nlp/glue_benchmark && \
    //     python glue_benchmark.py \
    //     model.dataset.data_dir=/home/TestData/nlp/glue_fake/MRPC \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.use_cache=false \
    //     model.language_model.pretrained_model_name=megatron-bert-345m-cased \
    //     trainer.accelerator=gpu \
    //     trainer.strategy=ddp \
    //     exp_manager=null'
    //       }
    //     }
    //   }
    // }

    stage('L2: STS-b') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('GLUE STS-b with AlBERT') {
          steps {
            sh 'python examples/nlp/glue_benchmark/glue_benchmark.py \
            model.dataset.use_cache=false \
            model.task_name=sts-b \
            model.dataset.data_dir=/home/TestData/nlp/glue_fake/STS-B \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            model.language_model.pretrained_model_name=albert-base-v1 \
            exp_manager=null'
          }
        }
       stage('Test Restore with AlBERT') {
          steps {
            sh 'python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
            +do_training=false \
            +do_testing=true \
            pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_and_Capitalization_albert.nemo \
            +model.test_ds.use_cache=false \
            ~model.train_ds \
            ~model.validation_ds \
            model.test_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage('Test Restore with RoBERTa') {
          steps {
            sh 'python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
            +do_training=false \
            +do_testing=true \
            pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_and_Capitalization_roberta.nemo \
            +model.test_ds.use_cache=false \
            ~model.train_ds \
            ~model.validation_ds \
            model.test_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
      }
    }
    stage('L2: SGD-GEN') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('SGD-GEN') {
          steps {
            sh 'TRANSFORMERS_OFFLINE=0 && cd examples/nlp/dialogue_state_tracking_generative && \
            python sgd_gen.py \
            model.dataset.data_dir=/home/TestData/nlp/sgd_small \
            model.language_model.lm_checkpoint=/home/TestData/nlp/gpt2/pytorch_model.bin\
            model.tokenizer.vocab_file=/home/TestData/nlp/gpt2/vocab.json\
            model.dataset.dialogues_example_dir=sgd_gen_outputs \
            model.dataset.task_name=debug_sample \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[0] \
            model.dataset.use_cache=false \
            model.tokenizer.special_tokens={pad_token:"endoftext"} \
            model.tokenizer.tokenizer_name=gpt2 \
            model.tokenizer.vocab_file=/home/TestData/nlp/gpt2/vocab.json\
            model.language_model.pretrained_model_name=/home/TestData/nlp/gpt2 \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_outputs'
          }
        }
        stage('SGD-GEN Backward compatible with SGDQA') {
          steps {
            sh 'TRANSFORMERS_OFFLINE=0 && cd examples/nlp/dialogue_state_tracking_generative && \
            python sgd_gen.py \
            model.dataset.data_dir=/home/TestData/nlp/sgd_small \
            model.dataset.dialogues_example_dir=sgd_gen_bert_outputs \
            model.dataset.task_name=debug_sample \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[1] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-cased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_bert_outputs'
          }
        }
        stage('SGD-GEN Backward compatible with IntentSlotClassificationModel') {
          steps {
            sh 'TRANSFORMERS_OFFLINE=0 && cd examples/nlp/dialogue_state_tracking_generative && \
            python sgd_gen.py \
            model.dataset.data_dir=/home/TestData/nlp/processed_assistant \
            model.dataset.dialogues_example_dir=sgd_gen_bert_intent_classification_outputs \
            model.dataset.task=assistant \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[1] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-uncased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_bert_intent_classification_outputs && TRANSFORMERS_OFFLINE=1'
          }
        }
      }
    }
    stage('L2: Parallel BERT SQUAD v1.1 / v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('BERT SQUAD 1.1') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.test_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            model.test_ds.num_samples=2 \
            model.test_ds.batch_size=2 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage('BERT SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
       stage('Duplex Text Normalization with Tarred dataset') {
          steps {
            sh 'cd examples/nlp/duplex_text_normalization && \
            python duplex_text_normalization_train.py \
            data.validation_ds.data_path=/home/TestData/nlp/duplex_text_norm/small_test.tsv \
            mode=tn \
            lang=en \
            tagger_model.do_training=false \
            decoder_model.transformer=t5-small \
            data.validation_ds.batch_size=2 \
            data.train_ds.use_cache=false \
            data.validation_ds.use_cache=false \
            data.test_ds.batch_size=2 \
            data.train_ds.decoder_data_augmentation=false \
            data.train_ds.num_workers=2 \
            decoder_trainer.devices=[0,1] \
            decoder_trainer.accelerator="gpu" \
            data.train_ds.use_tarred_dataset=true \
            +decoder_trainer.fast_dev_run=true \
            decoder_exp_manager.create_checkpoint_callback=false \
            data.train_ds.tar_metadata_file=/home/TestData/nlp/duplex_text_norm/tarred_small/metadata.json \
            data.test_ds.use_cache=false \
            data.test_ds.data_path=/home/TestData/nlp/duplex_text_norm/small_test.tsv'

          }
        }
      }
    }
    // Runs out of memory on the 12G TITAN V (GPU 0 on main CI)
    // TODO: add when megatron bert is supported again in NeMo
    // stage('L2: MegaBERT Token Classification') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh 'cd examples/nlp/token_classification && \
    //     python token_classification_train.py \
    //     model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
    //     model.language_model.pretrained_model_name=megatron-bert-345m-uncased \
    //     model.train_ds.batch_size=10 \
    //     model.dataset.max_seq_length=50 \
    //     model.dataset.use_cache=false \
    //     trainer.accelerator=gpu \
    //     trainer.strategy=ddp \
    //     trainer.precision=16 \
    //     trainer.devices=[1] \
    //     trainer.accelerator="gpu" \
    //     +trainer.fast_dev_run=true \
    //     exp_manager=null'
    //   }
    // }

    stage('L2: Parallel SQUAD v1.1 & v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        // TODO: use megatron bert when supported again
        stage('SQUAD v2.0 with DistilBERT Uncased') {
        // stage('SQUAD v2.0 with Megatron with ckpt & config') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          // model.language_model.pretrained_model_name=megatron-bert-uncased  \
          // model.language_model.lm_checkpoint=/home/TestData/nlp/megatron_345m_uncased/model_optim_rng.pt \
          // model.language_model.config_file=/home/TestData/nlp/megatron_345m_uncased/345m_config.json \
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=1 \
            model.train_ds.num_samples=1 \
            model.validation_ds.batch_size=1 \
            model.validation_ds.num_samples=1 \
            trainer.accelerator=gpu \
            trainer.strategy=ddp \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=distilbert-base-uncased  \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage('RoBERTa SQUAD 1.1') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.language_model.pretrained_model_name=roberta-base \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage ('Text Classification with BERT Test') {
          steps {
            sh 'cd examples/nlp/text_classification && \
            python text_classification_with_bert.py \
            model.dataset.num_classes=6 \
            model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
            model.validation_ds.file_path=/home/TestData/nlp/retail_text_classification/dev.tsv \
            model.language_model.pretrained_model_name=distilbert-base-uncased \
            model.train_ds.batch_size=10 \
            model.dataset.max_seq_length=50 \
            model.dataset.use_cache=false \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Intent and Slot Classification Tasks') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Intent and Slot Classification') {
          steps {
            sh 'cd examples/nlp/intent_slot_classification && \
            python intent_slot_classification.py \
            model.data_dir=/home/TestData/nlp/retail \
            model.validation_ds.prefix=dev \
            model.test_ds.prefix=dev \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=checkpoints'
            sh 'rm -rf checkpoints'
          }
        }
        stage('L2: Multi-Label Intent and Slot Classification') {
          steps {
            sh 'cd examples/nlp/intent_slot_classification && \
            python multi_label_intent_slot_classification.py \
            model.data_dir=/home/TestData/nlp/new_multiatis \
            model.validation_ds.prefix=dev \
            model.test_ds.prefix=dev \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=checkpoints2'
            sh 'rm -rf checkpoints2'
          }
        }
      }
    }

    // TODO: add when megatron-bert is supported again
    // stage('L2: Model Parallel Size 2 Megatron Text Classification') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python text_classification_with_bert.py \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     trainer.num_nodes=1 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.num_classes=6 \
    //     model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
    //     model.train_ds.batch_size=4 \
    //     model.language_model.pretrained_model_name=megatron-bert-uncased \
    //     model.language_model.config_file=/home/TestData/nlp/mp_2_bert_toy/config.json \
    //     model.language_model.lm_checkpoint=/home/TestData/nlp/mp_2_bert_toy/iter_2000000 \
    //     model.nemo_path=null \
    //     ~model.infer_samples \
    //     exp_manager=null'
    //   }
    // }

    // stage('L2: Model Parallel Size 2 Megatron Autoresume') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python text_classification_with_bert.py \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     trainer.num_nodes=1 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     trainer.max_epochs=1 \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.num_classes=6 \
    //     model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
    //     model.train_ds.batch_size=4 \
    //     model.language_model.pretrained_model_name=megatron-bert-uncased \
    //     model.language_model.config_file=/home/TestData/nlp/mp_2_bert_toy/config.json \
    //     model.language_model.lm_checkpoint=/home/TestData/nlp/mp_2_bert_toy/iter_2000000 \
    //     model.nemo_path=null \
    //     ~model.infer_samples \
    //     +exp_manager.explicit_log_dir=/home/TestData/nlp/mp_autoresume \
    //     +exp_manager.resume_if_exists=true'
    //   }
    // }

    // stage('L2: Model Parallel Size 2 Megatron Evaluation from .nemo') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python model_parallel_text_classification_evaluation.py \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     trainer.num_nodes=1 \
    //     model.dataset.num_classes=6 \
    //     model.test_ds.file_path=/home/TestData/nlp/retail_text_classification/dev.tsv \
    //     model.nemo_path=/home/TestData/nlp/mp_2_nemo/retail_text_class_350M.nemo \
    //     exp_manager=null'
    //   }
    // }

    // stage('L2: Model Parallel Size 2 Megatron Train from .nemo') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/token_classification && \
    //     python token_classification_train.py \
    //     pretrained_model=/home/TestData/nlp/mp_2_nemo/ner_350M.nemo \
    //     model.dataset.data_dir=/home/TestData/nlp/ner/ \
    //     model.train_ds.batch_size=2 \
    //     model.dataset.use_cache=false \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.class_balancing="weighted_loss" \
    //     exp_manager=null'
    //   }
    // }

    stage('L2: Parallel NLP Examples 2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('NER finetuning from pretrained Test') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification_train.py \
            pretrained_model=ner_en_bert \
            model.dataset.data_dir=/home/TestData/nlp/ner/ \
            model.train_ds.batch_size=2 \
            model.dataset.use_cache=false \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            model.dataset.class_balancing="weighted_loss" \
            exp_manager.exp_dir=null'
          }
        }
        stage ('Punctuation and capitalization finetuning from pretrained test') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python punctuation_capitalization_train_evaluate.py \
            pretrained_model=punctuation_en_bert \
            model.train_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
            model.validation_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
            model.test_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
            +model.train_ds.use_cache=false \
            +model.validation_ds.use_cache=false \
            +model.test_ds.use_cache=false \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=null'
          }
        }
        stage ('NER with TurkuNLP/bert-base-finnish-cased-v1') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification_train.py \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name="TurkuNLP/bert-base-finnish-cased-v1" \
            exp_manager.exp_dir=null'
          }
        }
        stage('Evaluation script for Token Classification') {
          steps {
            sh 'python examples/nlp/token_classification/token_classification_evaluate.py \
            model.dataset.data_dir=/home/TestData/nlp/ner/ \
            model.dataset.use_cache=false \
            pretrained_model=/home/TestData/nlp/pretrained_models/NER_Model_with_BERT_base_uncased.nemo'
          }
        }
        stage('Evaluation script for Punctuation') {
          steps {
            sh 'python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
            +do_training=false \
            +do_testing=true \
            model.test_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
            ~model.train_ds \
            ~model.validation_ds \
            +model.test_ds.use_cache=false \
            pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_Capitalization_with_DistilBERT_base_uncased.nemo'
          }
        }
        stage('L2: Punctuation & Capitalization, 2GPUs with DistilBERT, Fine-tuning on different data') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            mkdir -p tmp_data && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt tmp_data/ && \
            python punctuation_capitalization_train_evaluate.py \
              model.train_ds.use_tarred_dataset=false \
              model.train_ds.ds_item=tmp_data \
              model.validation_ds.ds_item=tmp_data \
              model.test_ds.ds_item=tmp_data \
              model.language_model.pretrained_model_name=distilbert-base-uncased \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.accelerator="gpu" \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              +exp_manager.explicit_log_dir=/home/TestData/nlp/token_classification_punctuation/output \
              +do_testing=true && \
            mv tmp_data tmp_data2 && \
            python punctuation_capitalization_train_evaluate.py \
              model.train_ds.use_tarred_dataset=false \
              model.train_ds.ds_item=tmp_data2 \
              model.validation_ds.ds_item=tmp_data2 \
              model.test_ds.ds_item=tmp_data2 \
              pretrained_model=/home/TestData/nlp/token_classification_punctuation/output/checkpoints/Punctuation_and_Capitalization.nemo \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.accelerator="gpu" \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              exp_manager=null && \
            rm -r tmp_data2 && \
            rm -rf /home/TestData/nlp/token_classification_punctuation/output/*'
          }
        }
      }
      post {
        always {
          sh 'pwd && ls nemo_* && rm -rf nemo_experiments && ls nemo_*'
        }
      }
    }
    stage('Punctuation & Capitalization tarred dataset') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      stages {
        stage('create and use tarred dataset') {
          steps {
            sh 'data_dir=/home/TestData/nlp/token_classification_punctuation && \
            usual_data=${data_dir}/wmt_wiki_10000 && \
            tarred_data=${data_dir}/train_tarred && \
            TIME=`date +"%Y-%m-%d-%T"` \
            output=${data_dir}/output_${TIME} && \
            tokens_in_batch=2000 && \
            max_seq_length=512 && \
            lm_model=distilbert-base-uncased && \
            python examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py \
              --text ${usual_data}/input.txt \
              --labels ${usual_data}/labels.txt \
              --output_dir ${tarred_data} \
              --tokens_in_batch ${tokens_in_batch} \
              --max_seq_length 512 \
              --lines_per_dataset_fragment 2000 \
              --num_batches_per_tarfile 5 \
              --tar_file_prefix punctuation_capitalization \
              --tokenizer_name ${lm_model} \
              --use_fast_tokenizer \
              --pad_label O \
              --n_jobs 3 && \
            echo "Number of tarred files in dataset:" && \
            ls ${tarred_data}/*.tar | wc -l && \
            echo "Label id files in dataset:" && \
            ls ${tarred_data}/*.csv && \
            metadata_file=${tarred_data}/metadata.punctuation_capitalization.tokens${tokens_in_batch}.max_seq_length${max_seq_length}.${lm_model}.json && \
            python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
              model.validation_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
              model.test_ds.ds_item=/home/TestData/nlp/token_classification_punctuation/ \
              model.train_ds.ds_item=${tarred_data} \
              model.language_model.pretrained_model_name=${lm_model} \
              model.train_ds.use_tarred_dataset=true \
              model.train_ds.tar_metadata_file=${metadata_file} \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.accelerator="gpu" \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              +exp_manager.explicit_log_dir=${output} && \
            rm -rf ${output}/* ${tarred_data}'
          }
        }
      }
    }
    stage('Punctuation & Capitalization inference') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      stages {
        stage('Restore punctuation and capitalization in long text') {
          steps {
            sh 'python examples/nlp/token_classification/punctuate_capitalize_infer.py \
            --input_manifest /home/TestData/nlp/token_classification_punctuation/iwslt_tst2019.manifest \
            --output_text iwslt_inference_result.txt \
            --max_seq_length 92 \
            --step 8 \
            --margin 16 \
            --pretrained_name punctuation_en_bert \
            --batch_size 32 && \
            rm iwslt_inference_result.txt'
          }
        }
      }
    }
    stage('L2: Parallel Pretraining BERT pretraining from Text/Preprocessed') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Pretraining BERT pretraining from Text') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_text_config.yaml \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              trainer.precision=16 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wikitext-2/train.txt  \
              model.train_ds.batch_size=32 \
              model.validation_ds.data_file=/home/TestData/nlp/wikitext-2/valid.txt  \
              model.validation_ds.batch_size=32 \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/bert_3200.json \
              model.optim.lr=0.01 \
              model.optim.sched.warmup_ratio=0.1 \
              model.tokenizer.tokenizer_name=sentencepiece \
              model.tokenizer.tokenizer_model=/home/TestData/nlp/wikitext-2/tokenizer_bpe_v3193/tokenizer.model \
              model.mask_prob=0.15 \
              model.short_seq_prob=0.1 \
              exp_manager.exp_dir=PretrainingBERTFromText \
              '
              sh 'rm -f /home/TestData/nlp/wikitext-2/*.pkl'
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromText'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
        stage('L2: Pretraining BERT from Preprocessed') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_preprocessed_config.yaml \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              trainer.precision=16 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wiki_book_mini/training \
              model.train_ds.batch_size=8 \
              model.language_model.lm_checkpoint=/home/TestData/nlp/bert_ckpts/nemo1.0/bert_base_uncased_mlm_final_1074591_nemo1.0.pt \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/uncased_L-12_H-768_A-12.json \
              model.optim.lr=0.875e-4 \
              model.optim.weight_decay=0.01 \
              model.optim.sched.warmup_ratio=0.01 \
              exp_manager.exp_dir=PretrainingBERTFromPreprocessed \
              exp_manager.create_checkpoint_callback=False \
              '
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromPreprocessed'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
      }
    }

    stage('L2: Entity Linking') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('Self Alignment Pretraining BERT') {
           steps {
             sh 'cd examples/nlp/entity_linking && \
             python self_alignment_pretraining.py \
             project_dir=. \
             trainer.val_check_interval=3 \
             model.raw_data=None \
             model.train_ds.data_file=/home/TestData/nlp/entity_linking/tiny_example_train_pairs.tsv \
             model.validation_ds.data_file=/home/TestData/nlp/entity_linking/tiny_example_validation_pairs.tsv \
             model.train_ds.batch_size=8 \
             model.validation_ds.batch_size=8 \
             exp_manager.exp_dir=null'
          }
        }
      }
    }

    stage('L2: NMT Attention is All You Need Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Training Post-LN') {
            steps {
              sh 'python examples/nlp/machine_translation/enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=false \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.encoder.num_layers=1 \
              model.encoder.hidden_size=64 \
              model.encoder.inner_size=256 \
              model.decoder.num_layers=1 \
              model.decoder.hidden_size=64 \
              model.decoder.inner_size=256 \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.val_check_interval=2 \
              +trainer.limit_val_batches=1 \
              +trainer.max_steps=2 \
              trainer.precision=16 \
              +exp_manager.explicit_log_dir=examples/nlp/machine_translation/nmt_results \
              +exp_manager.create_checkpoint_callback=true \
              '
              sh 'python examples/nlp/machine_translation/enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.encoder.num_layers=1 \
              model.encoder.hidden_size=64 \
              model.encoder.inner_size=256 \
              model.decoder.num_layers=1 \
              model.decoder.hidden_size=64 \
              model.decoder.inner_size=256 \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.val_check_interval=10 \
              +trainer.limit_val_batches=1 \
              +trainer.limit_test_batches=1 \
              +trainer.max_steps=10 \
              +exp_manager.explicit_log_dir=examples/nlp/machine_translation/nmt_results \
              +exp_manager.create_checkpoint_callback=true \
              +exp_manager.resume_if_exists=True \
              '
            }
        }

        stage('L2: NMT Training Pre-LN') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.encoder.pre_ln=true \
              model.decoder.pre_ln=true \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              +trainer.limit_test_batches=2 \
              exp_manager=null \
              '
            }
        }
        stage('L2: NMT Multi-Validation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref \
              model.validation_ds.src_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src] \
              model.validation_ds.tgt_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref] \
              model.test_ds.src_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src] \
              model.test_ds.tgt_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref] \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              +trainer.limit_test_batches=2 \
              exp_manager=null \
              '
            }
        }
      }
    }

    stage('L2: NMT Attention is All You Need Inference') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Inference - PostLN') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python nmt_transformer_infer.py \
              --model=/home/TestData/nlp/nmt/toy_data/TransformerLargeDe-En.nemo \
              --srctext=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.test.src \
              --tgtout=/home/TestData/nlp/nmt/toy_data/out.txt \
              --target_lang en \
              --source_lang de \
              '
            }
        }
        stage('L2: NMT Inference - Pre-LN') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python nmt_transformer_infer.py \
              --model=/home/TestData/nlp/nmt/toy_data/en_de_24x6_preln.nemo \
              --srctext=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.test.src \
              --tgtout=/home/TestData/nlp/nmt/toy_data/out.txt \
              --target_lang de \
              --source_lang en \
              '
            }
        }
      }
    }
    stage('L2: NMT with HuggingFace') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Pretrained HF Encoder') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=huggingface \
              model.shared_tokenizer=False \
              model.encoder_tokenizer.library=huggingface \
              model.encoder.library=huggingface \
              model.encoder.model_name=distilbert-base-cased \
              model.encoder.pretrained=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tokens_in_batch=128 \
              model.validation_ds.tokens_in_batch=128 \
              model.test_ds.tokens_in_batch=128 \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder.hidden_size=768 \
              model.decoder.inner_size=256 \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Custom HF Encoder') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=huggingface \
              model.shared_tokenizer=True \
              model.encoder_tokenizer.library=yttm \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.encoder.library=huggingface \
              model.encoder.model_name=null \
              model.encoder.pretrained=false \
              +model.encoder._target_=transformers.BertConfig \
              +model.encoder.hidden_size=48 \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tokens_in_batch=128 \
              model.validation_ds.tokens_in_batch=128 \
              model.test_ds.tokens_in_batch=128 \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder.hidden_size=48 \
              model.decoder.inner_size=256 \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }
      }
    }


    stage('L2: NMT Tarred Dataset Creation') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Auto Tarred Dataset Creation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_training=false \
              model.preproc_out_dir=$PWD/preproc_out_dir \
              model.train_ds.use_tarred_dataset=true \
              model.train_ds.n_preproc_jobs=2 \
              model.train_ds.lines_per_dataset_fragment=500 \
              model.train_ds.num_batches_per_tarfile=10 \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.vocab_size=2000 \
              model.decoder_tokenizer.vocab_size=2000 \
              ~model.test_ds \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Script Tarred Dataset Creation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python create_tarred_parallel_dataset.py \
              --src_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              --tgt_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              --out_dir $PWD/out_dir \
              --encoder_tokenizer_vocab_size=2000 \
              --decoder_tokenizer_vocab_size=2000 \
              --tokens_in_batch=1000 \
              --lines_per_dataset_fragment=500 \
              --num_batches_per_tarfile=10 \
              --n_preproc_jobs=2 \
              '
            }
        }
      }
    }

    // stage('L2: NMT Bottleneck Fallback') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('L2: seq2seq (no bottleneck)') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=nll \
    //           model.encoder.arch=seq2seq \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref \
    //           model.validation_ds.src_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src] \
    //           model.validation_ds.tgt_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref] \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           trainer.devices=[1] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //   }
    // }
    // stage('L2: NMT Bottleneck Architecture') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Bridge Encoder (identity)') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=nll \
    //           model.encoder.arch=bridge \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=identity \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //		 trainer.devices=[0] \
    // 		 trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //     stage('Perceiver Encoder (params)') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=nll \
    //           model.encoder.arch=perceiver \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           trainer.devices=[1] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //   }
    // }
    // stage('L2: NMT Bottleneck LVM') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('VAE') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=vae \
    //           model.encoder.arch=perceiver \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           trainer.devices=[0] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //     stage('MIM') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=mim \
    //           model.encoder.arch=perceiver \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
    //           trainer.devices=[1] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //   }
    // }
    stage('L2: Megatron Bert Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=2 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=2 \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: BioMegatron Bert NER Task') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/token_classification/token_classification_train.py \
        exp_manager.exp_dir=examples/nlp/language_modeling/token_classification_results \
        trainer.max_epochs=1 \
        model.dataset.data_dir=/home/TestData/nlp/ner \
        model.language_model.pretrained_model_name=biomegatron345m_biovocab_30k_cased \
        model.tokenizer.tokenizer_name=null"
        sh "rm -rf examples/nlp/language_modeling/token_classification_results"
      }
    }
    // stage('L2: Megatron P-Tuning GPT LM') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python  examples/nlp/text_classification/ptune_text_classification.py \
    //     trainer.devices=2 \
    //     trainer.accelerator=gpu \
    //     trainer.max_epochs=1 \
    //     +trainer.limit_val_batches=10 \
    //     +trainer.limit_train_batches=10 \
    //     +trainer.limit_test_batches=10 \
    //     exp_manager.exp_dir=examples/nlp/language_modeling/ptune_results \
    //     model.tokenizer.vocab_file=/home/TestData/nlp/ptune/gpt2-vocab.json \
    //     model.tensor_model_parallel_size=2 \
    //     model.tokenizer.merge_file=/home/TestData/nlp/ptune/gpt2-merges.txt \
    //     model.language_model.nemo_file=/home/TestData/nlp/ptune/small_gpt.nemo \
    //     model.dataset.classes=[positive,neutral,negative] \
    //     model.train_ds.file_path=/home/TestData/nlp/ptune/data/train_0.txt \
    //     model.validation_ds.file_path=/home/TestData/nlp/ptune/data/validation_0.txt \
    //     model.test_ds.file_path=/home/TestData/nlp/ptune/data/test_0.txt "
    //     sh "rm -rf examples/nlp/language_modeling/ptune_results"
    //   }
    // }
    stage('L2: Megatron GPT Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
//        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
//        trainer.devices=2 \
//        trainer.accelerator=gpu \
//        trainer.log_every_n_steps=1 \
//        trainer.val_check_interval=10 \
//        trainer.limit_val_batches=1 \
//        trainer.accumulate_grad_batches=1 \
//        trainer.max_steps=20 \
//        trainer.precision=16 \
//        trainer.gradient_clip_val=1.0 \
//        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
//        exp_manager.resume_if_exists=True \
//        model.tensor_model_parallel_size=2 \
//        model.optim.name=fused_adam \
//        model.optim.lr=2e-4 \
//        model.optim.sched.warmup_steps=2 \
//        model.optim.sched.constant_steps=2 \
//        model.optim.sched.min_lr=8e-5 \
//        model.max_position_embeddings=128 \
//        model.encoder_seq_length=128 \
//        model.data.seq_length=128 \
//        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
//        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
//        model.num_layers=8 \
//        model.hidden_size=256 \
//        model.num_attention_heads=8 \
//        model.activations_checkpoint_method='block' \
//        model.activations_checkpoint_num_layers=1 \
//        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
//        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    stage('L2: Megatron GPT Pretraining and Resume Training PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        // trainer.devices=2 \
        // trainer.log_every_n_steps=1 \
        // trainer.val_check_interval=10 \
        // trainer.limit_val_batches=2 \
        // trainer.accumulate_grad_batches=1 \
        // trainer.max_steps=20 \
        // trainer.precision=16 \
        // trainer.gradient_clip_val=1.0 \
        // exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        // exp_manager.resume_if_exists=True \
        // model.pipeline_model_parallel_size=2 \
        // model.tensor_model_parallel_size=1 \
        // model.optim.name=fused_adam \
        // model.optim.lr=2e-4 \
        // model.optim.sched.warmup_steps=2 \
        // model.optim.sched.constant_steps=2 \
        // model.optim.sched.min_lr=8e-5 \
        // model.max_position_embeddings=128 \
        // model.encoder_seq_length=128 \
        // model.data.seq_length=128 \
        // model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        // model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        // model.num_layers=8 \
        // model.hidden_size=256 \
        // model.num_attention_heads=8 \
        // model.activations_checkpoint_method='block' \
        // model.activations_checkpoint_num_layers=1 \
        // model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        // model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    stage('L2: Megatron GPT Eval') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
            model_file=/home/TestData/nlp/megatron_gpt/125M/megatron_gpt.nemo \
            prompts=['How to fix GPU memory? A:'] \
            tensor_model_parallel_size=1 \
            inference.tokens_to_generate=32 \
            trainer.precision=16"
      }
    }
  
    stage('L2: Megatron GPT Prompt Tuning and Inference') {
      when {
	anyOf {
	  branch 'main'
	  changeRequest target: 'main'
	}
      }
      failFast true
      steps {
	sh "python tests/collections/nlp/test_prompt_tuning.py"
	sh "python examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py \
    --config-name=megatron_prompt_tuning_gpt \
    restore_from_path='/home/TestData/nlp/megatron_gpt/125M/megatron_gpt.nemo' \
    trainer.val_check_interval=2 \
    trainer.max_steps=5 \
    model.new_prompt_tags=['Winogrande, BoolQ'] \
    model.new_prompt_init_text=['logic choose person name, None'] \
    model.new_prompt_init_methods=['text, random'] \
    model.data.train_ds='/home/TestData/nlp/prompt_tuning/wino_bool_prompt_tuning_train.json' \
    model.data.valid_ds='/home/TestData/nlp/prompt_tuning/wino_bool_prompt_tuning_val.json' \
    +model.data.test_ds='/home/TestData/nlp/prompt_tuning/wino_bool_prompt_tuning_val.json' \
    model.micro_batch_size=2 \
    model.global_batch_size=4 \
    model.optim.lr=2e-2 \
    model.optim.sched.min_lr=2e-3 \
    model.optim.sched.warmup_steps=2 \
    model.optim.sched.constant_steps=8 \
    model.encoder_seq_length=2048"
  // disable it for now need to fix this later.
	//sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
	//    --use_soft_prompts \
	//    --model_file=nemo_experiments/PromptTuning/checkpoints/PromptTuning.nemo \
	//    --tokens_to_generate=3 \
	//    --prompt_tag='Winogrande' \
	//    --prompt='option1: wood option2: bag sentence: The _ is soft. answer:'"
	sh "rm -rf nemo_experiments"
      }
    }


    // TODO: Add this test back. Test was failing on CI machines due to HW error
    // stage('L2: Megatron GPT Convert from Megatron-LM checkpoing and Eval') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python -m torch.distributed.launch --nproc_per_node=2 \
    //     examples/nlp/language_modeling/megatron_lm_ckpt_to_nemo.py \
    //     --checkpoint_folder=/home/TestData/nlp/megatron_gpt/data/gpt/iter_0008700 \
    //     --checkpoint_name=model_optim_rng.pt \
    //     --hparams_file=/home/TestData/nlp/megatron_gpt/data/gpt/iter_0008700/hparams.yaml \
    //     --nemo_file_path=examples/nlp/language_modeling/small_gpt.nemo \
    //     --model_type=gpt \
    //     --pipeline_model_parallel_size=1 \
    //     --gpus_per_node=2 \
    //     --tensor_model_parallel_size=2"
    //     sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
    //     --model_file=examples/nlp/language_modeling/small_gpt.nemo \
    //     --tokens_to_generate=32 \
    //     --tensor_model_parallel_size=2 \
    //     --prompt='This is a test.'"
    //     sh "rm examples/nlp/language_modeling/small_gpt.nemo"
    //   }
    // }
    stage('L2: Megatron Change Partitions') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_change_num_partitions.py \
            --model_file \
            /home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
            --target_file \
            /home/TestData/nlp/megatron_gpt/TP2/test-split.nemo \
            --tensor_model_parallel_size \
            2 \
            --target_tensor_model_parallel_size \
            1"
          sh "rm /home/TestData/nlp/megatron_gpt/TP2/test-split.nemo"
      }
    }
    stage('L2: Megatron T5 Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='swiglu' \
        model.bias_gelu_fusion=False \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='swiglu' \
        model.bias_gelu_fusion=False \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 Pretraining and Resume Training PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='gelu' \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='gelu' \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 Eval') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_t5_eval.py \
            --model_file \
            /home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            --prompt \
            'How do I fix my GPU memory issue? I am seeing <mask> out of memory.' \
            --tensor_model_parallel_size 1"
      }
    }
    stage('L2: Megatron BART Pretraining and Resume Training, TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='reglu' \
        model.bias_gelu_fusion=False \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='reglu' \
        model.bias_gelu_fusion=False \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "rm -rf examples/nlp/language_modeling/bart_pretrain_results"
      }
    }
    stage('L2: Megatron BART Pretraining and Resume Training, PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='geglu' \
        model.bias_gelu_fusion=False \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=128 \
        model.num_layers=4 \
        model.hidden_size=64 \
        model.num_attention_heads=8 \
        model.activation='geglu' \
        model.bias_gelu_fusion=False \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "rm -rf examples/nlp/language_modeling/bart_pretrain_results"
      }
    }
    stage('L2: Megatron T5 GLUE/XNLI Finetuning') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        // TODO(Oktai15): update it in 1.8.0 version
        stage('T5 GLUE RTE') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_glue.py \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=1 \
            +trainer.limit_val_batches=2 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_steps=2 \
            trainer.precision=16 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_glue_results \
            model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            model.pipeline_model_parallel_size=1 \
            model.pipeline_model_parallel_split_rank=0 \
            model.data.train_ds.task_name=rte \
            model.data.train_ds.global_batch_size=4 \
            model.data.train_ds.micro_batch_size=2 \
            model.data.validation_ds.global_batch_size=4 \
            model.data.validation_ds.micro_batch_size=2 \
            model.data.train_ds.file_path=/home/TestData/nlp/megatron_t5/data/train_ci.tsv \
            model.data.validation_ds.task_name=rte \
            model.data.validation_ds.file_path=/home/TestData/nlp/megatron_t5/data/dev_ci.tsv \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_glue_results"
          }
        }
        stage('T5 GLUE XNLI') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_glue.py \
            -cn megatron_t5_config_finetune_glue_xnli \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=1 \
            +trainer.limit_val_batches=2 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_steps=2 \
            trainer.precision=16 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_xnli_results \
            model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            model.pipeline_model_parallel_size=1 \
            model.pipeline_model_parallel_split_rank=0 \
            model.data.train_ds.global_batch_size=4 \
            model.data.train_ds.micro_batch_size=2 \
            model.data.validation_ds.global_batch_size=4 \
            model.data.validation_ds.micro_batch_size=2 \
            model.data.train_ds.task_name=rte \
            model.data.train_ds.file_path=/home/TestData/nlp/megatron_t5/data/train_ci.tsv \
            model.data.validation_ds.task_name=xnli \
            model.data.validation_ds.file_path=/home/TestData/nlp/megatron_t5/data/xnli_dev_ci.tsv \
            model.data.test_ds.global_batch_size=4 \
            model.data.test_ds.micro_batch_size=2 \
            model.data.test_ds.task_name=xnli \
            model.data.test_ds.file_path=/home/TestData/nlp/megatron_t5/data/xnli_dev_ci.tsv \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_xnli_results"
          }
        }
      }
    }
    stage('L2: TTS Fast dev runs 1') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      parallel {
        // TODO(Oktai15): update it in 1.8.0 version
        stage('Tacotron 2') {
          steps {
            sh 'python examples/tts/tacotron2.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=null \
            model.train_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.decoder.decoder_rnn_dim=256 \
            model.decoder.attention_rnn_dim=1024 \
            model.decoder.prenet_dim=128 \
            model.postnet.postnet_n_convolutions=3 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        stage('WaveGlow') {
          steps {
            sh 'python examples/tts/waveglow.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=null \
            model.train_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.waveglow.n_flows=4 \
            model.waveglow.n_wn_layers=2 \
            model.waveglow.n_wn_channels=32 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        stage('FastPitch') {
          steps {
            sh 'python examples/tts/fastpitch.py \
            --config-name fastpitch_align_v1.05 \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/beta_priors \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=null \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=1 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=1 \
            model.symbols_embedding_dim=64 \
            model.input_fft.d_inner=384 \
            model.input_fft.n_layer=2 \
            model.output_fft.d_inner=384 \
            model.output_fft.n_layer=2 \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('Mixer-TTS') {
          steps {
            sh 'python examples/tts/mixer_tts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/sup_data \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=null \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=1 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=1 \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('Hifigan') {
          steps {
            sh 'python examples/tts/hifigan.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 +trainer.max_epochs=1 \
            trainer.strategy=null \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=1 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=1 \
            model.generator.upsample_initial_channel=64 \
            +model.debug=true \
            ~trainer.check_val_every_n_epoch'
          }
        }
      }
    }

    stage('L??: Speech Checkpoints tests') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/speech_to_text_eval.py \
            pretrained_name=QuartzNet15x5Base-En  \
            dataset_manifest=/home/TestData/librispeech/librivox-dev-other.json \
            batch_size=64 \
            tolerance=0.1012'
        sh 'rm -f examples/asr/evaluation_transcripts.json'
      }
    }
  }

  post {
    always {
      sh 'chmod -R 777 .'
      cleanWs()
    }
  }
}
