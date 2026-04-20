import argparse
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
import os
from lhotse.utils import fastcopy
import numpy as np
import logging
import math
import io
import soundfile as sf
from lhotse import Recording, SupervisionSegment, CutSet, MonoCut
import numpy as np
import re

# num=0
# src_manifest=/export/fs06/ahussei6/nvidia/data/cvss/es-US_en-US/nemo_prepared_dev/sharded_manifests/manifest_${num}.jsonl
# src_tar=/export/fs06/ahussei6/nvidia/data/cvss/es-US_en-US/nemo_prepared_dev/audio_${num}.tar

# tgt_manifest=/export/fs06/ahussei6/nvidia/results/cvss/es-US_en-US/dev/sharded_manifests/manifest_${num}.jsonl
# tgt_tar=/export/fs06/ahussei6/nvidia/results/cvss/es-US_en-US/dev/audio_${num}.tar


# python /export/fs06/ahussei6/nvidia/github/NeMo/examples/speechlm2/combine_info_concat_v.py --src_tar $src_tar --tgt_tar $tgt_tar --src_manifest $src_manifest --tgt_manifest $tgt_manifest --output_dir debug

def parse_args():
    parser = argparse.ArgumentParser(description="Load data from tar and manifest using lhotse.")
    parser.add_argument('--src_tar', type=str, required=True, help='Path to the tar file containing the data.')
    parser.add_argument('--src_manifest', type=str, required=True, help='Path to the manifest file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--tgt_tar', type=str, required=True, help='Path to the tar file containing the target data.')
    parser.add_argument('--tgt_manifest', type=str, required=True, help='Path to the manifest file containing the target data.')
    parser.add_argument('--min_duration', type=float, default=1.0, help='Minimum duration in seconds.')
    parser.add_argument('--max_duration', type=float, default=30.0, help='Maximum duration in seconds.')
    return parser.parse_args()



def clean_txt(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_extension_from_segment_id(segment):
    return fastcopy(segment, id=os.path.splitext(segment.id)[0])

def load_nemo_tarred_from_dir(manifest_path: str, tar_paths: str) -> CutSet:

    # Initialize iterator
    iterator = LazyNeMoTarredIterator(
                        manifest_path=manifest_path,
                        tar_paths=tar_paths,
                        allow_skipme=False,
                        shuffle_shards=False,
                    )
    return CutSet.from_cuts(iterator)

def recording_from_numpy(waveform: np.ndarray, sr: int, rec_id: str = "rec_in_memory"):
    """
    Create a Lhotse Recording from a NumPy waveform entirely in memory.
    waveform shape: (1, num_samples) or (num_samples,) → mono.
    """
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # (1, num_samples)

    # Encode the waveform into WAV bytes in memory
    buffer = io.BytesIO()
    sf.write(buffer, waveform.T, sr, format="WAV")
    wav_bytes = buffer.getvalue()

    # Use Lhotse's from_bytes()
    recording = Recording.from_bytes(data=wav_bytes, recording_id=rec_id)
    return recording

def pad_right(arr: np.ndarray, T: int, value: float = 0.0) -> np.ndarray:

    assert arr.ndim == 2, "Array must be 2D"
    C, N = arr.shape
    if N >= T:
        return arr
    pad = np.full((C, (T - N)), value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)

def combine_cuts(src_cuts, tgt_cuts):
    cuts = []
    for src_cut in src_cuts:
        tgt_cut = tgt_cuts[src_cut.id]
        supervisions = []
        if src_cut.sampling_rate != tgt_cut.sampling_rate:
            src_cut = src_cut.resample(tgt_cut.sampling_rate)
        total_duration = src_cut.duration + tgt_cut.duration
        src_start, src_dur = src_cut.supervisions[0].start, src_cut.supervisions[0].duration
        tgt_start, tgt_dur = tgt_cut.supervisions[0].start, tgt_cut.supervisions[0].duration
        src_cut = src_cut.pad(duration=total_duration, direction="right")
        tgt_cut = tgt_cut.pad(duration=total_duration, direction="left")
        src_wav = src_cut.load_audio()
        tgt_wav = tgt_cut.load_audio()
        src_recording = recording_from_numpy(src_wav, tgt_cut.sampling_rate, rec_id=f"{src_cut.id}")
        tgt_recording = recording_from_numpy(tgt_wav, tgt_cut.sampling_rate, rec_id=f"{src_cut.id}")

        supervisions.append(
            SupervisionSegment(
                id=f"{src_cut.supervisions[0].id}_src",
                recording_id=f"{src_cut.id}",
                start=src_start,
                duration=src_dur,
                channel=0,
                text=src_cut.supervisions[0].text,
                speaker="user",
            )
        )
        supervisions.append(
            SupervisionSegment(
                id=f"{tgt_cut.supervisions[0].id}_tgt",
                recording_id=f"{src_cut.id}",
                start=src_start+ src_dur + tgt_start,
                duration=tgt_dur,
                channel=0,
                text=tgt_cut.supervisions[0].text,
                speaker="agent",
            )
        )

        new_cut = MonoCut(id=src_cut.id, start=src_start, duration=total_duration, channel=0, recording=src_recording, supervisions=supervisions)
        new_cut.target_audio = tgt_recording
        cuts.append(new_cut)
    return CutSet.from_cuts(cuts)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    # Placeholder for loading data using lhotse
    logging.info(f"Tar file: {args.src_tar}")
    logging.info(f"Manifest file: {args.src_manifest}")
    logging.info(f"Target tar file: {args.tgt_tar}")
    logging.info(f"Target manifest file: {args.tgt_manifest}")

    
    # loading text alignment
    
    src_cuts = load_nemo_tarred_from_dir(args.src_manifest, args.src_tar)
    shard_id = src_cuts[0].custom['shard_id']
    tgt_cuts = load_nemo_tarred_from_dir(args.tgt_manifest, args.tgt_tar)
    # remove the extension from the cut ids
    src_cuts = src_cuts.modify_ids(lambda id: os.path.splitext(id)[0])
    tgt_cuts = tgt_cuts.modify_ids(lambda id: os.path.splitext(id)[0])
    
    # filter out the short and long cuts
    src_cuts = src_cuts.filter(lambda x: args.min_duration <= x.duration <= args.max_duration)
    tgt_cuts = tgt_cuts.filter(lambda x: 1 <= x.duration <= args.max_duration)
   
    # remove the extension from the supervision ids
    src_cuts = src_cuts.map_supervisions(remove_extension_from_segment_id)
    tgt_cuts = tgt_cuts.map_supervisions(remove_extension_from_segment_id)
    
    # trim the cuts to the alignments
    src_cuts = src_cuts.trim_to_supervisions(keep_overlapping=False)
    tgt_cuts = tgt_cuts.trim_to_supervisions(keep_overlapping=False)
    # make sure the cuts have same ids
    
    common_ids = set(src_cuts.ids) & set(tgt_cuts.ids)
    src_cuts = src_cuts.filter(lambda cut: cut.id in common_ids)
    tgt_cuts = tgt_cuts.filter(lambda cut: cut.id in common_ids)

    src_cuts = src_cuts.sort_like(tgt_cuts)
    # adding source trajectory
    tgt_cuts_dict = {c.id: c for c in tgt_cuts} 
  
    new_cuts  = combine_cuts(src_cuts, tgt_cuts_dict)
    
    new_cuts.to_shar(
            args.output_dir,
            shard_size=len(new_cuts),
            shard_offset=shard_id,
            fields={
                'recording': 'wav',
                'target_audio': 'wav',
            }
    )
    # loaded_cut = CutSet.from_shar(fields={ "cuts": ["test_combine/cuts.003868.jsonl.gz"],"recording": ["test_combine/recording.003868.tar"],"target_audio": ["test_combine/target_audio.003868.tar"],})
        # loaded_cut = loaded_cut.map(set_target_duration)
                                    
# cuts[1].trim_to_alignments(type="word",
#                                 max_pause=0.5,
#                                 keep_all_channels=False,
#                                 max_segment_duration=5.0,
#                                 get_all_segments=False
#                             )