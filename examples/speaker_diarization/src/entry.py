import logging
import hydra
import os
from omegaconf import OmegaConf, DictConfig
from nemo_helper.speaker_diarization import SpeakerDiarizer

@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    validate_config(cfg)
    # cfg.sd is configuration for speaker diarization
    speaker_diarizer = SpeakerDiarizer.from_config(cfg.sd)
    
    timestamps = speaker_diarizer.run_diarize(cfg.target_audio_file)
    #=> [('speaker_1', (0.0, 4.375)), ('speaker_0', (4.375, 6.360499858856201))]

    print(timestamps)

def validate_config(cfg):
    if OmegaConf.is_missing(cfg=cfg, key='target_audio_file'):
        logging.warning("USAGE:\npython src/entry.py target_audio_file=/app/myaudiofile.wav")
        logging.warning("maybe this program will fail")

    vad_model_path = cfg.sd.diarizer.vad.model_path
    if '/' in vad_model_path and not os.path.exists(vad_model_path):
        logging.warning(f"cfg.sd.diarizer.vad.model_path = {vad_model_path}")
        logging.warning("But file not found")
        logging.warning("check your configuration. maybe this program will fail...")
        logging.warning("Update conf/sd/sd_conf.yaml")
        logging.warning("or Use Command-line Option.")
        logging.warning("EXAMPLE: python src/entry.py target_audio_file=/app/myaudiofile.wav sd.diarizer.vad.model_path=/app/vad.onnx sd.diarizer.speaker_embeddings.model_path=/app/tita.onnx")
    
    sr_model_path = cfg.sd.diarizer.speaker_embeddings.model_path
    if '/' in sr_model_path and not os.path.exists(sr_model_path):
        logging.warning(f"cfg.sd.diarizer.vad.model_path = {sr_model_path}")
        logging.warning("But file not found")
        logging.warning("check your configuration. maybe this program will fail")
        logging.warning("Update conf/sd/sd_conf.yaml")
        logging.warning("or Use Command-line Option.")
        logging.warning("EXAMPLE: python src/entry.py target_audio_file=/app/myaudiofile.wav sd.diarizer.vad.model_path=/app/vad.onnx sd.diarizer.speaker_embeddings.model_path=/app/tita.onnx")

if __name__ == "__main__":
    main()
