import hydra
from nemo_helper.speaker_recognition import SpeakerRecognition
from omegaconf import DictConfig
@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    # TODO add more code
    
    # cfg.sr is configuration for speaker recognition 
    speaker_recognition = SpeakerRecognition.from_config(cfg.sr)
    results = speaker_recognition.batch_infer([cfg.target_audio_file])
    print(results)

if __name__ == "__main__":
    main()
