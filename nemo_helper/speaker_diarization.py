import json
import os
import logging 

from .nemo_onnx.msdd_models_onnx import NeuralDiarizerOnnx

class SpeakerDiarizer:
    def __init__(self, sd_model):
        self.sd_model = sd_model

    def run_diarize(self, audio_file):
        """Diarize and return timestamps such as:
            [('speaker_1', (0.0, 4.375)), ('speaker_0', (4.375, 6.360499858856201))]
        """
        self.prepare_manifest(audio_file)        
        try:
            logging.info("START sd_model.diarize()")
            self.sd_model.diarize()
            logging.info(self.sd_model.refs)
            logging.info(self.sd_model.hyps)
            diar_res = self.sd_model.hyps[0]
            logging.info("OK sd_model.diarize()")
        except ValueError as e:
            logging.debug(e)
            return []
        annotation = diar_res[0][1]
        timestamps = [(list(annotation.get_labels(seg))[0], tuple(seg)) for seg in annotation.get_timeline().segments_list_]
        
        return timestamps
    
    def prepare_manifest(self, audio_file):
        meta = {
            'audio_filepath': audio_file, # an4_audio
            'offset': 0,
            'duration':None,
            'label': 'infer',
            'text': '-',
            'num_speakers': None, 
            'rttm_filepath': None, # an4_rttm
            'uem_filepath' : None
        }
        with open(self.sd_model._cfg.diarizer.manifest_filepath, 'w') as fp:
            json.dump(meta, fp)
            fp.write('\n')
        return True

    @staticmethod
    def from_config(cfg):
        ROOT = os.getcwd()
        data_dir = os.path.join(ROOT, 'data')
        os.makedirs(data_dir, exist_ok=True)

        output_dir = os.path.join(ROOT, 'sd_outputs')
        cfg.diarizer.manifest_filepath = 'data/input_manifest.json'
        cfg.diarizer.out_dir = output_dir #Directory to store intermediate files and prediction outputs
        os.makedirs(output_dir, exist_ok=True)

        # sd_model = ClusteringDiarizerOnnx(cfg=cfg)
        if cfg.diarizer.vad.external_vad_manifest is not None:
            sd_model.has_vad_model = False
        sd_model = NeuralDiarizerOnnx(cfg=cfg)
        speaker_diarizer = SpeakerDiarizer(sd_model=sd_model)

        return speaker_diarizer
