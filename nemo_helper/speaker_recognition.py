import logging
import io
import json
import numpy as np
import torchaudio
import torch
import onnxruntime
from omegaconf import DictConfig
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

def to_numpy(tensor_obj: torch.Tensor):
    return tensor_obj.detach().numpy()
 
class SpeakerRecognition:
    def __init__(
            self, 
            enroll_embs_npy_file,
            enroll_truelabels_npy_file,
            enroll_id2label_file,
            speaker_key2name_file,
            preprocessor,
            onnx_filename
    ):
        self.preprocessor = preprocessor
        self.enroll_embs_npy_file = enroll_embs_npy_file
        self.enroll_truelabels_npy_file = enroll_truelabels_npy_file
        self.enroll_id2label_file = enroll_id2label_file
        self.speaker_key2name_file = speaker_key2name_file
        self.onnx_filename = onnx_filename
    
    @staticmethod
    def from_config(cfg):
        assert type(cfg) == DictConfig
        preprocessor = EncDecSpeakerLabelModel.from_config_dict(cfg.preprocessor)
        return SpeakerRecognition(
            enroll_embs_npy_file=cfg.speaker_recognition.enroll_embs_npy_file, 
            enroll_truelabels_npy_file=cfg.speaker_recognition.enroll_truelabels_npy_file,
            enroll_id2label_file=cfg.speaker_recognition.enroll_id2label_file,
            speaker_key2name_file=cfg.speaker_recognition.speaker_key2name_file,
            preprocessor=preprocessor,
            onnx_filename = cfg.speaker_recognition.sr_onnx_filename
        )

    def batch_infer(self, files=None, waveforms=None, threshold=0.7):
        """Usage:
            results = spk_reco.batch_infer(waveforms = [waveform1, waveform2])
            # or
            results = spk_reco.batch_infer(files = ['foo.wav', 'test.wav'])
            .
          Format of results: 
            [('spk_foo', 0.78), ('unknown', '0.4'), ...]
        """
        SAMPLE_RATE = 16000
        if files is not None:
            waveforms = [torchaudio.load(file)[0] for file in files]
        else:
            waveforms_ = []
            for waveform in waveforms:
                if type(waveform) == np.ndarray:
                    waveform = torch.tensor(waveform)
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                waveforms_.append(waveform)
            waveforms = waveforms_

        with open(self.speaker_key2name_file, 'r') as fp:
            speaker_key2name = json.load(fp)

        with open(self.enroll_id2label_file, 'r') as fp:
            enroll_id2label = json.load(fp)
        
        enroll_embs = np.load(self.enroll_embs_npy_file)
        enroll_truelabels = np.load(self.enroll_truelabels_npy_file)
        session = onnxruntime.InferenceSession(self.onnx_filename)
        
        results = []
        for waveform in waveforms:
            waveform_size = waveform.squeeze().shape[0]
            if waveform_size/16000 > 100:
                WINDOW_SIZE = 30 * 16000
                logging.info(f"large waveform. original_waveform_size = {waveform_size}, WINDOW_SIZE = {WINDOW_SIZE}")
                new_waveform = waveform.squeeze()[:((waveform_size//WINDOW_SIZE)*WINDOW_SIZE)]
                new_waveforms = new_waveform.reshape(waveform_size//WINDOW_SIZE, WINDOW_SIZE)
                
                mx_score = -1.0
                mx_name = 'unknown'
                for new_waveform in new_waveforms:
                    logging.info(f"target's duration = {new_waveform.shape[0]/16000}")
                    shape_tensor = torch.tensor([new_waveform.shape[0]])
                    processed_signal, processed_signal_len = self.preprocessor(input_signal=new_waveform.unsqueeze(0), length=shape_tensor)
                    ort_inputs = { session.get_inputs()[0].name: to_numpy(processed_signal), session.get_inputs()[1].name: to_numpy(processed_signal_len) }
                    logging.info(f"START session.run(...)")
                    logits, embs = session.run(None, ort_inputs)
                    logging.info(f"OK session.run(...)")

                    logging.info("START SpeakerRecognition.find_matched_labels_and_scores(...)")
                    matched_labels, scores =  SpeakerRecognition.find_matched_labels_and_scores(enroll_id2label, enroll_embs, enroll_truelabels, embs)
                    logging.info("OK SpeakerRecognition.find_matched_labels_and_scores(...)")
                    if matched_labels is None:
                        continue

                    name = speaker_key2name[enroll_id2label[str(matched_labels[0])]]
                    score = scores[0][matched_labels[0]]
                    if score > mx_score:
                        mx_score = score
                        if score >= threshold:
                            mx_name = name
                results.append((mx_name, mx_score))
                logging.debug((mx_name, mx_score))
            else:
                shape_tensor = torch.tensor([waveform.squeeze().shape[0]])
                processed_signal, processed_signal_len = self.preprocessor(input_signal=waveform, length=shape_tensor)
                # processed_signal, processed_signal_len = self.featurizer(waveform, shape_tensor)
                ort_inputs = { session.get_inputs()[0].name: to_numpy(processed_signal), session.get_inputs()[1].name: to_numpy(processed_signal_len) }

                logging.info(f"target's duration = {waveform_size/16000}")
                logging.info(f"START session.run(...)")
                logits, embs = session.run(None, ort_inputs)
                logging.info(f"OK session.run(...)")
                embs = SpeakerRecognition.normalize_embs(embs)

                logging.info("START SpeakerRecognition.find_matched_labels_and_scores(...)")
                matched_labels, scores = SpeakerRecognition.find_matched_labels_and_scores(enroll_id2label, enroll_embs, enroll_truelabels, embs)
                logging.info("OK SpeakerRecognition.find_matched_labels_and_scores(...)")

                if matched_labels is None:
                    return 'unknown', 0.0
                name = speaker_key2name[enroll_id2label[str(matched_labels[0])]]
                score = scores[0][matched_labels[0]]
                if score < threshold:
                    name = 'unknown'
                results.append((name, score))
                logging.debug((name, score))
        return results


    def infer(self, file=None, waveform=None, threshold=0.6):
        """DEPRECATED
        """
        assert file is None or type(file) is str
        assert waveform is None or type(waveform) is torch.Tensor or type(waveform) is np.ndarray
        assert file is None or waveform is None
        assert file is not None or waveform is not None

        SAMPLE_RATE = 16000
        if file is not None:
            waveform, sr = torchaudio.load(file)
        else:
            if type(waveform) == np.ndarray:
                waveform = torch.tensor(waveform)
            sr = SAMPLE_RATE 
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
        assert len(waveform.shape) == 2
        assert sr == SAMPLE_RATE

        with open(self.speaker_key2name_file, 'r') as fp:
            speaker_key2name = json.load(fp)

        with open(self.enroll_id2label_file, 'r') as fp:
            enroll_id2label = json.load(fp)
        
        enroll_embs = np.load(self.enroll_embs_npy_file)
        enroll_truelabels = np.load(self.enroll_truelabels_npy_file)
        logging.debug(waveform.shape) 
        shape_tensor = torch.tensor([waveform.squeeze().shape[0]])
        processed_signal, processed_signal_len = self.preprocessor(input_signal=waveform, length=shape_tensor)
        session = onnxruntime.InferenceSession(self.onnx_filename)
        ort_inputs = { 
            session.get_inputs()[0].name: to_numpy(processed_signal), 
            session.get_inputs()[1].name: to_numpy(processed_signal_len) 
        }
        logits, embs = session.run(None, ort_inputs)
        embs = SpeakerRecognition.normalize_embs(embs)
        matched_labels, scores = SpeakerRecognition.find_matched_labels_and_scores(enroll_id2label, enroll_embs, enroll_truelabels, embs)

        if matched_labels is None:
            return 'unknown', 0.0
        name = speaker_key2name[enroll_id2label[str(matched_labels[0])]]
        score = scores[0][matched_labels[0]]

        if score < threshold:
            name = 'unknown'

        return name, score
    
    @staticmethod
    def enrollment(
        speaker_blob_tuples,
        cfg: DictConfig,
    ):
        """Enrollment and Save. Usage:
                from nemo_helper.speaker_recognition import SpeakerRecognition
                from omegaconf import OmegaConf
                
                cfg = OmegaConf.load('conf/sr/sr.yaml')
                speaker_blob_tuples = [(<some-uniq-string>, <speaker-name-string>, <16k-wav-format-blob>), (...), ...]
                # eg) [('1qw2e-23fa-2zd', 'james', b'..RIFF..a'), ('2sa-3fs-ac', 'james', b'..RIFF..bbd'), ('2asd-ad', 'foo', b'...'), ...] 

                SpeakerRecognition.enrollment(speaker_blob_tuples, cfg)
        """
        assert type(cfg) == DictConfig

        dst_enroll_embs_npy_file = cfg.speaker_recognition.enroll_embs_npy_file
        dst_enroll_truelabels_npy_file = cfg.speaker_recognition.enroll_truelabels_npy_file
        dst_enroll_id2label_file = cfg.speaker_recognition.enroll_id2label_file
        dst_speaker_key2name_file = cfg.speaker_recognition.speaker_key2name_file

        preprocessor = EncDecSpeakerLabelModel.from_config_dict(cfg.preprocessor)
        session = onnxruntime.InferenceSession(cfg.speaker_recognition.sr_onnx_filename)

        speaker_key2name = {speaker_key: speaker_name for speaker_key, speaker_name, _ in speaker_blob_tuples}
        with open(dst_speaker_key2name_file, "w") as fp:
            json.dump(speaker_key2name, fp)
            fp.write('\n')

        enroll_truelabels = [key for key, name, blob in speaker_blob_tuples]
        enroll_truelabels = np.asarray(enroll_truelabels)
        np.save(file=dst_enroll_truelabels_npy_file, arr=enroll_truelabels)

        enroll_id2label = { i: key for i, key in enumerate(enroll_truelabels) }
        with open(dst_enroll_id2label_file, "w") as fp:
            json.dump(enroll_id2label, fp)
            fp.write('\n')

        enroll_embs = []
        for key, name, blob in speaker_blob_tuples:
            bo = io.BytesIO(blob)
            waveform, sample_rate = torchaudio.load(bo)
            assert sample_rate == 16000
            processed_signal, processed_signal_len = preprocessor(input_signal=waveform, length=torch.tensor([waveform.squeeze().shape[0]]))
            ort_inputs = { session.get_inputs()[0].name: to_numpy(processed_signal), session.get_inputs()[1].name: to_numpy(processed_signal_len) }
            logit, emb = session.run(None, ort_inputs)
            enroll_embs.append(emb.squeeze())

        enroll_embs = np.asarray(enroll_embs)
        enroll_embs = SpeakerRecognition.normalize_embs(enroll_embs)
        np.save(file=dst_enroll_embs_npy_file, arr=enroll_embs)


    @staticmethod
    def normalize_embs(embs):
        return embs / (np.linalg.norm(embs, ord=2, axis=-1, keepdims=True))

    @staticmethod
    def find_matched_labels_and_scores(enroll_id2label, enroll_embs, enroll_truelabels, test_embs):
        reference_embs = []
        keyslist = list(enroll_id2label.values())
        for label_id in keyslist:
            indices = np.where(enroll_truelabels == label_id)
            embedding = (enroll_embs[indices].sum(axis=0).squeeze()) / len(indices)
            reference_embs.append(embedding)
        reference_embs = np.asarray(reference_embs)
        try:
            scores = np.matmul(test_embs, reference_embs.T)
        except ValueError as e:
            logging.error(e)
            logging.error(test_embs.shape)
            return None, -1

        matched_labels = scores.argmax(axis=-1)
        return matched_labels, scores

