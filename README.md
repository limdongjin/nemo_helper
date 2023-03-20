# nemo_helper

## Description

Helper for NVIDIA NeMo. such as onnx support.

- nemo_helper/speaker_diarizer.py : you can use SpeakerDiarizer with speaker recognitiion onnx model file or VAD onnx file. 
- nemo_helper/speaker_recognition.py :you can use SpeakerRecognition with onnx file, enrolled information files. 

## Getting started

1. Install Nemo Toolkit
```bash
pip install nemo_toolkit[all]
```

2. Prepare your model files (such as, titanet_large.onnx, vad.nemo)

3. Install nemo_helper

4. Prepare Configuration. (reference: /examples/speaker_diarization/conf/)

5. Run 

```bash
.venv/bin/python3.9 examples/speaker_diarization/src/entry.py target_audio_file=/app/samples/foo2.wav
```
