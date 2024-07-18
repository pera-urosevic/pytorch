import torch
import torchaudio
from config import device

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H


def predict(audio_file):
    model = bundle.get_model().to(device)

    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        print(f'Got sample rate of {sample_rate}, converting to {bundle.sample_rate}')
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        result, _ = model(waveform)
    prediction = result[0]

    return prediction
