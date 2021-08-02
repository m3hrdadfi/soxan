# Soxan

> در زبان پارسی به نام سخن


This repository consists of models, scripts, and notebooks that help you to use all the benefits of Wav2Vec 2.0 in your
research. In the following, I'll show you how to train speech tasks in your dataset and how to use the pretrained
models.

## How to train

I'm just at the beginning of all the possible speech tasks. To start, we continue the training script with the speech
emotion recognition problem.

### Training - Notebook

| Task                                     | Notebook                                                                                                                                                                                                            |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Speech Emotion Recognition (Wav2Vec 2.0) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb) |
| Speech Emotion Recognition (Hubert)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_HuBERT.ipynb)   |
| Audio Classification (Wav2Vec 2.0)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb)             |

### Training - CMD

```bash
python3 run_wav2vec_clf.py \
    --pooling_mode="mean" \
    --model_name_or_path="lighteternal/wav2vec2-large-xlsr-53-greek" \
    --model_mode="wav2vec2" \ # or you can use hubert
    --output_dir=/path/to/output \
    --cache_dir=/path/to/cache/ \
    --train_file=/path/to/train.csv \
    --validation_file=/path/to/dev.csv \
    --test_file=/path/to/test.csv \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=5.0 \
    --evaluation_strategy="steps"\
    --save_steps=100 \
    --eval_steps=100 \
    --logging_steps=100 \
    --save_total_limit=2 \
    --do_eval \
    --do_train \
    --fp16 \
    --freeze_feature_extractor
```

### Prediction

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification

model_name_or_path = "path/to/your-pretrained-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

# for wav2vec
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# for hubert
model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs


path = "/path/to/disgust.wav"
outputs = predict(path, sampling_rate)    
```

Output:

```bash
[
    {'Emotion': 'anger', 'Score': '0.0%'},
    {'Emotion': 'disgust', 'Score': '99.2%'},
    {'Emotion': 'fear', 'Score': '0.1%'},
    {'Emotion': 'happiness', 'Score': '0.3%'},
    {'Emotion': 'sadness', 'Score': '0.5%'}
]
```


## Demos

| Demo                                                     | Link                                                                                                               |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Speech To Text With Emotion Recognition (Persian) - soon | [huggingface.co/spaces/m3hrdadfi/speech-text-emotion](https://huggingface.co/spaces/m3hrdadfi/speech-text-emotion) |


## Models

| Dataset                                                                                                                      | Model                                                                                                                                           |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [ShEMO: a large-scale validated database for Persian speech emotion detection](https://github.com/mansourehk/ShEMO)          | [m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition) |
| [ShEMO: a large-scale validated database for Persian speech emotion detection](https://github.com/mansourehk/ShEMO)          | [m3hrdadfi/hubert-base-persian-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/hubert-base-persian-speech-emotion-recognition)     |
| [ShEMO: a large-scale validated database for Persian speech emotion detection](https://github.com/mansourehk/ShEMO)          | [m3hrdadfi/hubert-base-persian-speech-gender-recognition](https://huggingface.co/m3hrdadfi/hubert-base-persian-speech-gender-recognition)       |
| [Speech Emotion Recognition (Greek) (AESDD)](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)              | [m3hrdadfi/hubert-large-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/hubert-large-greek-speech-emotion-recognition)       |
| [Speech Emotion Recognition (Greek) (AESDD)](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)              | [m3hrdadfi/hubert-base-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/hubert-base-greek-speech-emotion-recognition)         |
| [Speech Emotion Recognition (Greek) (AESDD)](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)              | [m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition)     |
| [Eating Sound Collection](https://www.kaggle.com/mashijie/eating-sound-collection)                                           | [m3hrdadfi/wav2vec2-base-100k-eating-sound-collection](https://huggingface.co/m3hrdadfi/wav2vec2-base-100k-eating-sound-collection)             |
| [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) | [m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres](https://huggingface.co/m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres)                       |