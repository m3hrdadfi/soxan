# Soxan
> در زبان پارسی به نام سخن


This repository consists of models, scripts, and notebooks that help you to use all the benefits of Wav2Vec 2.0 in your research. In the following, I'll show you how to train speech tasks in your dataset and how to use my pretrained models.

## How to train

I'm just at the beginning of all the possible speech tasks. To start, we continue the training script with the speech emotion recognition problem.

### Speech Emotion Recognition (SER)

There are two ways to accomplish this task: using Google Colab notebook or the shell script.

#### Notebook

| Task                       	| Notebook                                                                                                                                                                                           	|
|----------------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| Speech Emotion Recognition 	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb) 	|

#### CMD

```bash
python3 run_wav2vec_clf.py \
    --pooling_mode="mean" \
    --model_name_or_path="lighteternal/wav2vec2-large-xlsr-53-greek" \
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
    --fp16
```


#### Models

| Dataset                                    	| Model                                                                                                                                       	|
|--------------------------------------------	|---------------------------------------------------------------------------------------------------------------------------------------------	|
| Speech Emotion Recognition (Greek) (AESDD) 	| [m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition) 	|
