import streamlit as st

import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC

from libs.dummy import outputs as dummy_outputs
from libs.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
from libs.utils import (
    set_session_state,
    get_session_state,
    local_css,
    remote_css,
    plot_result
)

import meta


class SpeechToText:

    def __init__(
            self,
            ctc_model_name="m3hrdadfi/wav2vec2-large-xlsr-persian-v3",
            cf_model_name="m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition",
            ctc_model_type="wav2vec",
            cf_model_type="wav2vec",
            device=0,
            channels=1,
            subtype="PCM_24",
    ):
        self.debug = False
        self.dummy_outputs = dummy_outputs

        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctc_model_name = ctc_model_name
        self.cf_model_name = cf_model_name
        self.ctc_model_type = ctc_model_type
        self.cf_model_type = cf_model_type

        self.device = device
        self.device_info = sd.query_devices(device, 'input')
        self.samplerate = int(self.device_info['default_samplerate'])
        self.channels = channels
        self.subtype = subtype

        self.cf_feature_extractor = None
        self.cf_config = None
        self.cf_samplerate = None
        self.cf_model = None

        self.ctc_processor = None
        self.ctc_samplerate = None
        self.ctc_model = None

    def recording(self, duration_in_seconds=10):
        recording = sd.rec(
            frames=int((duration_in_seconds + 0.5) * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            blocking=True,
        )
        sd.wait()
        return recording

    def load_cf(self):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cf_model_name)
        config = AutoConfig.from_pretrained(self.cf_model_name)

        if self.cf_model_type == "wav2vec":
            model = Wav2Vec2ForSpeechClassification.from_pretrained(self.cf_model_name).to(self.torch_device)
        elif self.cf_model_type == "hubert":
            model = HubertForSpeechClassification.from_pretrained(self.cf_model_name).to(self.torch_device)
        else:
            model = Wav2Vec2ForSpeechClassification.from_pretrained(self.cf_model_name).to(self.torch_device)

        self.cf_feature_extractor = feature_extractor
        self.cf_config = config
        self.cf_samplerate = feature_extractor.sampling_rate
        self.cf_model = model

    def load_ctc(self):
        processor = Wav2Vec2Processor.from_pretrained(self.ctc_model_name)

        if self.ctc_model_type == "wav2vec":
            model = Wav2Vec2ForCTC.from_pretrained(self.ctc_model_name).to(self.torch_device)
        elif self.ctc_model_type == "hubert":
            model = HubertForCTC.from_pretrained(self.ctc_model_name).to(self.torch_device)
        else:
            model = Wav2Vec2ForCTC.from_pretrained(self.ctc_model_name).to(self.torch_device)

        self.ctc_processor = processor
        self.ctc_samplerate = processor.feature_extractor.sampling_rate
        self.ctc_model = model

    def load(self):
        if not self.debug:
            self.load_cf()
            self.load_ctc()

    def _speech_file_to_array_fn(self, path, samplerate):
        speech_array, sr = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sr, samplerate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def predict_cf(self, path):
        speech = self._speech_file_to_array_fn(path, self.cf_samplerate)
        inputs = self.cf_feature_extractor(speech, sampling_rate=self.cf_samplerate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.torch_device) for key in inputs}

        with torch.no_grad():
            logits = self.cf_model(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [
            {
                "label": self.cf_model.config.id2label[i],
                "score": score
            } for i, score in enumerate(scores)
        ]
        return outputs

    def predict_ctc(self, path):
        speech = self._speech_file_to_array_fn(path, self.ctc_samplerate)
        inputs = self.ctc_processor(speech, sampling_rate=self.ctc_samplerate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.torch_device) for key in inputs}

        with torch.no_grad():
            logits = self.ctc_model(**inputs).logits

        return self.ctc_processor.batch_decode(torch.argmax(logits, dim=-1))

    def predict(self, path):
        if self.debug:
            return self.dummy_outputs

        cf = self.predict_cf(path)
        ctc = self.predict_ctc(path)

        return {
            "ctc": ctc[0] if len(ctc) > 0 else ctc,
            "cf": cf
        }


@st.cache(allow_output_mutation=True)
def load_tts():
    tts = SpeechToText(
        ctc_model_name="m3hrdadfi/wav2vec2-large-xlsr-persian-v3",
        # cf_model_name="m3hrdadfi/hubert-base-persian-speech-emotion-recognition",
        cf_model_name="m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition",
        ctc_model_type="wav2vec",
        # cf_model_type="hubert",
        cf_model_type="wav2vec",
    )
    tts.load()
    return tts


def main():
    st.set_page_config(
        page_title="Speech To Text (Persian)",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    remote_css("https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font/dist/font-face.css")
    set_session_state("_is_recording", False)
    local_css("assets/style.css")
    st.write(f"DEVICES: {sd.query_devices()}")

    tts = load_tts()


    col1, col2 = st.beta_columns([5, 7])
    with col2:
        st.markdown('<div class="mt"></div>', unsafe_allow_html=True)
        audio_player = st.empty()
        speech_text = st.empty()

    with col1:
        st.markdown(meta.INFO, unsafe_allow_html=True)
        duration = st.slider('Choose your recording duration (seconds)', 5, 20, 5)
        recorder_btn = st.button("Recording")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    info = st.empty()

    if recorder_btn:
        if not get_session_state("_is_recording"):
            set_session_state("_is_recording", True)

            info.info(f"Recording for {duration} seconds ...")
            np_audio = tts.recording(duration_in_seconds=duration)
            if len(np_audio) > 0:
                filename = tempfile.mktemp(prefix='tmp_sf_', suffix='.wav', dir='')
                with sf.SoundFile(
                        filename,
                        mode='x',
                        samplerate=tts.samplerate,
                        channels=tts.channels,
                        subtype=tts.subtype
                ) as tmp_audio:
                    tmp_audio.write(np_audio)

                audio_player.audio(filename)
                speech_text.info(f"Converting speech to text ...")
                result = tts.predict(filename)
                speech_text.markdown(
                    f'<p class="ctc-box rtl"><strong>متن: </strong>{result["ctc"]}</p>',
                    unsafe_allow_html=True
                )

                info.info(f"Recognizing emotion ...")
                plot_result(result["cf"])

                if os.path.exists(filename):
                    os.remove(filename)

                info.empty()
                set_session_state("_is_recording", False)


if __name__ == '__main__':
    main()
