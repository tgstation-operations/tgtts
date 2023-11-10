import scipy.signal as signal
import torch
from TTS.api import TTS
import os
import io
import json
import gc
import random
import numpy as np
import ffmpeg
from typing import *
from modules import models
from modules.utils import load_audio
from flask import Flask, request, send_file, abort, make_response
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence
from fairseq import checkpoint_utils
from fairseq.models.hubert.hubert import HubertModel
from modules.shared import ROOT_DIR, device, is_half
import requests
import librosa
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
import numpy as np
import time
import random
import statistics
import string
import torch
import onnxruntime as ort
import numpy as np
from typing import Any
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Any, Protocol
from fairseq import checkpoint_utils
from torch import device
import torchcrepe
import onnxcrepe
from onnxcrepe.session import CrepeInferenceSession
import io
import librosa
from pydub import AudioSegment
import json
from pyceps.core import rceps, cepsf0, cepsenv
from pyceps.utils import upsample



config = VitsConfig()
config.load_json("E:/model_output_2/config.json")
vits = Vits.init_from_config(config)

vits.load_onnx("tgstation_onnx_fp16.onnx", True)
### READ ME
# How to use this version after doing normal TTS setup.
# 1. Clone https://github.com/ddPn08/rvc-webui.git somewhere, and pip install the ./requirements/main.txt requirements file.
# 2. This will downgrade Librosa, which doesn't matter, TTS still runs properly, ignore it.
# 3. Put this .py file and the two .wav files next to it in the base of the rvc-webui repository you cloned.
# 4. Place your .pth files and .json files in the ./models/checkpoints folder in the cloned repository.
# 5. Download hubert_base.pt from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main and place it in the ./models/embeddings folder in the cloned repository.
# 6. Boot this instead of tts.py.
# "What does this actually do?"
# This puts the Retrieval-Voice-Conversion model between the TTS and the actual webserver, allowing for improved speaker accuracy and improved audio quality.
### READ ME
# UPDATE ME FOR YOUR OWN MODEL FILES YOU TRAIN
vc_models = {
	pipeline_1: "./speakers_tgstation_1.json",
	pipeline_2: "./speakers_tgstation_2.json",
	pipeline_3: "./speakers_tgstation_3.json",
}
loaded_models = []
voice_lookup = {}
for model in vc_models.keys():
	voice_lookup[model] = json.load(open(vc_models[model], "r"))
	loaded_models.append(model)

voice_name_mapping = {}
use_voice_name_mapping = True
with open("./tts_voices_mapping.json", "r") as file:
	voice_name_mapping = json.load(file)
	if len(voice_name_mapping) == 0:
		use_voice_name_mapping = False
				
voice_id_mapping = {}
with open("./tts_voice_to_id.json", "r") as file:
	voice_id_mapping = json.load(file)

app = Flask(__name__)
letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
random_factor = 0.35
os.makedirs('samples', exist_ok=True)
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}

request_count = 0

@app.route("/generate-tts")
def text_to_speech():
	global request_count
	text = request.json.get("text", "")
	voice = request.json.get("voice", "")
	pitch_adjustment = request.json.get("pitch", "")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]

	result = None
	with io.BytesIO() as data_bytes:
		text_inputs = np.asarray(
			vits.tokenizer.text_to_ids(text, language="en"),
			dtype=np.int64,
		)[None, :]
		tts_audio = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping[voice])
		save_wav(wav=tts_audio[0], path=data_bytes, sample_rate=22050)
		speaker_id = "NO SPEAKER"
		model_to_use = None	
		found_model = False
		for model in voice_lookup.keys():
			speaker_list = voice_lookup[model]
			for speaker in speaker_list.keys():
				if voice == speaker:
					speaker_id = speaker_list[speaker]
					found_model = True
					break
			if found_model:
				model_to_use = loaded_models[list(voice_lookup.keys()).index(model)]
				break
		if speaker_id == "NO SPEAKER" or model_to_use == None:
			abort(500)
		audio, _ = librosa.load(io.BytesIO(data_bytes.getvalue()), sr=16000)

		processed_audio = model_to_use.exec(speaker_id, np.expand_dims(audio, axis=0), int(pitch_adjustment))
		audio = AudioSegment(
			processed_audio,
			frame_rate=40000,
			sample_width=2,
			channels=1,
		)
		audio.export(
			data_bytes,
			format="wav",
		)
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	request_count += 1
	return result

@app.route("/generate-tts-blips")
def text_to_speech_blips():
	global request_count
	text = request.json.get("text", "").upper()
	voice = request.json.get("voice", "")
	pitch_adjustment = request.json.get("pitch", "")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]

	result = None
	with io.BytesIO() as data_bytes:
		with torch.no_grad():
			result_sound = AudioSegment.empty()
			if not os.path.exists('samples/' + voice):
				os.makedirs('samples/' + voice, exist_ok=True)
				for i, value in enumerate(letters_to_use):
					text_inputs = np.asarray(
						vits.tokenizer.text_to_ids(value + ".", language="en"),
						dtype=np.int64,
					)[None, :]
					tts_audio = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping[voice])
					save_wav(wav=tts_audio[0], path="samples/" + voice + "/" + value + ".wav", sample_rate=22050)
					sound = AudioSegment.from_file("samples/" + voice + "/" + value + ".wav", format="wav")
					silenced_word = strip_silence(sound)
					silenced_word.export("samples/" + voice + "/" + value + ".wav", format='wav')
			speaker_id = "NO SPEAKER"
			model_to_use = None
			found_model = False
			for model in voice_lookup.keys():
				speaker_list = voice_lookup[model]
				for speaker in speaker_list.keys():
					if voice == speaker:
						speaker_id = speaker_list[speaker]
						found_model = True
						break
				if found_model:
					model_to_use = loaded_models[list(voice_lookup.keys()).index(model)]
					break
			if speaker_id == "NO SPEAKER" or model_to_use == None:
				abort(500)
			for i, letter in enumerate(text):
				if not letter.isalpha() or letter.isnumeric() or letter == " ":
					continue
				if letter == ' ':
					new_sound = letter_sound._spawn(b'\x00' * (40000 // 3), overrides={'frame_rate': 40000})
					new_sound = new_sound.set_frame_rate(40000)
					result_sound += new_sound
				else:
					if not i % 2 == 0:
						continue # Skip every other letter
					if not os.path.isfile("samples/" + voice + "/" + letter + ".wav"):
						continue
					if not os.path.isdir("samples/" + voice + "/pitch_" + pitch_adjustment):
						os.mkdir("samples/" + voice + "/pitch_" + pitch_adjustment)
					if not os.path.isfile("samples/" + voice + "/pitch_" + pitch_adjustment + "/" + letter + ".wav"):
						audio, _ = librosa.load("samples/" + voice + "/" + letter + ".wav", 16000)

						processed_audio = model_to_use.exec(speaker_id, np.expand_dims(audio, axis=0), int(pitch_adjustment))
						output_sound = AudioSegment(
							processed_audio,
							frame_rate=40000,
							sample_width=2,
							channels=1,
						)
						output_sound.export("samples/" + voice + "/pitch_" + pitch_adjustment + "/" + letter + ".wav", format="wav")
					letter_sound = AudioSegment.from_file("samples/" + voice + "/pitch_" + pitch_adjustment + "/" + letter + ".wav")

					raw = letter_sound.raw_data[5000:-5000]
					octaves = 1 + random.random() * random_factor
					frame_rate = int(letter_sound.frame_rate * (2.0 ** octaves))

					new_sound = letter_sound._spawn(raw, overrides={'frame_rate': frame_rate})
					new_sound = new_sound.set_frame_rate(40000)
				result_sound = new_sound if result_sound is None else result_sound + new_sound
			result_sound.export(data_bytes, format='wav')
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	request_count += 1
	return result

@app.route("/tts-voices")
def voices_list():
	if use_voice_name_mapping:
		data = list(voice_name_mapping.values())
		data.sort()
		return json.dumps(data)
	else:
		return json.dumps(tts.voices)

@app.route("/health-check")
def tts_health_check():
	gc.collect()
	if request_count > 2048:
		return f"EXPIRED: {request_count}", 500
	return f"OK: {request_count}", 200

@app.route("/pitch-available")
def pitch_available():
	return make_response("Pitch available", 200)

if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="0.0.0.0", port=5003, threads=4, backlog=8, connection_limit=24, channel_timeout=10)
