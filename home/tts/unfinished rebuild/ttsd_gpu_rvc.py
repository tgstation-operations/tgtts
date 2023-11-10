from .common import *
from .tgpipeline import *

import torch
from TTS.api import TTS
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
import librosa
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
import numpy as np
import time
import statistics
import string
import onnxruntime as ort
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import device
import torchcrepe
import onnxcrepe
from onnxcrepe.session import CrepeInferenceSession
import io
import librosa
from pydub import AudioSegment
import json
from julius.resample import ResampleFrac
import threading



rvc_resample_function = ResampleFrac(22050, 16000) #compile a gpu kernal for a given resample fraction
simple_resample_function = ResampleFrac(22050, 40000) #compile a gpu kernal for a given resample fraction

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ttslock = threading.Lock()
ttstokenizerlock = threading.Lock()
juliuslock = threading.Lock()
rvclock = threading.Lock()

#systemd stuff.
watchdog_timer = None
try:
	import sd_notify
	watchdog = sd_notify.Notifier()
	if not watchdog.enabled(): 
		watchdog = None
except:
	watchdog = None
	
tts_errors = 0
rvc_errors = 0
last_request_time = time.time()
avg_request_time = 0
avg_tts_time = 0
avg_rvc_time = 0
avg_request_len = 0
avg_request_delay = 0
avg_request_rate = 0

timeofdeath = 0



with torch.inference_mode():
	print("Loading RVC models")
	rvc_1 = OnnxRVCInferencer()
	rvc_2 = OnnxRVCInferencer()
	rvc_3 = OnnxRVCInferencer()
	rvc_4 = OnnxRVCInferencer()
	rvc_1.loadModel("TGStation_Crepe_Model_Update_1_simple.onnx")
	rvc_2.loadModel("TGStation_Crepe_Model_Update_2_simple.onnx")
	rvc_3.loadModel("TGStation_Crepe_Model_Update_3_simple.onnx")
	rvc_4.loadModel("TGStation_Crepe_4_simple.onnx")
	print("Done loading RVC models")
	print("Loading contentvec encoder")
	#encoder = FairseqHubert()
	#encoder.loadModel("./models/embeddings/hubert_base.pt", "cuda:0")
	encoder_2 = FairseqHubertOnnx()
	encoder_2.loadModel("hubert_base_simple.onnx")
	print("Loaded")
	print("Loading CrepePitch")
	pitch_extract = CrepePitchExtractor()
	print("Loaded")
	print("Loading Pipelines")
	pipeline_1 = Pipeline(encoder_2, rvc_1, pitch_extract, 40000, "cuda:0", True)
	pipeline_2 = Pipeline(encoder_2, rvc_2, pitch_extract, 40000, "cuda:0", True)
	pipeline_3 = Pipeline(encoder_2, rvc_3, pitch_extract, 40000, "cuda:0", True)
	pipeline_4 = Pipeline(encoder_2, rvc_4, pitch_extract, 40000, "cuda:0", True)
	print("Loaded")
	config = VitsConfig()
	config.load_json("./tts_data/config.json")
	vits = Vits.init_from_config(config)
	#vits.load_checkpoint(config,  "E:/model_output_2/checkpoint_1157000.pth")

	#vits.export_onnx("tgstation_onnx.onnx")
	vits.load_onnx("tgstation_onnx.simplified.onnx", True)
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
	pipeline_4: "./speakers_tgstation_4.json",
}
loaded_models = []
voice_lookup = {}
for model in vc_models.keys():
	voice_lookup[model] = json.load(open(vc_models[model], "r"))
	loaded_models.append(model)

voice_name_mapping = {}
use_voice_name_mapping = True
with open("./tg_voices_mapping.json", "r") as file:
	voice_name_mapping = json.load(file)
	if len(voice_name_mapping) == 0:
		use_voice_name_mapping = False
				
voice_id_mapping = {}
with open("./tgstation_voice_to_id.json", "r") as file:
	voice_id_mapping = json.load(file)

app = Flask(__name__)
letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
random_factor = 0.35
sample_version = 'v3'
sample_dir = f"samples/{sample_version}/"
os.makedirs(sample_dir, exist_ok=True)
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}

request_count = 0

def tts_log(text):
	return
	with open('ttsd-gpu-runlog.txt', 'a') as f:
		f.write(f"{text}\n")


def maketts():
	return
	tts_log("maketts()")
	with torch.inference_mode():
		rtn = TTS(model_path="./tts_data/model_no_disc.pth", config_path="./tts_data/config.json", progress_bar=False, gpu=True)
		
		rtn.synthesizer.tts_model.half()
		rtn.synthesizer.tts_model.eval().share_memory()
		return rtn
		#return TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=True)


def mc_avg(old, new):
	if (old > 0):
		return (old*0.99) + (new*0.01)
	return new

def two_way_round(number, ndigits = 0):
	number = round(number, ndigits)
	return (f"%0.{ndigits}f") % number

def ping_watchdog():
	global request_count, last_request_time
	if watchdog:
		watchdog.status(f"count:{request_count}({tts_errors}/{rvc_errors}) len:{two_way_round(avg_request_len, 1)} time:{two_way_round(avg_request_time, 4)}s({two_way_round(avg_tts_time, 4)}/{two_way_round(avg_rvc_time, 4)}) rate:{two_way_round(avg_request_rate, 1)}/s last:{two_way_round(time.time() - last_request_time, 1)}s")
		if timeofdeath > 0 and time.time() > timeofdeath:
			watchdog.notify_error()
			return
		#if tts_errors > 5:
			#watchdog.notify_error()
		#	return
		#if (request_count > 10) and (time.time() > last_request_time+30):
		#	request_count += 1
		#	test_tts()
		#	last_request_time = time.time()

		with ttslock:
			watchdog.notify()
			schedule_watchdog()

def test_tts():
	tts_log("test_tts()")
	with ttslock:
		tts_log("test_tts():lock")

		with io.BytesIO() as data_bytes:
			with torch.inference_mode():
				tts_log("do test_tts()")
				with ttstokenizerlock:
					steptime = time.time()
					text_inputs = np.asarray(
						vits.tokenizer.text_to_ids("The quick brown fox jumps over the lazy dog", language="en"),
						dtype=np.int64, 
					)[None, :]
				tts_res = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping["maleadult01default"])
				
				#tts.tts_to_file(text="The quick brown fox jumps over the lazy dog.", speaker="maleadult01default", file_path=data_bytes)

def schedule_watchdog():
	global watchdog_timer
	if watchdog:
		watchdog_timer = threading.Timer(1, ping_watchdog)
		watchdog_timer.start()

def readyup():
	print("readyup()\n")
	tts_log("readyup()")
	if watchdog:
		tts_log("readyup(): watchdog")
		test_tts()
		gc_loop()
		tts_log("readyup(): getlock")
		with ttslock:
			tts_log("readyup(): gotlock, sending ready")
			watchdog.ready()
			tts_log("readyup(): ready sent")
			schedule_watchdog()
	else:
		gc_loop()
		
def gc_loop():
	threading.Timer(60, gc_loop)
	#gc.collect()
	#torch.cuda.empty_cache()
	

@app.route("/generate-tts")
def text_to_speech():
	global request_count, last_request_time, avg_request_time, avg_tts_time, avg_rvc_time, tts_errors, rvc_errors, avg_request_len, avg_request_rate, avg_request_delay
	request_count += 1
	tts_errors += 1
	starttime = time.time()
	
	text = request.json.get("text", "")
	voice = request.json.get("voice", "")
	pitch_adjustment = request.json.get("pitch", 0)
	
	avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
	avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
	avg_request_len = mc_avg(avg_request_len, len(text))
	
	print(f"\ntts start: {voice=}, {pitch_adjustment=}, {text=}")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]
	
	result = None
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
		tts_errors -= 1
		abort(500)
	with torch.inference_mode():
		with io.BytesIO() as data_bytes:
			tts_starttime = time.time()
			#with ttstokenizerlock:
			steptime = time.time()
			text_inputs = np.asarray(
				vits.tokenizer.text_to_ids(text, language="en"),
				dtype=np.int64, 
			)[None, :]
			print(f"tts.tokenize:{time.time()-steptime}")
			with ttslock:
				steptime = time.time()
				tts_res = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping[voice])
				print(f"tts.main:{time.time()-steptime}")
			
			avg_tts_time = mc_avg(avg_tts_time, time.time()-tts_starttime)
			rvc_starttime = time.time()
			rvc_errors += 1
			with juliuslock:
				steptime = time.time()
				audio = resample_function(torch.as_tensor(tts_res[0])).numpy() #resample to 16000 using precompiled gpu kernal
				#audio = librosa.resample(tts_res[0], orig_sr=22050, target_sr=16000)
				print(f"resample:{time.time()-steptime}")
				del tts_res
			
			steptime = time.time()
			processed_audio = model_to_use.exec(speaker_id, np.expand_dims(audio, axis=0), int(pitch_adjustment))
			print(f"rvc:{time.time()-steptime}")
			
			with rvclock:			
				final_audio = processed_audio.detach().cpu().numpy()
			del processed_audio
			#return 
			audio = AudioSegment(
				final_audio,
				frame_rate=40000,
				sample_width=2,
				channels=1,
			)

			audio.export(
				data_bytes,
				format="wav",
			)
			rvc_errors -= 1
			avg_rvc_time = mc_avg(avg_rvc_time, time.time()-rvc_starttime)
			result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")

	request_count += 1
	tts_errors -= 1
	last_request_time = time.time()
	avg_request_time = mc_avg(avg_request_time, last_request_time-starttime)
	print(f"total:{time.time()-starttime}")
	return result

@app.route("/generate-tts-blips")
def text_to_speech_blips():
	global request_count
	text = request.json.get("text", "").upper()
	voice = request.json.get("voice", "")
	pitch_adjustment = request.json.get("pitch", 0)
	print(f"\nblips start: {voice=}, {pitch_adjustment=}, {text=}")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]

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

	result = None
	with io.BytesIO() as data_bytes:
		with torch.inference_mode():
			result_sound = AudioSegment.empty()
			if not os.path.exists(f"{sample_dir}{voice}/{letters_to_use[-1]}.wav"):
				print(f"\nblips generate voice samples: {voice=}, {pitch_adjustment=}, {text=}")
				os.makedirs(sample_dir + voice, exist_ok=True)
				for i, value in enumerate(letters_to_use):
					#with ttstokenizerlock:
					text_inputs = np.asarray(
						vits.tokenizer.text_to_ids(value + ".", language="en"),
						dtype=np.int64,
					)[None, :]
					with ttslock:
						tts_audio = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping[voice])
					with juliuslock:
						tts_audio = resample_function(torch.as_tensor(tts_audio[0])).numpy()
					save_wav(wav=tts_audio, path=sample_dir + voice + "/" + value + ".wav", sample_rate=16000)
					sound = AudioSegment.from_file(sample_dir + voice + "/" + value + ".wav", format="wav")
					silenced_word = strip_silence(sound)
					silenced_word.export(sample_dir + voice + "/" + value + ".wav", format='wav')
			if not os.path.exists(f"{sample_dir}{voice}/pitch_{str(pitch_adjustment)}/{letters_to_use[-1]}.wav"):
				print(f"\nblips generate pitch samples: {voice=}, {pitch_adjustment=}, {text=}")
				os.makedirs(sample_dir + voice + "/pitch_" + str(pitch_adjustment), exist_ok=True)
				for i, value in enumerate(letters_to_use):
					audio, _ = librosa.load(sample_dir + voice + "/" + value + ".wav", 16000)

					processed_audio = model_to_use.exec(speaker_id, np.expand_dims(audio, axis=0), int(pitch_adjustment))
					final_audio = processed_audio.detach().cpu().numpy()
					output_sound = AudioSegment(
						final_audio,
						frame_rate=40000,
						sample_width=2,
						channels=1,
					)
					output_sound.export(sample_dir + voice + "/pitch_" + str(pitch_adjustment) + "/" + value + ".wav", format="wav")
			word_letter_count = 0
			for i, letter in enumerate(text):
				if not letter.isalpha() and not letter.isnumeric():
					continue
				if letter == ' ':
					word_letter_count = 0
					letter_sound = AudioSegment.empty()
					new_sound = letter_sound._spawn(b'\x00' * (40000 // 3), overrides={'frame_rate': 40000})
					new_sound = new_sound.set_frame_rate(40000)
					result_sound += new_sound
				else:
					if not word_letter_count % 2 == 0:
						word_letter_count += 1
						continue # Skip every other letter
					if not os.path.isfile(sample_dir + voice + "/" + letter + ".wav"):
						continue
					if not os.path.isdir(sample_dir + voice + "/pitch_" + str(pitch_adjustment)):
						os.mkdir(sample_dir + voice + "/pitch_" + str(pitch_adjustment))
					
					letter_sound = AudioSegment.from_file(sample_dir + voice + "/pitch_" + str(pitch_adjustment) + "/" + letter + ".wav")

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
	global request_count, last_request_time, timeofdeath, tts
	if timeofdeath > 0:
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	#if ((request_count > 100) and ((time.time() > last_request_time+120) or (avg_request_time > 5))) or tts_errors >= 64:
	#	timeofdeath = time.time() + 3
	#	return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if request_count > 32768:
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if (time.time() > last_request_time+(1*60*60)):
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	#if last_request_time < 1:
	#	request_count += 1
	#	test_tts()
	#	last_request_time = time.time()
	
	return f"OK count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 200

@app.route("/pitch-available")
def pitch_available():
	return make_response("Pitch available", 200)

readyup()
if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5231, threads=16, backlog=32, connection_limit=128, cleanup_interval=1, channel_timeout=10)
