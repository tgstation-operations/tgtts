import torch
from TTS.api import TTS
import os
import signal
import sys
import io
import time
import json
import gc
import random
import numpy as np
import ffmpeg
from typing import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'rvc_webui')))
from rvc_webui.modules import models
from rvc_webui.modules.utils import load_audio
from flask import Flask, request, send_file, abort, make_response
from numpy import interp
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence
from fairseq import checkpoint_utils
from fairseq.models.hubert.hubert import HubertModel
from rvc_webui.modules.shared import ROOT_DIR, device, is_half
import requests
import librosa

from julius.resample import ResampleFrac

import cProfile
import threading
import sd_notify
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.4,max_split_size_mb:128'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'
#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
tts = None
resamplefunc = ResampleFrac(22050, 16000)
vc_models = {
	"TGStation_Crepe_1.pth": "./models/checkpoints/speakers_tgstation_1.json",
	"TGStation_Crepe_2.pth": "./models/checkpoints/speakers_tgstation_2.json",
	"TGStation_Crepe_3.pth": "./models/checkpoints/speakers_tgstation_3.json",
}
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))
#torch.backends.cudnn.benchmark = True

ttslock = threading.Lock()
juliuslock = threading.Lock()
rvclock = threading.Lock()

letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
random_factor = 0.35
os.makedirs('samples', exist_ok=True)

app = Flask(__name__)

#systemd stuff.
watchdog_timer = None
watchdog = sd_notify.Notifier()
if not watchdog.enabled(): 
	watchdog = None

request_count = 0
tts_errors = 0
rvc_errors = 0
last_request_time = time.time()
avg_request_time = 0
avg_tts_time = 0
avg_rvc_time = 0

timeofdeath = 0

voice_name_mapping = {}
use_voice_name_mapping = True
with open("./tts_data/tts_voices_mapping.json", "r") as file:
	voice_name_mapping = json.load(file)
	if len(voice_name_mapping) == 0:
		use_voice_name_mapping = False

voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}

def load_embedder():
	global embedder_model, loaded_embedder_model
	with torch.inference_mode():
		emb_file = "./models/embeddings/hubert_base.pt"
		models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
			[emb_file],
			suffix="",
		)
		embedder_model = models[0]
		embedder_model = embedder_model.to(device)

		if is_half:
			embedder_model = embedder_model.half()
		else:
			embedder_model = embedder_model.float()
		embedder_model.eval().share_memory()

		loaded_embedder_model = "hubert_base"
	return embedder_model


loaded_models = []
embedder_model: Optional[HubertModel] = load_embedder()
voice_lookup = {}
with torch.inference_mode():
	for model in vc_models.keys():
		print(model)
		voice_lookup[model] = json.load(open(vc_models[model], "r"))

		model_path = os.path.join(models.MODELS_DIR, "checkpoints", model)
		weight = torch.load(model_path, map_location="cuda:0")
		vc_model = models.VoiceConvertModel(model, weight)
		loaded_models.append(vc_model)
		print("Loaded model " + str(model))
#vc_model = models.get_vc_model(model_path)
embedding_output_layer = 12

def tts_log(text):
	return
	with open('ttsd-gpu-runlog.txt', 'a') as f:
		f.write(f"{text}\n")

def maketts():
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
		watchdog.status(f"count:{request_count}({tts_errors}/{rvc_errors}) t:{two_way_round(avg_request_time, 4)}s({two_way_round(avg_tts_time, 4)}/{two_way_round(avg_rvc_time, 4)}) last:{two_way_round(time.time() - last_request_time, 1)}s")
		if timeofdeath > 0 and time.time() > timeofdeath:
			watchdog.notify_error()
			return
		if tts_errors > 5:
			#watchdog.notify_error()
			return
		if (request_count > 10) and (time.time() > last_request_time+30):
			request_count += 1
			test_tts()
			last_request_time = time.time()

		with ttslock:
			watchdog.notify()
			schedule_watchdog()

def test_tts():
	global tts
	tts_log("test_tts()")
	with ttslock:
		tts_log("test_tts():lock")
		if not tts: 
			tts = maketts()
		with io.BytesIO() as data_bytes:
			with torch.inference_mode():
				tts_log("do test_tts()")
				tts.tts_to_file(text="The quick brown fox jumps over the lazy dog.", speaker="maleadult01default", file_path=data_bytes)

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
	global request_count, last_request_time, avg_request_time, avg_tts_time, avg_rvc_time, tts_errors, rvc_errors, tts
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	request_count += 1
	tts_errors += 1
	
	starttime = time.time()
	text = request.json.get("text", "")
	voice = request.json.get("voice", "")
	pitch_adjustment = request.json.get("pitch", "0")
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
			tts_starttime = time.time()
			with io.BytesIO() as tts_data_bytes:
				steptime = time.time()
				with ttslock:
					audio_bytes = tts.tts(text=text, speaker=voice)
				print(f"tts:{time.time()-steptime}")
				steptime = time.time()
				my_locals = locals()
				
				
				avg_tts_time = mc_avg(avg_tts_time, time.time()-tts_starttime)
				rvc_starttime = time.time()
				rvc_errors += 1
				cProfile.runctx('audio_tensor = torch.torch.as_tensor(audio_bytes)', globals(), my_locals, filename='rvcprofile.torch')
				print(f"as_tensor:{time.time()-steptime}")
				steptime = time.time()
				
				cProfile.runctx('audio_tensor = audio_tensor.float()', globals(), my_locals, filename='rvcprofile.torch2')
				print(f"as_tensor:{time.time()-steptime}")
				steptime = time.time()
				
				#cProfile.runctx('audio_numpy = np.array(audio_bytes)', globals(), my_locals, filename='rvcprofile.np_audio')
				print(f"nparray:{time.time()-steptime}")
				steptime = time.time()
				
				with juliuslock:
					cProfile.runctx('audio = resamplefunc(audio_tensor)', globals(), my_locals, filename='rvcprofile.librosa')
				#audio = librosa.resample(np_audio, orig_sr=tts.synthesizer.tts_config.audio["sample_rate"], target_sr=16000)
				
				print(f"librosa:{time.time()-steptime}")
				steptime = time.time()
				
				
				
				with rvclock:
					cProfile.runctx('''audio_opt = model_to_use.vc(
						embedder_model,
						embedding_output_layer,
						model_to_use.net_g,
						speaker_id,
						audio,
						int(pitch_adjustment),
						"dio",
						"",
						0,
						model_to_use.state_dict.get("f0", 1),
						f0_file=None,
					)''', globals(), my_locals, filename='rvcprofile.rvc')
					audio_opt = my_locals['audio_opt']
				
				print(f"vc:{time.time()-steptime}")
				steptime = time.time()
				AudioSegment(audio_opt, frame_rate=model_to_use.tgt_sr, sample_width=2, channels=1).export(data_bytes, format="wav")
				print(f"audiosegment:{time.time()-steptime}")
				steptime = time.time()
			rvc_errors -= 1
			avg_rvc_time = mc_avg(avg_rvc_time, time.time()-rvc_starttime)
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	
	#gc.collect()
	last_request_time = time.time()
	tts_errors -= 1
	avg_request_time = mc_avg(avg_request_time, last_request_time-starttime)
	return result

@app.route("/generate-tts-blips")
def text_to_speech_blips():
	global request_count, tts
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	text = request.json.get("text", "").upper()
	voice = request.json.get("voice", "")
	pitch_adjustment = request.json.get("pitch", "0")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]

	result = None
	
	result_sound = AudioSegment.empty()
	if not os.path.exists('samples/v2/' + voice):
		os.makedirs('samples/v2/' + voice, exist_ok=True)
		with torch.inference_mode():
			with ttslock:
				for i, value in enumerate(letters_to_use):					
					tts.tts_to_file(text=value + ".", speaker=voice, file_path="samples/v2/" + voice + "/" + value + ".wav")
					sound = AudioSegment.from_file("samples/v2/" + voice + "/" + value + ".wav", format="wav")
					silenced_word = strip_silence(sound)
					silenced_word.export("samples/v2/" + voice + "/" + value + ".wav", format='wav')
	if not os.path.isdir("samples/v2/" + voice + "/pitch_" + pitch_adjustment):
		os.makedirs("samples/v2/" + voice + "/pitch_" + pitch_adjustment, exist_ok=True)
		for i, value in enumerate(letters_to_use):
			audio, _ = librosa.load("samples/v2/" + voice + "/" + letter + ".wav", 16000)
			audio_opt = model_to_use.vc(
				embedder_model,
				embedding_output_layer,
				model_to_use.net_g,
				speaker_id,
				audio,
				int(pitch_adjustment),
				"crepe",
				"",
				0,
				model_to_use.state_dict.get("f0", 1),
				f0_file=None,
			)
			output_sound = AudioSegment(
				audio_opt,
				frame_rate=model_to_use.tgt_sr,
				sample_width=2,
				channels=1,
			)
			output_sound.export("samples/v2/" + voice + "/pitch_" + pitch_adjustment + "/" + letter + ".wav", format="wav")
	word_letter_count = 0
	for i, letter in enumerate(text):
		if not letter.isalpha() or letter.isnumeric() or letter == " ":
			continue
		if letter == ' ':
			word_letter_count = 0
			new_sound = letter_sound._spawn(b'\x00' * (40000 // 3), overrides={'frame_rate': 40000})
			new_sound = new_sound.set_frame_rate(40000)
		else:
			if not word_letter_count % 2 == 0:
				word_letter_count += 1
				continue # Skip every other letter
			if not os.path.isfile("samples/v2/" + voice + "/pitch_" + pitch_adjustment + "/" + letter + ".wav"):
				continue
			word_letter_count += 1
			letter_sound = AudioSegment.from_file("samples/v2/" + voice + "/pitch_" + pitch_adjustment + "/" + letter + ".wav")


			raw = letter_sound.raw_data[5000:-5000]
			octaves = 1 + random.random() * random_factor
			frame_rate = int(letter_sound.frame_rate * (2.0 ** octaves))

			new_sound = letter_sound._spawn(raw, overrides={'frame_rate': frame_rate})
			new_sound = new_sound.set_frame_rate(40000)

		result_sound = new_sound if result_sound is None else result_sound + new_sound
	with io.BytesIO() as data_bytes:
		result_sound.export(data_bytes, format='wav')
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	request_count += 1
	return result

@app.route("/tts-voices")
def voices_list():
	global tts
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	#gc.collect()
	if use_voice_name_mapping:
		data = list(voice_name_mapping.values())
		data.sort()
		return json.dumps(data)
	else:
		with ttslock:
			return json.dumps(tts.voices)

@app.route("/health-check")
def tts_health_check():
	global request_count, last_request_time, timeofdeath, tts
	#gc.collect()
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	if timeofdeath > 0:
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if ((request_count > 100) and (time.time() > last_request_time+60)) or (avg_request_time > 2) or tts_errors >= 5:
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if request_count > 4096:
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if (time.time() > last_request_time+(1*60*60)):
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if last_request_time < 1:
		request_count += 1
		test_tts()
		last_request_time = time.time()
	
	return f"OK count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 200

@app.route("/pitch-available")
def pitch_available():
	return "Pitch available", 200

tts_log("START")
readyup()


if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5231, threads=2, backlog=2, connection_limit=128, cleanup_interval=1, channel_timeout=10)
