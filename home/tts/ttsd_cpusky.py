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
import requests
from flask import Flask, request, send_file
from numpy import interp
from pydub import AudioSegment
from pydub.silence import split_on_silence
import threading
import sd_notify
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.1'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'
#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

tts = None
ttslock = threading.Lock()

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
last_request_time = time.time()
avg_request_time = 0

timeofdeath = 0

voice_name_mapping = {}
use_voice_name_mapping = True
with open("./tts_data/tts_voices_mapping.json", "r") as file:
	voice_name_mapping = json.load(file)
	if len(voice_name_mapping) == 0:
		use_voice_name_mapping = False

voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}

def tts_log(text):
	return
	with open('ttsd-gpu-runlog.txt', 'a') as f:
		f.write(f"{text}\n")

def maketts():
	tts_log("maketts()")
	return
	return TTS(model_path="./tts_data/model_no_disc.pth", config_path="./tts_data/config.json", progress_bar=False, gpu=False)
	#return TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=True)
	
def ping_watchdog():
	global request_count, last_request_time
	if watchdog:
		watchdog.status(f"count:{request_count}({tts_errors}) t:{round(avg_request_time, 4)}s last:{round(time.time() - last_request_time, 1)}s")
		if timeofdeath > 0 and time.time() > timeofdeath:
			watchdog.notify_error()
			return
		if tts_errors > 20:
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
	return
	tts_log("test_tts()")
	with ttslock:
		tts_log("test_tts():lock")
		if not tts: 
			tts = maketts()
		with io.BytesIO() as data_bytes:
			with torch.inference_mode():
				tts_log("do test_tts()")
				tts.tts_to_file(text="Hi.", speaker="maleadult01default", file_path=data_bytes)

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
	gc.collect()
	
@app.route("/tts")
def text_to_speech_old():
	global request_count, last_request_time, avg_request_time, tts_errors, tts
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	request_count += 1
	tts_errors += 1
	
	starttime = time.time()
	text = request.json.get("text", "")
	voice = request.json.get("voice", "")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]

	result = None
	with io.BytesIO() as data_bytes:
		with torch.inference_mode():
			with ttslock:
				tts.tts_to_file(text=text, speaker=voice, file_path=data_bytes)
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	
	#gc.collect()
	last_request_time = time.time()
	tts_errors -= 1
	avg_request_time = ((avg_request_time * 0.99) + ((last_request_time-starttime) * 0.01))
	return result
	
@app.route("/generate-tts")
def text_to_speech():
	global request_count, last_request_time, avg_request_time, tts_errors, tts
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	request_count += 1
	tts_errors += 1
	
	starttime = time.time()
	text = request.json.get("text", "")
	voice = request.json.get("voice", "")
	if use_voice_name_mapping:
		voice = voice_name_mapping_reversed[voice]

	result = None
	with io.BytesIO() as data_bytes:
		with torch.inference_mode():
			with ttslock:
				tts.tts_to_file(text=text, speaker=voice, file_path=data_bytes)
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	
	#gc.collect()
	last_request_time = time.time()
	tts_errors -= 1
	avg_request_time = ((avg_request_time * 0.99) + ((last_request_time-starttime) * 0.01))
	return result

@app.route("/generate-tts-blips")
def text_to_speech_blips():
	global request_count, last_request_time, avg_request_time, tts_errors, tts
	if not tts: 
		with ttslock:
			if not tts: 
				tts = maketts()
	
	
	text = request.json.get("text", "").upper()
	voice = request.json.get("voice", "")
	ogvoice = voice
	if use_voice_name_mapping:
		ogvoice = voice
		voice = voice_name_mapping_reversed[voice]

	result = None
	
	result_sound = None
	if not os.path.exists('samples/' + voice):
		print(f"calling to generate samples for {voice}/{ogvoice}", file=sys.stderr)
		print(f"calling to generate samples for {voice}/{ogvoice}")
		response = requests.get(f"http://localhost:5103/generate-tts-blips", json={ 'text': text, 'voice': ogvoice }, stream=True)
		return response.raw.read(), response.status_code, {k: v for k, v in response.headers.items() if k not in ['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']}.items()
	request_count += 1
	tts_errors += 1
	starttime = time.time()
	word_letter_count = 0
	for i, letter in enumerate(text):
		if not letter.isalpha() or letter.isnumeric() or letter == " ":
			continue
		if letter == ' ':
			word_letter_count = 0
			new_sound = letter_sound._spawn(b'\x00' * (22050 // 3), overrides={'frame_rate': 22050})
			new_sound = new_sound.set_frame_rate(22050)
		else:
			if not word_letter_count % 2 == 0:
				word_letter_count += 1
				continue # Skip every other letter
			if not os.path.isfile("samples/" + voice + "/" + letter + ".wav"):
				continue
			word_letter_count += 1
			letter_sound = AudioSegment.from_file("samples/" + voice + "/" + letter + ".wav")

			raw = letter_sound.raw_data[2500:-2500]
			octaves = 1 + random.random() * random_factor
			frame_rate = int(letter_sound.frame_rate * (2.0 ** octaves))

			new_sound = letter_sound._spawn(raw, overrides={'frame_rate': frame_rate})
			new_sound = new_sound.set_frame_rate(22050)

		result_sound = new_sound if result_sound is None else result_sound + new_sound
	with io.BytesIO() as data_bytes:
		if result_sound:
			result_sound.export(data_bytes, format='wav')
		result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
	
	last_request_time = time.time()
	tts_errors -= 1
	avg_request_time = ((avg_request_time * 0.99) + ((last_request_time-starttime) * 0.01))
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
	if ((request_count > 100) and (time.time() > last_request_time+60)) or (avg_request_time > 2) or tts_errors > 20:
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if request_count > 16384:
		timeofdeath = time.time() + 3
	if last_request_time < 1:
		request_count += 1
		test_tts()
		last_request_time = time.time()
	
	return f"OK count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 200

tts_log("START")
readyup()


if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5121, threads=8, cleanup_interval=1, channel_timeout=10)
