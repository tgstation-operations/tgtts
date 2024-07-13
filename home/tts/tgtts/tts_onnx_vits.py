import sys
import os
import io
import gc
import time
import math
import random
import statistics
import string
import threading
import json
from typing import *
from flask import Flask, request, send_file, abort, make_response
import numpy as np
import torch
import torch.nn.functional as F
from torch import device
from torch.cuda.amp import autocast
import onnxruntime as ort
from tgvits import TgVits as Vits
from TTS.tts.configs.vits_config import VitsConfig
from julius.resample import ResampleFrac
import setproctitle
setproctitle.setproctitle(os.path.basename(__file__))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ttslock = threading.Lock()

#systemd stuff.
watchdog_timer = None
try:
	import sd_notify
	watchdog = sd_notify.Notifier()
	if not watchdog.enabled(): 
		watchdog = None
except:
	watchdog = None

def watchdog_status(status_text):
	if watchdog:
		watchdog.status(status_text)

watchdog_status("Starting")

tts_errors = 0
last_request_time = time.time()
avg_request_time = 0
avg_request_len = 0
avg_request_delay = 0
avg_request_rate = 0
request_count = 0

models_hydrated = False
timeofdeath = 0

tg_tts_providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'DEFAULT',
		'cudnn_conv_use_max_workspace': '1',
    }),
    #'CPUExecutionProvider',
]
tg_tts_sess_options = ort.SessionOptions()
tg_tts_sess_options.enable_mem_pattern = False
tg_tts_sess_options.enable_cpu_mem_arena = False
tg_tts_sess_options.enable_mem_reuse = False

tg_tts_run_options = ort.RunOptions()
tg_tts_run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")

watchdog_status("Loading GPU Models")
with torch.inference_mode():
	resample_function = ResampleFrac(22050, 16000) #compile a gpu kernal for a given resample fraction
	config = VitsConfig()
	config.load_json("./models/vits/tgvits.config.json")
	vits = Vits.init_from_config(config)
	vits.load_onnx("./models/vits/tgvits.simplified.onnx", True)

watchdog_status("Loading Application")
voice_name_mapping = {}
use_voice_name_mapping = True
with open("./models/vits/tgvits.voice_mappings.json", "r") as file:
	voice_name_mapping = json.load(file)
	if len(voice_name_mapping) == 0:
		use_voice_name_mapping = False
				
voice_id_mapping = {}
with open("./models/vits/tgvits.voice_ids.json", "r") as file:
	voice_id_mapping = json.load(file)

voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}

app = Flask(__name__)

def tts_log(text):
	return
	with open('ttsd-gpu-runlog.txt', 'a') as f:
		f.write(f"{text}\n")
		
def mc_avg(old, new):
	if (old > 0):
		if (request_count > 10):
			if (request_count > 100):
				return (old*0.99) + (new*0.01)
			return (old*0.90) + (new*0.10)
		return (old*0.70) + (new*0.30)
	return new

def two_way_round(number, ndigits = 0, predigits = 0):
	number = round(number, ndigits)
	return (f"%0{predigits}.{ndigits}f") % number

def toms(number):
	return number*1000

def ping_watchdog():
	if watchdog:
		watchdog.status(f"count:{request_count}({tts_errors}) len:{two_way_round(avg_request_len, 1, 2)} time:{two_way_round(toms(avg_request_time), 0, 3)}ms rate:{two_way_round(avg_request_rate, 1, 3)}/s last:{two_way_round(time.time() - last_request_time, 1, 3)}s(avg:{two_way_round(avg_request_delay, 2, 4)}s)")
		if timeofdeath > 0 and time.time() > timeofdeath:
			watchdog.notify_error()
			return
		with ttslock:
			watchdog.notify()
			schedule_watchdog()


def hydrate_models():
	watchdog_status("Hydrating models")
	text_inputs = np.asarray(
		vits.tokenizer.text_to_ids("The quick brown fox jumps over the lazy dog", language="en"),
		dtype=np.int64, 
	)[None, :]
	speaker = random.choice(list(voice_id_mapping.keys()))
	with ttslock:
		tts_res = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping[speaker])

	audio = resample_function(torch.as_tensor(tts_res[0])).numpy() #resample to 16000 using precompiled gpu kernal
	del tts_res

def test_tts():
	tts_log("test_tts()")
	with ttslock:
		tts_log("test_tts():lock")
		
		with io.BytesIO() as data_bytes:
			with torch.inference_mode():
				tts_log("do test_tts()")
				text_inputs = np.asarray(
					vits.tokenizer.text_to_ids("The quick brown fox jumps over the lazy dog", language="en"),
					dtype=np.int64, 
					)[None, :]
				speaker = random.choice(list(voice_id_mapping.keys()))
				tts_res = vits.inference_onnx(text_inputs, speaker_id=voice_id_mapping[speaker])

def schedule_watchdog():
	global watchdog_timer
	if watchdog:
		watchdog_timer = threading.Timer(0.3, ping_watchdog)
		watchdog_timer.start()

def readyup():
	global models_hydrated
	print("readyup()\n")
	tts_log("readyup()")
	if watchdog:
		tts_log("readyup(): watchdog")
		try:
			hydrate_models()
		finally:
			models_hydrated = True #mark them as hydrated anyways and let the normal health checking stuff detect if there is an errored state
		gc_loop()
		tts_log("readyup(): getlock")
		with ttslock:
			tts_log("readyup(): gotlock, sending ready")
			watchdog.ready()
			tts_log("readyup(): ready sent")
			schedule_watchdog()
	else:
		models_hydrated = True
		gc_loop()
		
def gc_loop():
	threading.Timer(1, gc_loop).start()
	#gc.collect()
	torch.cuda.empty_cache()

@app.route("/generate-vits")
def text_to_speech():
	global request_count, last_request_time, avg_request_time, tts_errors, avg_request_len, avg_request_rate, avg_request_delay
	request_count += 1
	tts_errors += 1
	starttime = time.time()
	
	text = request.json.get("text", "")
	text_ids = request.json.get("text_ids", None)
	voice = request.json.get("voice", "")
	format = request.args.get("format", "wav").lower()
	
	log_prefix = request.headers.get('X-Log-Prefix', "")
	
	if last_request_time > 0:
		avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
	if avg_request_delay > 0:
		avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
	avg_request_len = mc_avg(avg_request_len, len(text))
	
	print(f"\n{log_prefix} vits start: {voice=}, {len(text)=}, {format=}")
	speaker_id = None
	if use_voice_name_mapping and voice in voice_name_mapping_reversed:
		speaker_id = voice_id_mapping[voice_name_mapping_reversed[voice]]
	else:
		speaker_id = voice_id_mapping[voice]
			
		
	with torch.inference_mode():
		with io.BytesIO() as data_bytes:
			ts_starttime = time.time()
			#with ttstokenizerlock:
			steptime = time.time()
			if text_ids is not None:
				text_inputs = np.asarray(
					text_ids,
					dtype=np.int64, 
				)[None, :]
			else:
				text_inputs = np.asarray(
					vits.tokenizer.text_to_ids(text, language="en"),
					dtype=np.int64, 
				)[None, :]
			print(f"{log_prefix} vits.tokenize:{time.time()-steptime}")
			
			steptime = time.time()
			tts_res = vits.inference_onnx(text_inputs, speaker_id=speaker_id)
			print(f"{log_prefix} vits.main:{time.time()-steptime}")

			audio = resample_function(torch.as_tensor(tts_res[0])).numpy() #resample to 16000 using precompiled gpu kernal
			del tts_res

			mimetype = "*/*"
			
			if format == "numpy":
				mimetype = "application/octet-stream"
				np.save(data_bytes, audio)
			elif format == "raw":
				audio = AudioSegment(
					audio,
					frame_rate=16000,
					sample_width=2,
					channels=1,
				)
				audio.export(data_bytes, format="raw")
				mimetype = "audio/pcm"
			else:
				audio = AudioSegment(
					audio,
					frame_rate=16000,
					sample_width=2,
					channels=1,
				)
				audio.export(data_bytes, format="wav")
				mimetype = "audio/wav"
				
			result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype=mimetype)
			result.headers['X-Audio-Framerate'] = "16000"
			result.headers['X-Audio-Samplewidth'] = "2"
			result.headers['X-Audio-Samplebits'] = "16"
			result.headers['X-Audio-Channels'] = "1"

	request_count += 1
	tts_errors -= 1
	last_request_time = time.time()
	avg_request_time = mc_avg(avg_request_time, last_request_time-starttime)
	print(f"{log_prefix} vits total:{time.time()-starttime}")
	torch.cuda.empty_cache()
	result.headers['X-Timing'] = f"{time.time()}"
	return result


@app.route("/tts-voices")
def voices_list():
	if use_voice_name_mapping:
		data = list(voice_name_mapping.values())
		data.sort()
		return json.dumps(data)
	else:
		return json.dumps(list(voice_id_mapping.keys()).sort())

	
stop_point = 32768*(random.randint(2,4)*random.random())
@app.route("/health-check")
def tts_health_check():
	global timeofdeath
	if not models_hydrated:
		return f"HYDRATING: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	if timeofdeath > 0:
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	
	if request_count > int(stop_point / max(min(avg_request_delay, 5), 0.1)):
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	
	if (time.time() > last_request_time+(1*60*60)):
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	#
	
	return f"OK: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 200

threading.Timer(1, readyup).start()
watchdog_status("Activating")
if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5041, threads=16, backlog=32, connection_limit=128, cleanup_interval=1, channel_timeout=10)