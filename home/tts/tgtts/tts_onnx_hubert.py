import sys
import os
import io
import gc
import time
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
#from fairseq import checkpoint_utils
#from fairseq.models.hubert.hubert import HubertModel
import setproctitle
setproctitle.setproctitle(os.path.basename(__file__))


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

watchdog_status("Starting.")

tts_errors = 0
last_request_time = time.time()
avg_request_time = 0
avg_tts_time = 0
avg_request_len = 0
avg_request_delay = 0
avg_request_rate = 0
request_count = 0

timeofdeath = 0

tg_tts_providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'DEFAULT',
		'cudnn_conv_use_max_workspace': '1',
    }),
    'CPUExecutionProvider',
]
tg_tts_sess_options = ort.SessionOptions()
tg_tts_sess_options.enable_mem_pattern = False
tg_tts_sess_options.enable_cpu_mem_arena = False
tg_tts_sess_options.enable_mem_reuse = False

tg_tts_run_options = ort.RunOptions()
tg_tts_run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")

class Embedder(Protocol):
	file: str
	isHalf: bool = True
	dev: device

	model: Any | None = None

	def loadModel(self, file: str, dev: device, isHalf: bool = True):
		...

	def extractFeatures(
		self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
	) -> torch.Tensor:
		...

	def getEmbedderInfo(self):
		return {
			"embedderType": self.embedderType.value,
			"file": self.file,
			"isHalf": self.isHalf,
			"devType": self.dev.type,
			"devIndex": self.dev.index,
		}

	def setProps(
		self,
		file: str,
		dev: device,
		isHalf: bool = True,
	):
		self.file = file
		self.isHalf = isHalf
		self.dev = dev

	def setHalf(self, isHalf: bool):
		self.isHalf = isHalf
		if self.model is not None and isHalf:
			self.model = self.model.half()
		elif self.model is not None and isHalf is False:
			self.model = self.model.float()

	def setDevice(self, dev: device):
		self.dev = dev
		if self.model is not None:
			self.model = self.model.to(self.dev)
		return self

class FairseqHubertOnnx(Embedder):
	def loadModel(self, file: str, dev: device = "cuda:0", isHalf: bool = True) -> Embedder:
		self.gpulock = threading.Lock()
		providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
		sess_options = ort.SessionOptions()
		self.onnx_sess = ort.InferenceSession(
			file,
			sess_options=tg_tts_sess_options,
			providers=tg_tts_providers,
		)

	def extractFeatures(
		self, feats: np.array, log_prefix = ''
	) -> np.array:
		#with self.gpulock:
		steptime = time.time()
		
		feats_out = self.onnx_sess.run(
			["logits"],
			{
				"feats": feats,
			},
			tg_tts_run_options
		)
		print(f"{log_prefix} rvc.hubert.onnx:{time.time()-steptime}")
		return np.array(feats_out)

watchdog_status("Loading GPU Models.")
with torch.inference_mode():
	print("Loading contentvec encoder")
	encoder_2 = FairseqHubertOnnx()
	encoder_2.loadModel("./models/hubert/hubert_base_simple.onnx")
	print("Loaded contentvec encoder")	

watchdog_status("Loading Application.")

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
		watchdog_status(f"count:{request_count}({tts_errors}) len:{two_way_round(avg_request_len, 0, 6)} time:{two_way_round(toms(avg_request_time), 0, 2)}ms rate:{two_way_round(avg_request_rate, 1, 3)}/s last:{two_way_round(time.time() - last_request_time, 1, 3)}s(avg:{two_way_round(avg_request_delay, 2, 4)}s)")
		if timeofdeath > 0 and time.time() > timeofdeath:
			watchdog.notify_error()
			return

		watchdog.notify()
		schedule_watchdog()
			
def schedule_watchdog():
	global watchdog_timer
	if watchdog:
		watchdog_timer = threading.Timer(0.3, ping_watchdog)
		watchdog_timer.start()

def readyup():
	print("readyup()\n")
	tts_log("readyup()")
	if watchdog:
		tts_log("readyup(): watchdog")
		gc_loop()
		tts_log("readyup(): getlock")
		watchdog.ready()
		tts_log("readyup(): ready sent")
		schedule_watchdog()
	else:
		gc_loop()
		
def gc_loop():
	threading.Timer(1, gc_loop).start()
	#gc.collect()
	torch.cuda.empty_cache()
	
	


@app.route("/generate-feats")
def text_to_speech():
	global request_count, last_request_time, avg_request_time, tts_errors, avg_request_len, avg_request_rate, avg_request_delay
	request_count += 1
	tts_errors += 1
	starttime = time.time()
	
	data = request.data
	
	log_prefix = request.headers.get('X-Log-Prefix', "")
	
	if last_request_time > 0:
		avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
	if avg_request_delay > 0:
		avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
	avg_request_len = mc_avg(avg_request_len, len(data))
	
	print(f"\n{log_prefix} hubert start: {len(data)=}")
	
	steptime = time.time()
	with io.BytesIO(data) as data_bytes:
		feats = np.load(data_bytes)
	print(f"{log_prefix} feats.load:{time.time()-steptime}")
	
	with torch.inference_mode():
		with io.BytesIO() as data_bytes:
			tts_starttime = time.time()
			#with ttstokenizerlock:
			
			steptime = time.time()
			feats = encoder_2.extractFeatures(feats, log_prefix=log_prefix)
			print(f"{log_prefix} feats.main:{time.time()-steptime}")
			
			steptime = time.time()
			np.save(data_bytes, feats)
			print(f"{log_prefix} feats.save:{time.time()-steptime}")
			result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="application/octet-stream")

	tts_errors -= 1
	last_request_time = time.time()
	avg_request_time = mc_avg(avg_request_time, last_request_time-starttime)
	print(f"{log_prefix} hubert total:{time.time()-starttime}")
	result.headers['X-Timing'] = f"{time.time()}"
	return result



	
stop_point = 65536*(random.randint(2,5)*random.random())
@app.route("/health-check")
def tts_health_check():
	global timeofdeath

	if timeofdeath > 0:
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	
	if request_count > int(stop_point / max(min(avg_request_delay, 5), 0.1)):
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	
	if (time.time() > last_request_time+(1*60*60)):
		timeofdeath = time.time() + 3
		return f"EXPIRED: count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 500
	#
	
	return f"OK count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 200

threading.Timer(2, readyup).start()

watchdog_status("Activating.")

if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5061, threads=16, backlog=32, connection_limit=128, cleanup_interval=1, channel_timeout=10)