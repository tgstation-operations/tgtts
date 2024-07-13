import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxcrepe')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxcrepe', 'onnxcrepe')))
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
from TTS.api import TTS
from pydub import AudioSegment
import requests
import torchcrepe
import onnxcrepe
from onnxcrepe.session import CrepeInferenceSession
from tgtts_utils import ElapsedFuturesSession
import setproctitle
setproctitle.setproctitle(os.path.basename(__file__))

request_session = requests.session()

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
avg_request_len = 0
avg_request_delay = 0
avg_request_rate = 0

request_count = 0

hubert_port = os.getenv("TTS_HUBERT_PORT", "5060")
vits_port = os.getenv("TTS_VITS_PORT", "5040")

models_hydrated = False
timeofdeath = 0


tensorrt_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tensorrt_cache'))
os.makedirs(tensorrt_cache_dir, exist_ok=True)
tg_tts_providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        #'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
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

# CLASSES BORROWED AND MODIFIED FROM https://github.com/w-okada/voice-changer

class CrepePitchExtractor():
	def __init__(self):
		self.session = CrepeInferenceSession(
			model='full',
			sess_options = tg_tts_sess_options,
			providers=tg_tts_providers)

	def extract(self, audio, f0_up_key, sr, window, silence_front=0, log_prefix=''):
		steptime = time.time()
		n_frames = int(len(audio) // window) + 1
		start_frame = int(silence_front * sr / window)
		real_silence_front = start_frame * window / sr

		silence_front_offset = int(np.round(real_silence_front * sr))
		audio = audio[silence_front_offset:]

		f0_min = 50
		f0_max = 1100
		f0_mel_min = 1127 * np.log(1 + f0_min / 700)
		f0_mel_max = 1127 * np.log(1 + f0_max / 700)
		f0, pd = onnxcrepe.predict(self.session, audio.cpu().numpy(), sr, fmin=f0_min, fmax=f0_max, batch_size=512, decoder=onnxcrepe.decode.weighted_argmax, return_periodicity=True)
		f0 = torch.from_numpy(f0).cpu()
		pd = torch.from_numpy(pd).cpu()
		f0 = torchcrepe.filter.median(f0, 3)  # 本家だとmeanですが、harvestに合わせmedianフィルタ
		pd = torchcrepe.filter.median(pd, 3)
		f0[pd < 0.1] = 0
		f0 = f0.squeeze()

		f0 = torch.nn.functional.pad(
			f0, (start_frame, n_frames - f0.shape[0] - start_frame)
		)

		f0 *= pow(2, f0_up_key / 12)
		f0bak = f0.detach().cpu().numpy()
		f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
		f0_mel = torch.clip(
			(f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0, 1.0, 255.0
		)
		f0_coarse = f0_mel.round().detach().cpu().numpy().astype(int)

		print(f"{log_prefix} rvc.crepe.onnx:{time.time()-steptime}")
		return f0_coarse, f0bak

class OnnxRVCInferencer():
	def loadModel(self, file: str):
		self.gpulock = threading.Lock()
		providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]

		self.onnx_sess = ort.InferenceSession(
			file,
			sess_options=tg_tts_sess_options,
			providers=tg_tts_providers,
		)
		# check half-precision
		first_input_type = self.onnx_sess.get_inputs()[0].type
		print(f"{first_input_type=}")
		if first_input_type == "tensor(float)":
			self.isHalf = False
		else:
			self.isHalf = True

		self.model = self.onnx_sess
		return self

	def infer(
		self,
		feats: torch.Tensor,
		pitch_length: torch.Tensor,
		pitch: torch.Tensor,
		pitchf: torch.Tensor,
		sid: torch.Tensor,
		log_prefix = '',
	) -> torch.Tensor:
		
		if pitch is None or pitchf is None:
			raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")
		
		steptime = time.time()
		binding = self.model.io_binding()
		#print(f"{feats.device=} {pitch_length.device=} {pitch.device=} {pitchf.device=} {sid.device=}")
		if self.isHalf:
			binding.bind_cpu_input('feats', feats.detach().cpu().numpy().astype(np.float16))
		else:
			binding.bind_cpu_input('feats', feats.detach().cpu().numpy().astype(np.float32))
		
		binding.bind_input(
			name='p_len',
			device_type='cuda',
			device_id=0,
			element_type=np.int64,
			shape=tuple(pitch_length.shape),
			buffer_ptr=pitch_length.detach().contiguous().data_ptr(),
		)
		
		binding.bind_input(
			name='pitch',
			device_type='cuda',
			device_id=0,
			element_type=np.int64,
			shape=tuple(pitch.shape),
			buffer_ptr=pitch.detach().contiguous().data_ptr(),
		)
		
		binding.bind_input(
			name='pitchf',
			device_type='cuda',
			device_id=0,
			element_type=np.float32,
			shape=tuple(pitchf.shape),
			buffer_ptr=pitchf.detach().contiguous().data_ptr(),
		)
		
		binding.bind_input(
			name='sid',
			device_type='cuda',
			device_id=0,
			element_type=np.int64,
			shape=tuple(sid.shape),
			buffer_ptr=sid.detach().contiguous().data_ptr(),
		)
		binding.bind_output('audio')
		print(f"{log_prefix} rvc.main.bindings:{time.time()-steptime:f}")
		with self.gpulock:
			steptime = time.time()
			self.model.run_with_iobinding(binding, tg_tts_run_options)
			print(f"{log_prefix} rvc.main.onnx:{time.time()-steptime:f}")
		steptime = time.time()
		audio1 = binding.copy_outputs_to_cpu()
		print(f"{log_prefix} rvc.main.outbindings:{time.time()-steptime:f}")
		steptime = time.time()
		
		res = torch.tensor(np.array(audio1))
		print(f"{log_prefix} rvc.main.setres:{time.time()-steptime:f}")
		return torch.tensor(np.array(audio1))

class Pipeline(object):
	inferencer: OnnxRVCInferencer
	pitchExtractor: CrepePitchExtractor
	
	index: Any | None
	big_npy: Any | None
	# feature: Any | None

	targetSR: int
	device: torch.device
	isHalf: bool

	def __init__(
		self,
		inferencer: OnnxRVCInferencer,
		pitchExtractor: CrepePitchExtractor,
		# feature: Any | None,
		targetSR,
		device,
		isHalf,
	):
		self.gpulock = threading.Lock()
		self.inferencer = inferencer
		self.pitchExtractor = pitchExtractor

		# self.feature = feature

		self.targetSR = targetSR
		self.device = device
		self.isHalf = isHalf

		self.sr = 16000
		self.window = 160

	def exec(
		self,
		sid,
		audio,
		f0_up_key,
		if_f0 = True,
		silence_front = 0,
		repeat = 0,
		log_prefix = ''
	):
		steptime = time.time()
		#with self.gpulock:
		# 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。
		# self.t_pad = self.sr * repeat  # 1秒
		# self.t_pad_tgt = self.targetSR * repeat  # 1秒　出力時のトリミング(モデルのサンプリングで出力される)
		
		audio = torch.from_numpy(audio).to(device=self.device, dtype=torch.float32)
		audio.unsqueeze(0)

		quality_padding_sec = (repeat * (audio.shape[1] - 1)) / self.sr  # padding(reflect)のサイズは元のサイズより小さい必要がある。

		self.t_pad = round(self.sr * quality_padding_sec)  # 前後に音声を追加
		self.t_pad_tgt = round(self.targetSR * quality_padding_sec)  # 前後に音声を追加　出力時のトリミング(モデルのサンプリングで出力される)
		audio_pad = F.pad(audio, (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
		p_len = audio_pad.shape[0] // self.window
		sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
		
		# tensor型調整
		#with self.gpulock:
		feats = audio_pad
		if feats.dim() == 2:  # double channels
			feats = feats.mean(-1)
		assert feats.dim() == 1, feats.dim()
		feats = feats.view(1, -1)
		print(f"{log_prefix} rvc.preamble:{time.time()-steptime:f}")
		# embedding
		with ElapsedFuturesSession(session=request_session) as session, io.BytesIO() as data_bytes:
			steptime = time.time()
			np.save(data_bytes, feats.detach().cpu().numpy())
			data_bytes.seek(0)
			print(f"{log_prefix} rvc.hubert.save:{time.time()-steptime}")
			response_future = session.get(
				f"http://localhost:{hubert_port}/generate-feats", 
				data=data_bytes,
				headers={
					'Content-Type': 'application/octet-stream', 
					'Accept': 'application/octet-stream',
					'X-Log-Prefix': log_prefix
				}
			)
			
			# RVC QualityがOnのときにはsilence_frontをオフに。
			silence_front = silence_front if repeat == 0 else 0

			# ピッチ検出
			pitch, pitchf = None, None
			if if_f0 == 1:
				pitch, pitchf = self.pitchExtractor.extract(
					audio_pad,
					f0_up_key,
					self.sr,
					self.window,
					silence_front=silence_front,
					log_prefix=log_prefix,
				)
				steptime = time.time()
				#with self.gpulock:
				pitch = pitch[:p_len]
				pitchf = pitchf[:p_len]
				pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
				pitchf = torch.tensor(pitchf, device=self.device, dtype=torch.float).unsqueeze(0)
				print(f"{log_prefix} rvc.crepe.postamble:{time.time()-steptime:f}")

			with autocast(enabled=self.isHalf):
				try:
					steptime = time.time()
					response = response_future.result()
					with io.BytesIO(response.content) as response_content:
						print(f"{log_prefix} rvc.hubert.subrequest.wait:{time.time()-steptime:f} rvc.hubert.subrequest.elapsed:{response.elapsed:f}")
						if response.overhead is not None:
							print(f"{log_prefix} rvc.hubert.subrequest.overhead:{response.overhead:f}")
						steptime = time.time()
						feats = np.load(response_content)
						print(f"{log_prefix} rvc.hubert.load:{time.time()-steptime:f}")
						steptime = time.time()
						feats = torch.tensor(feats)
						print(f"{log_prefix} rvc.hubert.to_tensor:{time.time()-steptime:f}")
				except RuntimeError as e:
					raise e
		steptime = time.time()
		#with self.gpulock:
		feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
		# ピッチサイズ調整
		p_len = audio_pad.shape[0] // self.window
		if feats.shape[1] < p_len:
			p_len = feats.shape[1]
			if pitch is not None and pitchf is not None:
				pitch = pitch[:, :p_len]
				pitchf = pitchf[:, :p_len]

		p_len = torch.tensor([p_len], device=self.device).long()

		# apply silent front for inference
		npyOffset = math.floor(silence_front * 16000) // 360
		feats = feats[:, npyOffset * 2 :, :]
		feats_len = feats.shape[1]
		if pitch is not None and pitchf is not None:
			pitch = pitch[:, -feats_len:]
			pitchf = pitchf[:, -feats_len:]
		p_len = torch.tensor([feats_len], device=self.device).long()
		print(f"{log_prefix} rvc.postamble:{time.time()-steptime:f}")
		# 推論実行
		try:
			with torch.no_grad():
				with autocast(enabled=self.isHalf):
					audio1 = self.inferencer.infer(feats, p_len, pitch, pitchf, sid, log_prefix=log_prefix)[0][0, 0]
					#with self.gpulock:
					audio1 = (
						torch.clip(
							audio1.to(dtype=torch.float32),
							-1.0,
							1.0,
						)
						* 32767.5
					).data.to(dtype=torch.int16)
		except RuntimeError as e:
			if "HALF" in e.__str__().upper():
				print("11", e)
			else:
				raise e
		
		steptime = time.time()
		#with self.gpulock:
		del feats, p_len, sid, audio, audio_pad, quality_padding_sec
		
		torch.cuda.empty_cache()
		
		# inferで出力されるサンプリングレートはモデルのサンプリングレートになる。
		# pipelineに（入力されるときはhubertように16k）
		if self.t_pad_tgt != 0:
			offset = self.t_pad_tgt
			end = -1 * self.t_pad_tgt
			audio1 = audio1[offset:end]

		del pitch, pitchf
		torch.cuda.empty_cache()
		return audio1
watchdog_status("Loading Models")
with torch.inference_mode():
	print("Loading RVC models")
	watchdog_status("Loading RVC model 1.")
	rvc_1 = OnnxRVCInferencer()
	rvc_1.loadModel("./models/rvc/tg_rvc_crepe_final_1_simple.onnx")
	watchdog_status("Loading RVC model 2.")
	rvc_2 = OnnxRVCInferencer()
	rvc_2.loadModel("./models/rvc/tg_rvc_crepe_final_2_simple.onnx")
	watchdog_status("Loading RVC model 3.")
	rvc_3 = OnnxRVCInferencer()
	rvc_3.loadModel("./models/rvc/tg_rvc_crepe_final_3_simple.onnx")
	watchdog_status("Loading RVC model 4.")
	rvc_4 = OnnxRVCInferencer()
	rvc_4.loadModel("./models/rvc/tg_rvc_crepe_final_4_simple.onnx")
	print("Done loading RVC models")
	print("Loading CrepePitch")
	watchdog_status("Loading Crepe Pitch Extractor.")
	pitch_extract = CrepePitchExtractor()
	print("Loaded")
	print("Loading Pipelines")
	watchdog_status("Loading Pipelines.")
	pipeline_1 = Pipeline(rvc_1, pitch_extract, 40000, "cuda:0", True)
	pipeline_2 = Pipeline(rvc_2, pitch_extract, 40000, "cuda:0", True)
	pipeline_3 = Pipeline(rvc_3, pitch_extract, 40000, "cuda:0", True)
	pipeline_4 = Pipeline(rvc_4, pitch_extract, 40000, "cuda:0", True)
	print("Loaded")
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
watchdog_status("Loading application.")
vc_models = {
	pipeline_1: "./models/rvc/tg_rvc_crepe_final_1_simple.speakers.json",
	pipeline_2: "./models/rvc/tg_rvc_crepe_final_2_simple.speakers.json",
	pipeline_3: "./models/rvc/tg_rvc_crepe_final_3_simple.speakers.json",
	pipeline_4: "./models/rvc/tg_rvc_crepe_final_4_simple.speakers.json",
}
loaded_models = []
voice_lookup = {}
for model in vc_models.keys():
	voice_lookup[model] = json.load(open(vc_models[model], "r"))
	loaded_models.append(model)

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
	global request_count, last_request_time
	if watchdog:
		watchdog_status(f"count:{request_count}({tts_errors}) len:{two_way_round(avg_request_len, 0, 6)} time:{two_way_round(toms(avg_request_time), 0, 3)}ms rate:{two_way_round(avg_request_rate, 1, 3)}/s last:{two_way_round(time.time() - last_request_time, 1, 3)}s(avg:{two_way_round(avg_request_delay, 2, 4)}s)")
		if timeofdeath > 0 and time.time() > timeofdeath:
			watchdog.notify_error()
			return

		watchdog.notify()
		schedule_watchdog()

def hydrate_models():
	watchdog_status("hydrating models: sending subrequests.")
	with ElapsedFuturesSession(session=request_session) as session:
		futures = []
		for index, model in enumerate(voice_lookup.keys(), start=1):
			speaker_list = voice_lookup[model]
			speaker = random.choice(list(speaker_list.keys()))
			speaker_id = speaker_list[speaker]
			future = session.get(f"http://localhost:{vits_port}/generate-vits?format=numpy", 
				json={
					'voice': speaker,
					'text': "The quick brown fox jumps over the lazy dog",
				}, headers={
					'Accept': 'application/octet-stream',
				}
			)
			future.speaker_id = speaker_id
			future.model_index = index
			future.model = model
			futures.append(future)
		
		for request in futures:
			watchdog_status(f"Hydrating model #{request.model_index}.")
			
			model = request.model
			speaker_id = request.speaker_id
			
			response = request.result()
			
			watchdog_status(f"Hydrating model #{request.model_index}..")
			with io.BytesIO(response.content) as response_content:
				audio = np.load(response_content)
				processed_audio = model.exec(speaker_id, np.expand_dims(audio, axis=0), int(0))
			
			watchdog_status(f"Hydrating model #{request.model_index}...")
			final_audio = processed_audio.detach().cpu().numpy()
			del processed_audio

def test_tts():
	with ElapsedFuturesSession(session=request_session) as session:
		model = random.choice(list(voice_lookup.keys()))
		
		speaker_list = voice_lookup[model]
		speaker = random.choice(list(speaker_list.keys()))
		speaker_id = speaker_list[speaker]
		
		future = session.get(f"http://localhost:{vits_port}/generate-vits?format=numpy", 
			json={
				'voice': speaker,
				'text': "The quick brown fox jumps over the lazy dog",
			}, headers={
				'Accept': 'application/octet-stream',
			}
		)
		
		response = request.result()
		
		with io.BytesIO(response.content) as response_content:
			audio = np.load(response_content)
			processed_audio = model.exec(speaker_id, np.expand_dims(audio, axis=0), int(0))
		
		final_audio = processed_audio.detach().cpu().numpy()
		del processed_audio

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
			tts_log("readyup(): sending ready")
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
	

@app.route("/generate-rvc")
def text_to_speech():
	global request_count, last_request_time, avg_request_time, avg_tts_time, tts_errors, avg_request_len, avg_request_rate, avg_request_delay
	request_count += 1
	tts_errors += 1
	starttime = time.time()
	

	voice = request.args.get("voice", "")
	pitch_adjustment = request.args.get("pitch", 0)
	data = request.data
	
	log_prefix = request.headers.get('X-Log-Prefix', "")
	
	if last_request_time > 0:
		avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
	if avg_request_delay > 0:
		avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
	avg_request_len = mc_avg(avg_request_len, len(data))
	
	print(f"\n{log_prefix} rvc start: {voice=}, {pitch_adjustment=} {len(data)=}")

	if use_voice_name_mapping and voice in voice_name_mapping_reversed:
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
			steptime = time.time()
			
			with io.BytesIO(data) as request_data:
				audio = np.load(request_data)
			
			print(f"{log_prefix} np.load:{time.time()-steptime:f}")
			steptime = time.time()
			processed_audio = model_to_use.exec(speaker_id, np.expand_dims(audio, axis=0), int(pitch_adjustment), log_prefix=log_prefix)
			print(f"{log_prefix} rvc:{time.time()-steptime}")
			
			final_audio = processed_audio.detach().cpu().numpy()
			del processed_audio
			#return 
			audio = AudioSegment(
				final_audio,
				frame_rate=40000,
				sample_width=2,
				channels=1,
			)
			audio1 = np.array(audio.get_array_of_samples())
			audio.export(
				data_bytes,
				format="wav",
			)

			result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")

	tts_errors -= 1
	last_request_time = time.time()
	avg_request_time = mc_avg(avg_request_time, last_request_time-starttime)
	print(f"{log_prefix} rvc total:{time.time()-starttime}")
	result.headers['X-Timing'] = f"{time.time()}"
	return result

@app.route("/tts-voices")
def voices_list():
	'Not Implemented', 501

stop_point = 32768*(random.randint(2,4)*random.random())
@app.route("/health-check")
def tts_health_check():
	global request_count, last_request_time, avg_request_delay, stop_point, timeofdeath, tts
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
	
	return f"OK count:{request_count}({tts_errors}) t:{avg_request_time}s last:{last_request_time}", 200

@app.route("/pitch-available")
def pitch_available():
	return "Pitch available", 200

watchdog_status("Activating.")

threading.Timer(5, readyup).start()
if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5051, threads=6, backlog=32, connection_limit=128, cleanup_interval=1, channel_timeout=10)
