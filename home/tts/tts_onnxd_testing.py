import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'rvc_webui')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxcrepe')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxcrepe', 'onnxcrepe')))

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

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
from julius.resample import ResampleFrac
import threading

resample_function = ResampleFrac(22050, 16000) #compile a gpu kernal for a given resample fraction
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

# CLASSES BORROWED AND MODIFIED FROM https://github.com/w-okada/voice-changer

class CrepePitchExtractor():
	def __init__(self):
		self.gpulock = threading.Lock()
		self.session = CrepeInferenceSession(
			model='full',
			sess_options = tg_tts_sess_options,
			providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})])

	def extract(self, audio, f0_up_key, sr, window, silence_front=0):
		with self.gpulock:
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

			print(f"rvc.crepe.onnx:{time.time()-steptime}")
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
	) -> torch.Tensor:
		with self.gpulock:
			steptime = time.time()
			if pitch is None or pitchf is None:
				raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")
			if self.isHalf:
				audio1 = self.model.run(
					["audio"],
					{
						"feats": feats.detach().cpu().numpy().astype(np.float16),
						"p_len": pitch_length.detach().cpu().numpy().astype(np.int64),
						"pitch": pitch.detach().cpu().numpy().astype(np.int64),
						"pitchf": pitchf.detach().cpu().numpy().astype(np.float32),
						"sid": sid.detach().cpu().numpy().astype(np.int64),
					},
					tg_tts_run_options
				)
			else:
				audio1 = self.model.run(
					["audio"],
					{
						"feats": feats.detach().cpu().numpy().astype(np.float32),
						"p_len": pitch_length.detach().cpu().numpy().astype(np.int64),
						"pitch": pitch.detach().cpu().numpy().astype(np.int64),
						"pitchf": pitchf.detach().cpu().numpy().astype(np.float32),
						"sid": sid.detach().cpu().numpy().astype(np.int64),
					},
					tg_tts_run_options
				)
			#print(f"{type(audio1)=} {type(feats)=} {type(pitch_length)=} {type(pitch)=} {type(pitchf)=} {type(sid)=}")
			print(f"rvc.main.onnx:{time.time()-steptime}")
			
			return torch.tensor(np.array(audio1))

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
		self, feats: torch.Tensor
	) -> torch.Tensor:
		with self.gpulock:
			steptime = time.time()
			
			feats_out = self.onnx_sess.run(
				["logits"],
				{
					"feats": feats.detach().cpu().numpy(),
				},
				tg_tts_run_options
			)
			print(f"rvc.hubert.onnx:{time.time()-steptime}")
			return torch.tensor(np.array(feats_out))

class FairseqHubert(Embedder):
	def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
		super().setProps(file, dev, isHalf)

		models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
			[file],
			suffix="",
		)
		model = models[0]
		model.eval()

		model = model.to(dev)
		if isHalf:
			model = model.half()

		self.model = model
		return self

	def extractFeatures(
		self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
	) -> torch.Tensor:
		padding_mask = torch.BoolTensor(feats.shape).to(self.dev).fill_(False)
		inputs = {
			"source": feats.to(self.dev),
			"padding_mask": padding_mask,
			"output_layer": embOutputLayer,  # 9 or 12
		}

		with torch.no_grad():
			logits = self.model.extract_features(**inputs)
			if useFinalProj:
				feats = self.model.final_proj(logits[0])
			else:
				feats = logits[0]
		return feats

class FairseqContentvec(FairseqHubert):
	def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
		super().loadModel(file, dev, isHalf)
		super().setProps(file, dev, isHalf)
		return self

class Pipeline(object):
	embedder: Embedder
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
		embedder: Embedder,
		inferencer: OnnxRVCInferencer,
		pitchExtractor: CrepePitchExtractor,
		# feature: Any | None,
		targetSR,
		device,
		isHalf,
	):
		self.gpulock = threading.Lock()
		self.embedder = embedder
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
	):
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
			)
			#with self.gpulock:
			pitch = pitch[:p_len]
			pitchf = pitchf[:p_len]
			pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
			pitchf = torch.tensor(pitchf, device=self.device, dtype=torch.float).unsqueeze(0)
		# tensor型調整
		#with self.gpulock:
		feats = audio_pad
		if feats.dim() == 2:  # double channels
			feats = feats.mean(-1)
		assert feats.dim() == 1, feats.dim()
		feats = feats.view(1, -1)
		# embedding
		with autocast(enabled=self.isHalf):
			try:
				feats = self.embedder.extractFeatures(feats)
			except RuntimeError as e:
				raise e
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

		# 推論実行
		try:
			with torch.no_grad():
				with autocast(enabled=self.isHalf):
					audio1 = self.inferencer.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
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
