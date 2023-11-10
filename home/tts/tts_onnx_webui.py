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
# CLASSES BORROWED AND MODIFIED FROM https://github.com/w-okada/voice-changer

class PitchExtractor():
	def extract(self, audio, f0_up_key, sr, window, silence_front=0):
		return "lol"

class CrepePitchExtractor(PitchExtractor):

	def __init__(self):
		self.session = CrepeInferenceSession(
			model='full',
			sess_options = ort.SessionOptions(),
			providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})])

	def extract(self, audio, f0_up_key, sr, window, silence_front=0):
		n_frames = int(len(audio) // window) + 1
		start_frame = int(silence_front * sr / window)
		real_silence_front = start_frame * window / sr

		silence_front_offset = int(np.round(real_silence_front * sr))
		audio = audio[silence_front_offset:]

		f0_min = 50
		f0_max = 1100
		f0_mel_min = 1127 * np.log(1 + f0_min / 700)
		f0_mel_max = 1127 * np.log(1 + f0_max / 700)
		f0, pd = onnxcrepe.predict(self.session, audio.astype(np.float32), sr, fmin=f0_min, fmax=f0_max, batch_size=512, decoder=onnxcrepe.decode.weighted_argmax, return_periodicity=True)
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

		return f0_coarse, f0bak

class OnnxRVCInferencer():
	def loadModel(self, file: str):
		providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
		sess_options = ort.SessionOptions()
		self.onnx_sess = ort.InferenceSession(
			file,
			sess_options=sess_options,
			providers=providers,
		)
		# check half-precision
		first_input_type = self.onnx_sess.get_inputs()[0].type
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
			)

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
			"source": feats,
			"padding_mask": padding_mask,
			"output_layer": embOutputLayer,  # 9 or 12
		}
		with torch.no_grad():
			logits = self.model.extract_features(**inputs)
			if useFinalProj:
				feats = self.model.final_proj(logits[0]).detach()
				del logits
				torch.cuda.empty_cache()
			else:
				feats = logits[0].detach()
				del logits
				torch.cuda.empty_cache()
		del padding_mask
		torch.cuda.empty_cache()
		return feats
	
class FairseqHubertOnnx(Embedder):
	def loadModel(self, file: str, dev: device = "cuda:0", isHalf: bool = True) -> Embedder:
		providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
		sess_options = ort.SessionOptions()
		self.onnx_sess = ort.InferenceSession(
			file,
			sess_options=sess_options,
			providers=providers,
		)

	def extractFeatures(
		self, feats: torch.Tensor
	) -> torch.Tensor:
		feats_out = self.onnx_sess.run(
			["logits"],
			{
				"feats": feats.detach().cpu().numpy(),
			},
		)
		return torch.tensor(np.array(feats_out))

class FairseqContentvec(FairseqHubert):
	def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
		super().loadModel(file, dev, isHalf)
		super().setProps(file, dev, isHalf)
		return self

class Pipeline(object):
	embedder: Embedder
	inferencer: OnnxRVCInferencer
	pitchExtractor: PitchExtractor

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
		pitchExtractor: PitchExtractor,
		# feature: Any | None,
		targetSR,
		device,
		isHalf,
	):
		self.embedder = embedder
		self.inferencer = inferencer
		self.pitchExtractor = pitchExtractor

		# self.feature = feature

		self.targetSR = targetSR
		self.device = device
		self.isHalf = isHalf

		if isinstance(device, str):
			device = torch.device(device)
		if device.type == "cuda":
			vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
		else:
			vram = None

		if vram is not None and vram <= 4:
			print("BITCHES")
			self.x_pad = 1
			self.x_query = 5
			self.x_center = 30
			self.x_max = 32
		elif vram is not None and vram <= 5:
			print("BIG BITCHES")
			self.x_pad = 1
			self.x_query = 6
			self.x_center = 38
			self.x_max = 41
		else:
			print("no BITCHES")
			self.x_pad = 3
			self.x_query = 10
			self.x_center = 60
			self.x_max = 65

		self.sr = 16000  # hubert input sample rate
		self.window = 160  # hubert input window
		self.t_pad = self.sr * self.x_pad  # padding time for each utterance
		self.t_pad_tgt = self.targetSR * self.x_pad
		self.t_pad2 = self.t_pad * 2
		self.t_query = self.sr * self.x_query  # query time before and after query point
		self.t_center = self.sr * self.x_center  # query cut point position
		self.t_max = self.sr * self.x_max  # max time for no query

	def exec(
		self,
		sid,
		audio,
		f0_up_key,
		if_f0 = True,
		silence_front = 0,
		repeat = 0,
	):
		
		audio = np.squeeze(audio)
		bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
		audio = signal.filtfilt(bh, ah, audio)
		audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
		opt_ts = []
		if audio_pad.shape[0] > self.t_max:
			audio_sum = np.zeros_like(audio)
			for i in range(self.window):
				audio_sum += audio_pad[i : i - self.window]
			for t in range(self.t_center, audio.shape[0], self.t_center):
				opt_ts.append(
					t
					- self.t_query
					+ np.where(
						np.abs(audio_sum[t - self.t_query : t + self.t_query])
						== np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
					)[0][0]
				)

		audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
		p_len = audio_pad.shape[0] // self.window
		sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
		pitch, pitchf = self.pitchExtractor.extract(
				audio_pad,
				f0_up_key,
				self.sr,
				self.window,
				silence_front=silence_front,
		)
		pitch = pitch[:p_len]
		pitchf = pitchf[:p_len]
		pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
		pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

		audio_opt = []

		s = 0
		t = None
		for t in opt_ts:
			t = t // self.window * self.window
			audio_opt.append(
				self._convert(
					sid,
					audio_pad[s : t + self.t_pad2 + self.window],
					pitch[:, s // self.window : (t + self.t_pad2) // self.window],
					pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
				)[self.t_pad_tgt : -self.t_pad_tgt]
			)
			s = t
		audio_opt.append(
			self._convert(
				sid,
				audio_pad[t:],
				pitch[:, t // self.window :] if t is not None else pitch,
				pitchf[:, t // self.window :] if t is not None else pitchf,
			)[self.t_pad_tgt : -self.t_pad_tgt]
		)
		audio_opt = np.concatenate(audio_opt)
		del pitch, pitchf, sid
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		return audio_opt

	def _convert(
		self,
		sid: int,
		audio: np.ndarray,
		pitch: np.ndarray,
		pitchf: np.ndarray,
	):
		feats = torch.from_numpy(audio)
		feats = feats.float()
		if feats.dim() == 2:  # double channels
			feats = feats.mean(-1)
		assert feats.dim() == 1, feats.dim()
		feats = feats.view(1, -1)

		with torch.no_grad():
			feats = self.embedder.extractFeatures(feats)

		feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

		p_len = audio.shape[0] // self.window
		if feats.shape[1] < p_len:
			p_len = feats.shape[1]
			if pitch != None and pitchf != None:
				pitch = pitch[:, :p_len]
				pitchf = pitchf[:, :p_len]
		p_len = torch.tensor([p_len], device=self.device).long()
		with torch.no_grad():
			audio1 = (
				(self.inferencer.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768)
				.data.cpu()
				.float()
				.numpy()
				.astype(np.int16)
			)
		del feats, p_len
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		return audio1

print("Loading RVC models")
rvc_1 = OnnxRVCInferencer()
rvc_2 = OnnxRVCInferencer()
rvc_3 = OnnxRVCInferencer()
rvc_1.loadModel("TGStation_Crepe_1_simple.onnx")
rvc_2.loadModel("TGStation_Crepe_2_simple.onnx")
rvc_3.loadModel("TGStation_Crepe_3_simple.onnx")
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
print("Loaded")
config = VitsConfig()
config.load_json("E:/model_output_2/config.json")
vits = Vits.init_from_config(config)
#vits.load_checkpoint(config,  "E:/model_output_2/checkpoint_1157000.pth")

#vits.export_onnx("tgstation_onnx.onnx")
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
