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