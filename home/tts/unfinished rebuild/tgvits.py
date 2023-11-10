import numpy as np
import torch
from TTS.tts.models.vits import Vits
import onnxruntime as ort
from .common import *

class tgVits(Vits):

	def load_onnx(self, model_path: str, cuda=False):
		#providers = ["CPUExecutionProvider" if cuda is False else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
		
		self.onnx_sess = ort.InferenceSession(
			model_path,
			sess_options=tg_tts_sess_options,
			providers=tg_tts_providers,
		)

	def inference_onnx(self, x, x_lengths=None, speaker_id=None):
		"""ONNX inference
		"""
		import onnxruntime as ort

		if isinstance(x, torch.Tensor):
			x = x.cpu().numpy()

		if x_lengths is None:
			x_lengths = np.array([x.shape[1]], dtype=np.int64)

		if isinstance(x_lengths, torch.Tensor):
			x_lengths = x_lengths.cpu().numpy()
		scales = np.array(
			[self.inference_noise_scale, self.length_scale, self.inference_noise_scale_dp],
			dtype=np.float32,
		)
		
		audio = self.onnx_sess.run(
			["output"],
			{
				"input": x,
				"input_lengths": x_lengths,
				"scales": scales,
				"sid": torch.tensor([speaker_id]).cpu().numpy(),
			},
			tg_tts_run_options
		)
		return audio[0][0]

