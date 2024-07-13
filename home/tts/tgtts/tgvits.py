import numpy as np
import torch
import onnxruntime as ort
from TTS.tts.models.vits import Vits
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from typing import Dict, List, Tuple, Union

class TgVits(Vits):
    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        if not config.model_args.encoder_sample_rate:
            assert (
                upsample_rate == config.audio.hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert (
                upsample_rate == effective_hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path, config.model_args.speaker_encoder_config_path
            )
        return TgVits(new_config, ap, tokenizer, speaker_manager, language_manager)
		
    def load_onnx(self, model_path: str, cuda=False):
        #providers = ["CPUExecutionProvider" if cuda is False else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
        tg_tts_providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'DEFAULT',
                'cudnn_conv_use_max_workspace': '1',
            }),
            'CPUExecutionProvider',
        ]
        tg_tts_sess_options = ort.SessionOptions()
        tg_tts_sess_options.enable_mem_pattern = False
        tg_tts_sess_options.enable_cpu_mem_arena = False
        tg_tts_sess_options.enable_mem_reuse = False
        
        self.onnx_sess = ort.InferenceSession(
            model_path,
            sess_options=tg_tts_sess_options,
            providers=tg_tts_providers,
        )

    def inference_onnx(self, x, x_lengths=None, speaker_id=None):
        """ONNX inference
        """

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
        tg_tts_run_options = ort.RunOptions()
        tg_tts_run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
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