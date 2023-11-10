import torch
from TTS.api import TTS
import os
import signal
import sys
import io
import gc
import json
import subprocess
import time
import random
import uuid
import cProfile

USE_GPU = True

def text_to_speech(text, tts, profile = False):
	print(f"\n\nTTS for `{text}`")
	tts_timing_info = {}
	start_time = time.time()
	voice = random.choice(tts.speakers)
	identifier = str(uuid.uuid4())
	with io.BytesIO() as wav_bytes:
		
		with torch.inference_mode():
			#tts_timing_info["setup"] = time.time() - start_time
			start_time = time.time()
			if profile:
				cProfile.runctx('tts.tts_to_file(text=text, speaker=voice, file_path=wav_bytes)', globals(), locals(), filename=profile)
			else:
				tts.tts_to_file(text=text, speaker=voice, file_path=wav_bytes)
		tts_timing_info["tts"] = time.time() - start_time
		#start_time = time.time()
	#gc.collect()
	#tts_timing_info["gc.collect() time:"] = time.time() - start_time
	return tts_timing_info["tts"]


if __name__ == "__main__":
	timing_data = {}
	timing_data["USE_GPU"] = USE_GPU
	
	timing_data["Inputs"] = {
		"Hello World": "Hello World",
		"Quick Brown Fox": "The quick brown fox jumps over the lazy dog.",
		"Bee Movie Intro": "According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible.",
		"Darth Plagueis": "Did you ever hear the tragedy of Darth Plagueis \"the wise\"? I thought not. It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life... He had such a knowledge of the dark side that he could even keep the ones he cared about from dying. The dark side of the Force is a pathway to many abilities some consider to be unnatural. He became so powerful... the only thing he was afraid of was losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his apprentice killed him in his sleep. It's ironic he could save others from death, but not himself."
	}
	
	
	run_tts = None
	with torch.inference_mode():
		run_tts = TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=USE_GPU)
		run_tts.synthesizer.tts_model.half()
		#run_tts = TTS(model_path="./tts_data/model.pth", config_path="./tts_data/config.json", progress_bar=False, gpu=USE_GPU)
	
	#preload the shit.
	text_to_speech("Did you ever hear the tragedy of Darth Plagueis \"the wise\"? I thought not. It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life... He had such a knowledge of the dark side that he could even keep the ones he cared about from dying. The dark side of the Force is a pathway to many abilities some consider to be unnatural. He became so powerful... the only thing he was afraid of was losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his apprentice killed him in his sleep. It's ironic he could save others from death, but not himself.", run_tts)
	
	run_data = {}
	run_start = time.time()
	start = time.time()
	res = text_to_speech("Hello World.", run_tts, profile = 'ttsprofile.hello')
	res = text_to_speech("The quick brown fox jumps over the lazy dog.", run_tts, profile = 'ttsprofile.fox')
	res = text_to_speech("According to all known laws of aviation, there is no way a bee should be able to fly Its wings are too small to get its fat little body off the ground The bee, of course, flies anyway because bees don't care what humans think is impossible", run_tts, profile = 'ttsprofile.bee')
	
	run_data["Hello World"] = res
	
	print(json.dumps(timing_data, indent=4))
	