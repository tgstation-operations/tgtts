#systemd stuff.
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
watchdog_status("Loading imports")

import os
import signal
import sys
import os.path
import io
import time
import gzip
import gc
import re
from requests_futures.sessions import FuturesSession
from concurrent.futures import as_completed
from collections import OrderedDict
from contextlib import contextmanager
import traceback
import subprocess
#import grequests
import requests
import hashlib
import threading
import sd_notify
import random
import string
import json
import pysbd
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence
from flask import Flask, request, send_file, abort
from urllib.parse import quote as url_quote
import numpy as np
from TTS.tts.models.vits import VitsCharacters
from TTS.tts.configs.vits_config import VitsConfig
from tgtts_utils import ElapsedFuturesSession
import setproctitle
setproctitle.setproctitle(os.path.basename(__file__))

watchdog_status("Loading")

authorization_token = os.getenv("TTS_AUTHORIZATION_TOKEN", "")
cache_url_base = os.getenv("TTS_URL_BASE", "https://tts.tgstation13.org/cache/")
file_store = os.getenv("TTS_FILE_STORE", None)
vits_port = os.getenv("TTS_VITS_PORT", "5040")
rvc_port = os.getenv("TTS_RVC_PORT", "5050")
max_active_subrequests = int(os.getenv("TTS_MAX_SUBREQUESTS", 9))

sample_version = 'v5'
sample_dir = f"./blips_samples/{sample_version}"

radio_starts = ["./sfx/on1.wav", "./sfx/on2.wav"]
radio_ends = ["./sfx/off1.wav", "./sfx/off2.wav", "./sfx/off3.wav", "./sfx/off4.wav"]

segmenter = pysbd.Segmenter(language="en", clean=True)
request_session = requests.session()

req_count = 1
cache_hits = 0
cache_misses = 0

last_request_time = time.time()

tts_stats = {
	'avg_request_time': 0,
	'avg_tts_request_time': 0,
	'avg_ffmpeg_time': 0,
	'failures': 0,
	'downstreamfailures': 0
}

avg_request_len = 0
avg_request_delay = 0
avg_request_rate = 0

watchdog_timer = None

failures = 0
downstreamfailures = 0

watchdog_status("Loading Vits")

letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

#tgvits_config = VitsConfig()
#tgvits_config.load_json("./models/vits/tgvits.config.json")
#tgvits_characters, _ = VitsCharacters.init_from_config(tgvits_config)
#vits.tokenizer.add_blank = False
#vits.tokenizer.use_eos_bos = False
#vits.load_onnx("./models/vits/tgvits.simplified.onnx", True)

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

random_factor = 0.35
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

watchdog_status("Loading Application")

app = Flask(__name__)

def mc_avg(old, new):
	if (old > 0):
		if (req_count > 10):
			if (req_count > 100):
				return (old*0.99) + (new*0.01)
			return (old*0.90) + (new*0.10)
		return (old*0.70) + (new*0.30)
	return new

def two_way_round(number, ndigits = 0, predigits = 0):
	number = round(number, ndigits)
	return (f"%0{predigits}.{ndigits}f") % number

def toms(number):
	return number*1000

def both_stats(key, suffex = '', ndigits = 0, predigits = 0):
	return f"{two_way_round(tts_stats[key], ndigits, predigits)}{suffex}"

def both_stats_ms(key, suffex = '', ndigits = 0, predigits = 0):
	return f"{two_way_round(toms(tts_stats[key]), ndigits, predigits)}{suffex}"

def ping_watchdog():
	global last_request_time
	if watchdog:
		watchdog_status(f"count:{two_way_round(req_count, 0, 5)}(a:{both_stats('failures', '', 0, 3)}, d:{both_stats('downstreamfailures', '', 0, 3)}) len:{two_way_round(avg_request_len, 1, 4)} t:{both_stats_ms('avg_request_time', 'ms', 0, 3)}(r:{both_stats_ms('avg_tts_request_time', 'ms', 0, 3)} f:{both_stats_ms('avg_ffmpeg_time', 'ms', 0, 3)}) {two_way_round(avg_request_rate, 1)}/s {two_way_round(cache_hits / req_count * 100, 2)}% last:{two_way_round(time.time() - last_request_time, 1)}s(avg:{two_way_round(avg_request_delay, 2)}s)")

		if (req_count > 10) and (time.time() > last_request_time+30):
			res = tts_health_check()
			if res[1] == 200:
				last_request_time = time.time()
		if req_count > 65536:
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
	if watchdog:
		watchdog.ready()
		schedule_watchdog()

#This is purely so we can force it to kill the process on context leave
@contextmanager
def ffmpeg_open(args):
	try:
		print(f"ffmpeg args: {json.dumps(args)}")
		process = subprocess.Popen(['ffmpeg', *args], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		yield process
	finally:
		process.stdin.close()
		process.stdout.close()
		process.stderr.close()
		process.poll()
		process.terminate()
		process.kill()
		process.poll()

def do_assert_blips_files(ogvoice, speaker_id, character_set, pitch, log_prefix=''):
	os.makedirs(f"{sample_dir}/speaker_{speaker_id}/pitch_{pitch}/", exist_ok=True)
	with ElapsedFuturesSession(session = request_session, max_workers = max_active_subrequests) as session:
		steptime = time.time()
		request_futures = []
		vits_set = []
		for character in character_set:
			phoneme_id = letters_to_use.index(character)
			if os.path.isfile(f"{sample_dir}/speaker_{speaker_id}/phoneme_{phoneme_id}.npz"):
				continue
			request_future = session.get(
				f"http://localhost:{vits_port}/generate-vits?format=numpy",  
				json = {
					'text': character,
					'voice': ogvoice,
				},
				headers = {
					'Accept': 'application/octet-stream',
					'X-Log-Prefix': log_prefix
				}
			)
			request_future.phoneme_id = phoneme_id
			request_futures.append(request_future)
			vits_set.append(character)
			
		
		for request_future in as_completed(request_futures):
			response = request_future.result()
			with io.BytesIO(response.content) as response_content:
				try:
					## the primary reason we even load this is so we don't save errored data
					vits_data = np.load(response_content)
				except Exception:
					print(f"{log_prefix} unable to load vits_data for {request_future.phoneme_id} from request:")
					print(Exception)
					print(traceback.format_exc())
					continue
			
			np.savez_compressed(f"{sample_dir}/speaker_{speaker_id}/phoneme_{request_future.phoneme_id}.npz", vits_data=vits_data)
		
		if len(vits_set) > 0:
			print(f"{log_prefix} vits time: {time.time() - steptime} {vits_set=}")
		
		
		#now do rvc
		steptime = time.time()
		request_futures = []
		rvc_set = []
		for character in character_set:
			phoneme_id = letters_to_use.index(character)
			vits_file = f"{sample_dir}/speaker_{speaker_id}/phoneme_{phoneme_id}.npz"
			if not os.path.isfile(vits_file):
				print(f"{log_prefix} Missing vits phoneme: `{vits_file}`")
				continue
			
			rvc_file = f"{sample_dir}/speaker_{speaker_id}/pitch_{pitch}/phoneme_{phoneme_id}.wav"
			if os.path.isfile(rvc_file):
				continue
			
			try:
				vits_data = np.load(vits_file)['vits_data']
			except Exception:
				print(f"{log_prefix} unable to load vits_data from file `{vits_file}`:")
				print(Exception)
				print(traceback.format_exc())
				continue
			
			with io.BytesIO() as data_bytes:
				#yes, we load it from disk to just save it to a virtual file
				np.save(data_bytes, vits_data) 
				data_bytes.seek(0)
				request_future = session.get(
					f"http://localhost:{rvc_port}/generate-rvc?pitch={int(pitch)}&voice={url_quote(ogvoice)}",
					data=data_bytes.getvalue(),
					headers={
						'Content-Type': 'application/octet-stream',
						'Accept': 'audio/wav',
						'X-Log-Prefix': log_prefix
					}
				)
				request_future.phoneme_id = phoneme_id
				request_futures.append(request_future)
				rvc_set.append(character)
		
		for request_future in as_completed(request_futures):
			response = request_future.result()
			if not response or response == None:
				print("{log_prefix} error1")
				continue
			if response.status_code != 200:
				print("{log_prefix} error2")
				continue
			with io.BytesIO(response.content) as response_content:
				try:
					output_sound = AudioSegment.from_file(response_content, format="wav")
					silenced_word = strip_silence(output_sound)
					silenced_word.export(f"{sample_dir}/speaker_{speaker_id}/pitch_{pitch}/phoneme_{request_future.phoneme_id}.wav", format="wav")
				except Exception:
					print(f"{log_prefix} unable to load rvc_data for {request_future.phoneme_id} from request:")
					print(Exception)
					print(traceback.format_exc())
					continue
		if len(rvc_set) > 0:
			print(f"{log_prefix} rvc time: {time.time() - steptime} {rvc_set=}")


def text_to_speech_handler(voice, text, filter_complex, pitch, authed, force_regenerate, stats, special_filters):
	global req_count, cache_hits, cache_misses, last_request_time, avg_request_len, avg_request_rate, avg_request_delay
	stats['failures'] += 1
	start_time = time.time()
	
	rand_cap = 9
		
	if filter_complex != "":
		tts_sample_rate = 40000
		filter_complex = filter_complex.replace("%SAMPLE_RATE%", str(tts_sample_rate))
		rand_cap -= 2
	
	if special_filters:
		rand_cap -= 2
	
	if pitch != "0":
		rand_cap -= 2
	
	clean_text = re.sub(r'(\W)(?=\1)', '', text)
	hashtext = f"#v11#blips-characters#{voice.lower()}#{clean_text}#{filter_complex}#{pitch}#{json.dumps(special_filters)}#{random.randint(0, rand_cap)}#{len(clean_text)}#"
	hash = hashlib.sha224(hashtext.encode()).hexdigest().lower()
	
	identifier = f"tts-{hash}"

	log_prefix = f"blips-{hash[0:7]}"

	path_prefix = "./cache"
	if file_store:
		path_prefix = os.path.expanduser(file_store)
	subpath = f"v11/blips/{hash[0:2]}/{hash[2:4]}/{hash[4:6]}/"
	
	path = f"{path_prefix}{subpath}{hash[6:]}"
	
	print(f"{log_prefix} {len(text)=} {len(clean_text)=} {voice=}, {pitch=}, {len(filter_complex)=}, {len(special_filters)=}")
	
	gzip_result = None
	if (file_store) and (not force_regenerate) and (random.randint(0, 9) != 0 or (not authed)):
		#it may seem silly to gzip these, but its really just the way of marking files that have ever had a cache hit
		if os.path.isfile(f"{path}.ogg") and os.path.isfile(f"{path}.len.txt") and os.path.isfile(f"{path}.mp3"):
			print(f"{log_prefix} cache: found first hit\n")
			gzip_result = subprocess.run(["gzip", "-f9", f"{path}.len.txt"], capture_output = True)
			gzip_result = subprocess.run(["gzip", "-f9", f"{path}.mp3"], capture_output = True)
			gzip_result = subprocess.run(["gzip", "-f9", f"{path}.ogg"], capture_output = True)
		
		if os.path.isfile(f"{path}.ogg.gz") and os.path.isfile(f"{path}.mp3.gz") and os.path.isfile(f"{path}.len.txt.gz"):
			avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
			avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
			avg_request_len = mc_avg(avg_request_len, len(clean_text))
			last_request_time = time.time()
			print(f"{log_prefix} cache: found hit")
			cache_hits += 1
			req_count += 1
			length = 0
			if os.path.isfile(f"{path}.len.txt.gz"):
				with gzip.open(f"{path}.len.txt.gz", 'r') as f:
					length = float(f.read())
			if length > 0:
				with gzip.open(f"{path}.ogg.gz", "rb") as gzip_file:
					response = send_file(io.BytesIO(gzip_file.read()), as_attachment=True, download_name=f"{identifier}.ogg", mimetype="audio/ogg")
					
					response.headers['audio-length'] = length
					response.headers['Audio-Length-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.len.txt"
					response.headers['Content-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.ogg"
					response.headers['Mp3-Content-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.mp3"
				stats['failures'] -= 1
				return response
	
	if not authed:
		stats['failures'] -= 1
		abort(401)
	cache_misses += 1
	req_count += 1
	avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
	avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
	
	
	#ffmpeg base arguments. quiet, yes to overwrite, format wav input stdin
	ffmpeg_args = ["-nostats", "-hide_banner", "-loglevel", "warning", "-y", "-f", "wav", "-thread_queue_size", "4096", "-i", "pipe:0"]
	ffmpeg_inputs = 1
	
	if "silicon" in special_filters:
		ffmpeg_args = [*ffmpeg_args, "-thread_queue_size", "4096", "-i", "./sfx/SynthImpulse.wav", "-thread_queue_size", "4096", "-i", "./sfx/RoomImpulse.wav",]
		SynthImpulse_input = ffmpeg_inputs
		RoomImpulse_input = ffmpeg_inputs+1

		ffmpeg_inputs += 2
		if filter_complex != "":
			filter_complex += ","
		filter_complex += f"aresample=44100 [silicon_re_1]; [silicon_re_1] apad=pad_dur=2 [silicon_in_1]; [silicon_in_1] asplit=2 [silicon_in_1_1] [silicon_in_1_2]; [silicon_in_1_1] [{SynthImpulse_input}] afir=dry=10:wet=10 [silicon_reverb_1]; [silicon_in_1_2] [silicon_reverb_1] amix=inputs=2:weights=8 1 [silicon_mix_1]; [silicon_mix_1] asplit=2 [silicon_mix_1_1] [silicon_mix_1_2]; [silicon_mix_1_1] [{RoomImpulse_input}] afir=dry=1:wet=1 [silicon_reverb_2]; [silicon_mix_1_2] [silicon_reverb_2] amix=inputs=2:weights=10 1 [silicon_mix_2]; [silicon_mix_2] equalizer=f=7710:t=q:w=0.6:g=-6,equalizer=f=33:t=q:w=0.44:g=-10 [silicon_out]; [silicon_out] alimiter=level_in=1:level_out=1:limit=0.5:attack=5:release=20:level=disabled"
	
	if "radio" in special_filters:
		radio_start_input = ffmpeg_inputs
		radio_main_input = "0"
		radio_end_input = ffmpeg_inputs + 1
		ffmpeg_args = [*ffmpeg_args, "-thread_queue_size", "4096", "-i", random.choice(radio_starts), "-thread_queue_size", "4096", "-i", random.choice(radio_ends)]
		ffmpeg_inputs += 2
		
		if filter_complex != "":
			filter_complex += " [out_to_radio_filter];"
			radio_main_input = "out_to_radio_filter"
		
		filter_complex += f"[{radio_start_input}:a][{radio_main_input}][{radio_end_input}:a] concat=n=3:v=0:a=1"
		
	
	ogg_path = f"{path}.ogg"
	if not file_store:
		ogg_path = "-"
		
	if filter_complex != "":
		ffmpeg_args = [*ffmpeg_args, "-filter_complex", filter_complex, "-c:a", "libvorbis", "-q:a", "7", "-f", "ogg", ogg_path]
	
	
	else:
		ffmpeg_args = [*ffmpeg_args, "-c:a", "libvorbis", "-q:a", "7", "-f", "ogg", ogg_path]
	
	tts_start = time.time()
	
	final_audio = pydub.AudioSegment.empty()
	ffmpeg_metadata_output = ""
	#open this now so ffmpeg can startup while we send our subrequests and do our work.
	with ffmpeg_open(ffmpeg_args) as ffmpeg_proc:
		ogvoice = voice
		speaker_id = None
		if use_voice_name_mapping and voice in voice_name_mapping_reversed:
			speaker_id = voice_id_mapping[voice_name_mapping_reversed[voice]]
		else:
			speaker_id = voice_id_mapping[voice]
		
		text = text.upper()
		
		character_set = set(text) & set(letters_to_use)
		
		steptime = time.time()
		#phoneme_ids = vits.tokenizer.text_to_ids(text, language="en")
		#phoneme_set = set(phoneme_ids)
		print(f"{log_prefix} {character_set=}")
		
		#phoneme_characters = vits.tokenizer.decode(phoneme_ids)
		#print(f"{phoneme_characters=}")
		
		steptime = time.time()
		try:
			stats['downstreamfailures'] += 1
			do_assert_blips_files(ogvoice, speaker_id, character_set, int(pitch), log_prefix=log_prefix)
			stats['downstreamfailures'] -= 1
		except Exception:
			print(Exception)
			print(traceback.format_exc())
		print(f"{log_prefix} blips.assert:{time.time()-steptime}")
		
		result_sound = None
		new_sound = None
		word_letter_count = 0
		for character in text:
			if character == ' ':
				word_letter_count = 0
				continue
			if character not in letters_to_use:
				continue
						
			if not word_letter_count % 2 == 0:
				word_letter_count += 1
				continue # Skip every other letter
			
			phoneme_id = letters_to_use.index(character)
			
			rvc_file = f"{sample_dir}/speaker_{speaker_id}/pitch_{pitch}/phoneme_{phoneme_id}.wav"
			if not os.path.isfile(rvc_file):
				print(f"{log_prefix} Missing rvc phoneme: {rvc_file}")
				continue
			word_letter_count += 1
			letter_sound = AudioSegment.from_file(rvc_file)

			raw = letter_sound.raw_data
			raw = raw[5000:-5000]
			octaves = 1 + random.random() * random_factor
			frame_rate = int(letter_sound.frame_rate * (2.0 ** octaves))

			new_sound = letter_sound._spawn(raw, overrides={'frame_rate': frame_rate})
			new_sound = new_sound.set_frame_rate(40000)

			result_sound = new_sound if result_sound is None else result_sound + new_sound
		
		with io.BytesIO() as data_bytes:
			result_sound.export(data_bytes, format="wav")

			stats['avg_tts_request_time'] = mc_avg(stats['avg_tts_request_time'], time.time()-tts_start)
			avg_request_len = mc_avg(avg_request_len, len(clean_text))
			
			last_request_time = time.time()

			if file_store:
				os.makedirs(f"{path_prefix}{subpath}", exist_ok=True) 
			ffmpeg_start = time.time()
			ffmpeg_stdout, ffmpeg_stderr = ffmpeg_proc.communicate(data_bytes.read(), timeout = 2)
			
			stats['avg_ffmpeg_time'] = mc_avg(stats['avg_ffmpeg_time'], time.time()-ffmpeg_start)
			ffmpeg_metadata_output = ffmpeg_stderr.decode()
	
			print(f"{log_prefix} ffmpeg time: {time.time()-ffmpeg_start} result size: {len(ffmpeg_stdout)} stderr = \n{ffmpeg_metadata_output}", file=sys.stderr)

	output_file = f"{path}.ogg"
	if not file_store:
		output_file = io.BytesIO(ffmpeg_stdout)
	length = result_sound.duration_seconds

	response = send_file(output_file, as_attachment=True, download_name=f"{identifier}.ogg", mimetype="audio/ogg")
	response.headers['audio-length'] = length
	
	if file_store:
		response.headers['Audio-Length-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.len.txt"
		response.headers['Content-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.ogg"
		with open(f"{path}.len.txt", "w") as f:
			f.write(str(float(length)))
	
	stats['failures'] -= 1
	stats['avg_request_time'] = mc_avg(stats['avg_request_time'], time.time()-start_time)
	
	return response
	return "Unknown error", 500

@app.route("/tts-blips")
def text_to_speech_blips():
	if authorization_token:
		authed = authorization_token == request.headers.get("Authorization", "")
	else:
		authed = True
	print(f"`{authorization_token=}` `{request.headers.get('Authorization', '')=}`")
	voice = request.args.get("voice", '')
	
	text = request.args.get("text", '')
	pitch = request.args.get("pitch", '0')
	special_filters = request.args.get("special_filters", '')
	special_filters = special_filters.split("|")
	silicon = request.args.get("silicon", False)
	if pitch == "":
		pitch = "0"
	
	if pitch == "RANDOM":
		pitch = str(random.randint(-7, 7))
	
	if voice == "RANDOM":
		voice = random.choice(list(json.loads(get_voice_json()[0])))
	
	if text == '':
		text = request.json.get("text", '')

	if silicon:
		special_filters = ["silicon"]
		
	identifier = request.args.get("identifier", '')
	
	filter_complex = request.args.get("filter", '')
	
	force_regenerate = ("force_regenerate" in request.args)
	
	return text_to_speech_handler(voice, text, filter_complex, pitch, authed, force_regenerate, tts_stats, special_filters)

def get_voice_json():
	global voices_json
	if voices_json:
		return voices_json, 200
	response = requests.get(f"http://localhost:{vits_port}/tts-voices")
	if response.status_code != 200:
		return None, response.status_code
	voices_json = response.content
	return response.content, 200

@app.route("/tts-voices")
def voices_list():
	#if authorization_token != request.headers.get("Authorization", ""):
	#	abort(401)
	gc.collect()
	voice_json, statuscode = get_voice_json()
	if statuscode != 200:
		abort(statuscode)
	return voice_json

@app.route("/pitch-available")
def pitch_available():
	response = requests.get(f"http://localhost:{rvc_port}/pitch-available")
	if response.status_code != 200:
		abort(500)
	return "Pitch available", 200

@app.route("/health-check")
def tts_health_check():
	global req_count, cache_hits, cache_misses
	gc.collect()
	if req_count > 65536:
		watchdog.notify_error()
		return f"EXPIRED:", 500
	response = requests.get(f"http://localhost:{vits_port}/health-check")
	response = requests.get(f"http://localhost:{rvc_port}/health-check")
	return f"OK: {req_count}, {cache_hits}, {cache_misses}, {round(cache_hits / req_count * 100, 1)}%\n", 200

threading.Timer(1, readyup).start()

if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5101, threads=16, backlog=8, connection_limit=256, cleanup_interval=1, channel_timeout=15)