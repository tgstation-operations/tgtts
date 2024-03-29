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
from contextlib import contextmanager
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



app = Flask(__name__)

authorization_token = os.getenv("TTS_AUTHORIZATION_TOKEN", "vote_goof_2024")
cache_url_base = os.getenv("TTS_URL_BASE", "https://tts.tgstation13.org/cache/")
tts_port = os.getenv("TTS_GPU_PORT", "5010")
blips_port = os.getenv("TTS_BLIPS_PORT", "5020")
max_active_subrequests = int(os.getenv("TTS_MAX_SUBREQUESTS", 2))


voices_json = None

radio_starts = ["./on1.wav", "./on2.wav"]
radio_ends = ["./off1.wav", "./off2.wav", "./off3.wav", "./off4.wav"]
segmenter = pysbd.Segmenter(language="en", clean=True)

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

blips_stats = {
	'avg_request_time': 0,
	'avg_tts_request_time': 0,
	'avg_ffmpeg_time': 0,
	'failures': 0,
	'downstreamfailures': 0
}

avg_request_len = 0
avg_request_delay = 0
avg_request_rate = 0

failures = 0
downstreamfailures = 0

#systemd stuff.
watchdog_timer = None
watchdog = sd_notify.Notifier()
if not watchdog.enabled(): 
	watchdog = None

def mc_avg(old, new):
	if (old > 0):
		return (old*0.99) + (new*0.01)
	return new

def two_way_round(number, ndigits = 0):
	number = round(number, ndigits)
	return (f"%0.{ndigits}f") % number

def both_stats(key, suffex = '', ndigits = 0):
	return f"({two_way_round(tts_stats[key], ndigits)}{suffex}/{two_way_round(blips_stats[key], ndigits)}{suffex})"

def ping_watchdog():
	global last_request_time
	if watchdog:
		watchdog.status(f"req:{req_count}(a:{both_stats('failures')}, t:{both_stats('downstreamfailures')}) len:{two_way_round(avg_request_len, 1)} T:[r:{both_stats('avg_tts_request_time', 's', 4)}, f:{both_stats('avg_ffmpeg_time', 's', 3)}, t:{both_stats('avg_request_time', 's', 3)}] r:{two_way_round(avg_request_rate, 1)}/s hit:{two_way_round(cache_hits / req_count * 100, 1)}% last:{two_way_round(time.time() - last_request_time, 1)}s")

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
		watchdog_timer = threading.Timer(0.1, ping_watchdog)
		watchdog_timer.start()

def readyup():
	if watchdog:
		watchdog.ready()
		schedule_watchdog()

#This is purely so we can force it to kill the process on context leave
@contextmanager
def ffmpeg_open(args):
	try:
		process = subprocess.Popen(['ffmpeg', *args], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		yield process
	finally:
		process.stdin.close()
		process.stdout.close()
		process.stderr.close()
		process.poll()
		process.terminate()
		process.poll()
		process.kill()
		process.poll()

def text_to_speech_handler(endpoint, voice, text, filter_complex, pitch, authed, force_regenerate, stats, special_filters, port=5010):
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
	hash = hashlib.sha224(f"#v7#{endpoint}#{voice.lower()}#{clean_text}#{filter_complex}#{pitch}#{json.dumps(special_filters)}#{random.randint(0, rand_cap)}#{len(clean_text)}#".encode()).hexdigest().lower()
	
	identifier = f"tts-{hash}"

	path_prefix = ".local/tts_gen_cache/"
	subpath = f"v7/{hash[0:2]}/{hash[2:4]}/{hash[4:6]}/"
	
	path = f"{path_prefix}{subpath}{hash[6:]}"
	
	print(f"checking cache\n")
	gzip_result = None
	if (not force_regenerate) and (random.randint(0, 9) != 0 or (not authed)):
		if os.path.isfile(f"{path}.ogg") and os.path.isfile(f"{path}.len.txt"):
			with open('firstcachehits.txt', 'a') as f:
				f.write(f"firstcachehit: {clean_text}\n")
			print(f"checking cache: found\n")
			gzip_result = subprocess.run(["gzip", "-f9", f"{path}.ogg"], capture_output = True)
			print(f"gzip result size: {len(gzip_result.stdout)} stderr = \n{gzip_result.stderr.decode()}")
		
		print(f"checking cache2\n")
		if os.path.isfile(f"{path}.ogg.gz") and os.path.isfile(f"{path}.mp3") and os.path.isfile(f"{path}.len.txt"):
			avg_request_delay = mc_avg(avg_request_delay, time.time()-last_request_time)
			avg_request_rate = mc_avg(avg_request_rate, 1/max(0.0000001, avg_request_delay))
			avg_request_len = mc_avg(avg_request_len, len(clean_text))
			last_request_time = time.time()
			print(f"checking cache2: found\n")
			cache_hits += 1
			req_count += 1
			length = 0
			with open('cachehits.txt', 'a') as f:
				f.write(f"cachehit: {clean_text}\n")
			if os.path.isfile(f"{path}.len.txt"):
				with open(f"{path}.len.txt", 'r') as f:
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
	stats['downstreamfailures'] += 1
	
	#ffmpeg base arguments. quiet, yes to overwrite, format wav input stdin
	ffmpeg_args = ["-nostats", "-hide_banner", "-loglevel", "warning", "-y", "-f", "wav", "-i", "pipe:0"]
	ffmpeg_inputs = 1
	
	if "silicon" in special_filters:
		ffmpeg_args = [*ffmpeg_args, "-i", "./SynthImpulse.wav", "-i", "./RoomImpulse.wav",]
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
		ffmpeg_args = [*ffmpeg_args, "-i", random.choice(radio_starts), "-i", random.choice(radio_ends)]
		ffmpeg_inputs += 2
		
		if filter_complex != "":
			filter_complex += " [out_to_radio_filter];"
			radio_main_input = "out_to_radio_filter"
		
		filter_complex += f"[{radio_start_input}:a][{radio_main_input}][{radio_end_input}:a] concat=n=3:v=0:a=1"
		
	
	if filter_complex != "":
		filter_complex += ",asplit=2[ogg][mp3]"
		ffmpeg_args = [*ffmpeg_args, "-filter_complex", filter_complex, "-map", "[ogg]", "-c:a", "libvorbis", "-q:a", "7", "-f", "ogg", f"{path}.ogg", "-map", "[mp3]", "-c:a", "libmp3lame", "-q:a", "1", "-f", "mp3", f"{path}.mp3"]
	
	
	else:
		ffmpeg_args = [*ffmpeg_args, "-c:a", "libvorbis", "-q:a", "7", "-f", "ogg", f"{path}.ogg", "-c:a", "libmp3lame", "-q:a", "1", "-f", "mp3", f"{path}.mp3"]
	
	tts_start = time.time()
	
	final_audio = pydub.AudioSegment.empty()
	ffmpeg_metadata_output = ""
	#open this now so ffmpeg can startup while we send our subrequests.
	with ffmpeg_open(ffmpeg_args) as ffmpeg_proc:
		with FuturesSession(max_workers = max_active_subrequests) as session:
			response_futures = []
			for sentence in segmenter.segment(clean_text):
				if len(''.join(ch for ch in sentence if ch not in string.punctuation)) < 1:
					continue
				steptime = time.time()
				response_futures.append(session.get(f"http://localhost:{port}/{endpoint}", json={ 'text': sentence, 'voice': voice, 'pitch': pitch }))
				print(f"subrequest time: {time.time() - steptime}")
			
			
			
			responses = []
			#not using as_completed() for order preserving reasons
			steptime = time.time()
			for response_future in response_futures: 
				responses.append(response_future.result()) 
			print(f"subresponse time: {time.time() - steptime}")
			
			for response in responses:
				if not response or response == None:
					print("error1")
					stats['failures'] -= 1
					abort(502)
				if response.status_code != 200:
					print("error2")
					stats['failures'] -= 1
					abort(response.status_code)
				
				sentence_audio = pydub.AudioSegment.from_file(io.BytesIO(response.content), "wav")
				
				sentence_silence = pydub.AudioSegment.silent(random.randint(200, 300), 40000)
				sentence_audio += sentence_silence
				final_audio += sentence_audio
				# ""Goldman-Eisler (1968) determined that typical speakers paused for an average of 250 milliseconds (ms), with a range from 150 to 400 ms.""
				# (https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=10153&context=etd)
			
		with io.BytesIO() as data_bytes:
			final_audio.export(data_bytes, format="wav")
			
			stats['avg_tts_request_time'] = mc_avg(stats['avg_tts_request_time'], time.time()-tts_start)
			avg_request_len = mc_avg(avg_request_len, len(clean_text))
			
			last_request_time = time.time()
			if response.status_code != 200:
				stats['failures'] -= 1
				abort(500)
			stats['downstreamfailures'] -= 1
			os.makedirs(f"{path_prefix}{subpath}", exist_ok=True) 
			ffmpeg_start = time.time()
			ffmpeg_stdout, ffmpeg_stderr = ffmpeg_proc.communicate(data_bytes.read(), timeout = 2)
			
			stats['avg_ffmpeg_time'] = mc_avg(stats['avg_ffmpeg_time'], time.time()-ffmpeg_start)
			ffmpeg_metadata_output = ffmpeg_stderr.decode()
	
			print(f"ffmpeg time: {time.time()-ffmpeg_start} result size: {len(ffmpeg_stdout)} stderr = \n{ffmpeg_metadata_output}", file=sys.stderr)

	length = final_audio.duration_seconds
	
	with open(f"{path}.len.txt", "w") as f:
		f.write(str(float(length)))
	
	response = send_file(f"{path}.ogg", as_attachment=True, download_name=f"{identifier}.ogg", mimetype="audio/ogg")
	response.headers['audio-length'] = length
	response.headers['Audio-Length-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.len.txt"
	response.headers['Content-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.ogg"
	response.headers['Mp3-Content-Location'] = f"{cache_url_base}{subpath}{hash[6:]}.mp3"
	
	stats['failures'] -= 1
	stats['avg_request_time'] = mc_avg(stats['avg_request_time'], time.time()-start_time)
	
	return response
	return "Unknown error", 500

@app.route("/tts")
def text_to_speech_normal():
	authed = authorization_token == request.headers.get("Authorization", "")

	voice = request.args.get("voice", '')
	
	text = request.args.get("text", '')
	pitch = request.args.get("pitch", '0')
	special_filters = request.args.get("special_filters", '')
	special_filters = special_filters.split("|")
	silicon = request.args.get("silicon", False)
	if pitch == "":
		pitch = "0"
	
	if text == '':
		text = request.json.get("text", '')

	if silicon:
		special_filters = ["silicon"]
	
	identifier = request.args.get("identifier", '')
	
	filter_complex = request.args.get("filter", '')
	
	force_regenerate = ("force_regenerate" in request.args)
	
	return text_to_speech_handler("generate-tts", voice, text, filter_complex, pitch, authed, force_regenerate, tts_stats, special_filters, tts_port)

@app.route("/tts-blips")
def text_to_speech_blips():
	authed = authorization_token == request.headers.get("Authorization", "")

	voice = request.args.get("voice", '')
	
	text = request.args.get("text", '')
	pitch = request.args.get("pitch", '0')
	special_filters = request.args.get("special_filters", '')
	special_filters = special_filters.split("|")
	silicon = request.args.get("silicon", False)
	if pitch == "":
		pitch = "0"
	
	if text == '':
		text = request.json.get("text", '')

	if silicon:
		special_filters = ["silicon"]
		
	identifier = request.args.get("identifier", '')
	
	filter_complex = request.args.get("filter", '')
	
	force_regenerate = ("force_regenerate" in request.args)
	
	return text_to_speech_handler("generate-tts-blips", voice, text, filter_complex, pitch, authed, force_regenerate, blips_stats, special_filters, blips_port)

@app.route("/tts-voices")
def voices_list():
	global voices_json
	#if authorization_token != request.headers.get("Authorization", ""):
	#	abort(401)
	gc.collect()
	if voices_json:
		return voices_json
	response = requests.get(f"http://localhost:{tts_port}/tts-voices")
	if response.status_code != 200:
		abort(response.status_code)
	voices_json = response.content
	return response.content

@app.route("/pitch-available")
def pitch_available():
	response = requests.get(f"http://localhost:{tts_port}/pitch-available")
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
	response = requests.get(f"http://localhost:{tts_port}/health-check")
	return f"OK: {req_count}, {cache_hits}, {cache_misses}, {round(cache_hits / req_count * 100, 1)}%\n", 200

threading.Timer(1, readyup).start()

if __name__ == "__main__":
	if os.getenv('TTS_LD_LIBRARY_PATH', "") != "":
		os.putenv('LD_LIBRARY_PATH', os.getenv('TTS_LD_LIBRARY_PATH'))
	from waitress import serve
	serve(app, host="localhost", port=5101, threads=16, backlog=8, connection_limit=256, cleanup_interval=1, channel_timeout=15)