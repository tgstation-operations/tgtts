# TG TTS

This is the files for the tg tts server. The directory structure matches where the files deploy to. Only works on python 3.10 (3.9 doesn't have a thing we use and 3.11 added a new syntax rule that invalided a bunch of unmaintained external code.

/home/tts/tgtts has the code files.

You can also look at the systemd unit files in `/etc/systemd/system` to get a feeling for what python files do what.

Pipeline:

ssl request -> nginx (terminate ssl) -> haproxy api port -> tts-api python daemon ->(for each sentence: haproxy vits port -> tts-vits python daemon -> vits tts result as numpy array <- back to tts-api -> haproxy rvc port -> tts-rvc python daemon -> hubert haproxy port | crepe pitch extraction modal (operations done in parallel) -> tts-hubert python daemon -> hubert features extraction model result <- back to tts-rvc -> rvc voice converstion result <- back to tts-api) -> merge sentences -> ffmpeg -> (optional) save ogg to webroot -> return result to nginx.

(The jist is that we use vits to make a basic tts response for a trained voice, and then use rvc trained on that same voice to voice swap from the vits attempt at that voice to the same voice. this second step smoothing over any rough edges. At each step haproxy acts as a load balancer that can also resend requests on certain errors.)


# Requires:

* python === 3.10 (I normally just compile my own with march=native as a user local binary since debian doesn't have a version with 3.10, but this is easier to do at the system level on ubuntu)
* Debian
* systemd (or something that can restart nodes that hit their max requests limit and fail heath checks)
* Nvidea CUDA Drivers
* onnxruntime-gpu being able to find those drivers.
* espeak (apt package
* ffmpeg (compile your own to remove 100ms from response times)

(pr's adding cpu env config support or forms of automatic fallback are welcomed)

# Install  
(you are expected to handle setting up haproxy with the example config given as well as nginx ssl termination via lets encrypt) 

Setup `tgtts` folder somewhere  
Clone https://github.com/yqzhishen/onnxcrepe into `tgtts/onnxcrepe` (so that the repo's readme is located at `tgtts/onnexrepe/readme.md` (production running with commit `ca7e5d7f2dfca5cc4d99e8d546b00793ca4e7157`)  
Download the full onnxcrepe model weights from the releases tab on their repo into `tgtts/onnxcrepe/assets/` (so that `full.onnx` is located at `tgtts/onnxcrepe/assets/full.onnx`)
Unzip the [model files](https://tts.tgstation13.download/cache/models.zip) into `tgtts/models` (so that the structure looks like `tgtts/models/hubert/...`, `tgtts/models/rvc/...`, `tgtts/models/vits/...`)  

Under the user account that will be running python, install the dependancies:  
`python3.10 -m pip install tts`  
`python3.10 -m pip install -r tgtts/requirements.txt`  

(these are done as two seperate steps because of a version conflict on librosa, it will complain about this, you can ignore this.)

Assuming cuda is working correctly (its not, good luck) you can just start the services:

`systemctl enable --now tgtts-api-tg{1..2} tgtts-vits-gpu{1..5} tgtts-rvc-gpu{1..7} tgtts-hubert-gpu{1..2} tgtts-api-blips`.

Localhost Ports:  
(Format is 5<api-id (0 for global)><service (0:api, 4:vits, 5:rvc: 6:hubert 7:bips api)><0:haproxy|1-9:instance-id>)

5100 - api load balancer (haproxy)  
5101 - tgtts-api-tg1  
5102 - tgtts-api-tg2  

5040 - global vits load balancer  
5041 - tgtts-vits-gpu1  
...  
5045 - tgtts-vits-gpu5  
5140 - api-tg vits load balancer (seperate from global above to limit concurrent requests in multi-tenent situations)  

5050 - global rvc load balancer  
5051 - tgtts-rvc-gpu1  
...  
5057 - tgtts-rvc-gpu7  
5150 - api-tg rvc load balancer  

5060 - global hubert load balancer  
5061 - tgtts-hubert-gpu1  
5062 - tgtts-hubert-gpu2  

5070 - blips load balancer  
5071 - tgtts-api-blips  

Two scripts are included in `/root` for monitoring the services:

`/root/status.sh` - shows an abridged system status line for each tts service.  
`/root/stream.sh` - shows a live feed of requests as they are completed with response times over 500 or response codes over 200 highlighted in red. (requires nginx configuration (provided))


# Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making a pr. This is not a python project and trying to turn it into one will get you banned from the repo. 

# License

This repo contains random bits of code compatible with an agpl license (mit and mpl mostly). Thus, the entire repo is agpl licensed but this does not infect the specific lines of code ~~stolen~~ copyed from other licensed repos.
