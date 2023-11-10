# TG TTS

This is the files for the tg tts server. The directory structure matches where the files deploy to. Last tested on python 3.8.

A fair chunk of these are unused. Look at the systemd unit files to get a feeling for what python files do what.

pipeline:

ssl request -> nginx (terminate ssl) -> haproxy api port -> tts-api python daemon -> haproxy gpu|cpu port -> tts-gpu|cpu python daemon (optionally -> haproxy gpu port for generating blips samples).

cpu nodes are used for blips.

venv is not used.

Some site-packages had to be modified, the modified files are included.

An unfinished attempt to copy-edit the pipeline to use object overrides and function reference patching instead of modifying site-packages is included in `/home/tts/unfinished rebuild`

# Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making a pr. This is not a python project and trying to turn it into one will get you banned from the repo. 

# License

This repo contains random bits of code compatible with an agpl license (mit and mpl mostly). Thus, the entire repo is agpl licensed but this does not infect the specific lines of code ~~stolen~~ copyed from agpl compatible repos.