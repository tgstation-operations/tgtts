 tail -f /var/log/nginx/stream.log|grep --color=always -P '(?:[13456789][0-9][0-9] (?=len=)|[0-9]?[1-9]+.[0-9]{3}s |[0-9]?[0-9]+.[5-9][0-9]{2}s |(?<=200) (?=len=))(?=.*"GET.*/tts")'