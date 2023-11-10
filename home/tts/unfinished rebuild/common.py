import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'rvc_webui')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxcrepe')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxcrepe', 'onnxcrepe')))

tg_tts_providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'DEFAULT',
		'cudnn_conv_use_max_workspace': '1',
    }),
    #'CPUExecutionProvider',
]
tg_tts_sess_options = ort.SessionOptions()
tg_tts_sess_options.enable_mem_pattern = False
tg_tts_sess_options.enable_cpu_mem_arena = False
tg_tts_sess_options.enable_mem_reuse = False

tg_tts_run_options = ort.RunOptions()
tg_tts_run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
