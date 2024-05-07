from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import time
import subprocess
import torch

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=10)  # generate 10 seconds.
all_descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor', 'dog barking and cat meowing']

EXP_DURATION = 10 * 60
def current_time_millis():
    return round(time.time() * 1000)

def current_time_secs():
    return round(time.time())

def get_gpu_memory():
    cmd = "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    labels = ['used=','total=','free=']
    output_str = result.stdout.decode('utf-8')
    output_str = output_str.split('\n')[2]
    parts = output_str.split(',')
    for idx, part in enumerate(parts):
        parts[idx] = '{}{}'.format(labels[idx], part.strip())
    return ','.join(parts)

def timed_execution(b):
    # 4, 8, 12, 16, 24, 32
    start_time = current_time_secs()
    curr_time = start_time
    total_generated = 0

    while curr_time - start_time < EXP_DURATION:
        loop_start_time = current_time_secs()
        print('Generating batch size: {}'.format(b))
        descriptions = [all_descriptions[i % len(all_descriptions)] for i in range(b)]
        torch.cuda.empty_cache()
        wav = model.generate(descriptions, progress=True)  # generates 3 samples.
        total_generated += len(descriptions)
        curr_time = current_time_secs()
        print('Memory: {}'.format(get_gpu_memory()))
        elapsed_time = curr_time - loop_start_time
        elapsed_time_min = elapsed_time / 60.0
        throughput = total_generated / elapsed_time_min
        print('Number generated: {}, {}, time taken: {} seconds'.format(total_generated, len(descriptions), elapsed_time_min * 60))
        print('Batch size: {}, throughput: {} clips/minute'.format(b, throughput))

        total_generated = 0
        

timed_execution(64)