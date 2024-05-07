import os

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import subprocess

model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration=30)  # generate 8 seconds.
all_descriptions = ['happy rock', 'energetic EDM', 'sad jazz', 'sad rock', 'happy EDM', 'energetic jazz', 'Happy taylor swift pop']

EXP_DURATION = 60

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

def generate_batch(b):
    print('Generating batch size: {}'.format(b))
    start_time = current_time_secs()
    num_generated = 0
    descriptions = [all_descriptions[i % len(all_descriptions)] for i in range(b)]
    wav = model.generate(descriptions, progress=True)  # generates 3 samples.
    num_generated += len(descriptions)
    curr_time = current_time_secs()
    elapsed_time = curr_time - start_time
    elapsed_time_min = elapsed_time / 60.0
    throughput = num_generated / elapsed_time_min
    print('Number generated: {}, {}, time taken: {} seconds'.format(num_generated, len(descriptions), elapsed_time_min * 60))
    print('Batch size: {}, throughput: {} clips/minute'.format(b, throughput))
    print('Memory: {}'.format(get_gpu_memory()))

def benchmark():
    for b in [32]:
        generate_batch(b)

start_time = current_time_secs()
while current_time_secs() - start_time < 10 * 60:
    generate_batch(16)
    
# for idx, one_wav in enumerate(wav):
#     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#     audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

# https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/models/lm.py#L526
       
# current_memory_used = torch.cuda.memory_allocated() / 1024.0 / 1024.0 / 1024.0
#             if current_memory_used > maximum_memory_used:
#                 maximum_memory_used = current_memory_used
#             torch.cuda.empty_cache()