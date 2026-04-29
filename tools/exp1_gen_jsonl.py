import json
import os
from tqdm import tqdm
from glob import glob

# 准备数据
data = []

input_dir = "/gpfs01/nfs_share/data20250106/yuqiangz/master_models/CosyVoice/audios_gemini_tts_for_prompt_16k"
wav_path_list = glob(os.path.join(input_dir, "*/*.wav"))
wav_path_list = wav_path_list*100
assert len(wav_path_list) > 0, "No wav files found in the input directory."
for wav_path in tqdm(wav_path_list, desc="Processing audio files"):
    txt_path = wav_path[:-4] + ".txt"
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    data.append({
        "text": text,
        "audio": wav_path,
        "voice_prompts": wav_path
    })
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')