from datasets import load_dataset
import json
from tqdm import tqdm

# load data from HF
# TODO: REMOVE THIS FILE when release.

dataset = load_dataset('OpenDatasets/dalle-3-dataset', cache_dir='data_set', ignore_verifications=True)

out_dir = './image_data/'
target_size = (1024, 1024)
target_data_size = 1000
data_count = 1500
caption_map = {}
for i in tqdm(range(data_count)):
    file_name = f'{len(caption_map):05d}.png'
    im = dataset['train'][i]['image']
    if im.size == (1024, 1024):
        dataset['train'][i]['image'].save(f'{out_dir}{file_name}')
        caption_map[file_name] = dataset['train'][i]['caption']

    if len(caption_map) == target_data_size:
        break

with open(f'{out_dir}caption.json', 'w', encoding='utf-8') as f:
    json.dump(caption_map, f, ensure_ascii=False, indent=4)

