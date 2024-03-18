import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import json
from tqdm import tqdm
with open("config.json") as f:
    config=json.load(f)

if "cache" in config["path"]:
    os.environ["HF_DATASETS_CACHE"] = config["path"]["cache"]
    os.environ["HF_HOME"] = config["path"]["cache"]
    os.environ["HUGGINGFACE_HUB_CACHE"] = config["path"]["cache"]
    os.environ["TRANSFORMERS_CACHE"] = config["path"]["cache"]

data_path = config["path"]['pg19']

# 获取目录下的所有项
all_items = os.listdir(data_path)

# 获取所有文件的绝对路径
train_files = [os.path.abspath(os.path.join(data_path, f)) for f in all_items if 'train' in f]
valid_files = [os.path.abspath(os.path.join(data_path, f)) for f in all_items if 'valid' in f]
test_files = [os.path.abspath(os.path.join(data_path, f)) for f in all_items if 'test' in f]

data_files = {
    'train': train_files,
    'validation': valid_files,
    'test': test_files
}

dataset = load_dataset('parquet', data_files=data_files, cache_dir=config["path"]["cache"])
print(dataset)
tokenizer = AutoTokenizer.from_pretrained(config["path"]["tokenizer_path"])
def prepare_data(dataset, split):
    
    all_ids = []
    for example in tqdm(dataset[split], desc="Processing examples"):
        all_text = ""
        all_text += "<bos>\nshort book title: "+example["short_book_title"]+"\n"
        all_text += "publication date: "+str(example["publication_date"])+"\n"
        all_text += "book text: "+example["text"]+"\n<eos>"
        ids = tokenizer(all_text)["input_ids"]
        all_ids += ids

    # print(all_text)
    tokens = torch.tensor(all_ids)
    print(tokens.shape)
    print(tokens)
    torch.save(tokens,os.path.join(config["path"]["prepare_data_path"],split+'.pt'))


    

prepare_data(dataset, split='validation')
prepare_data(dataset, split='test')
prepare_data(dataset, split='train')
"""
python /home/liuxinyu/zrs/forget-me-not/data_utils/prepare_data.py > prepare_data.log
"""

    