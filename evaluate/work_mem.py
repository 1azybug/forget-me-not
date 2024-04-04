import torch
import random
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def generate_random_integer_list(min_value, max_value, length):
    return [random.randint(min_value, max_value) for _ in range(length)]

def get_random_example(length):
    min_value = 10
    max_value = 31999
    bos_token_id=1
    eos_token_id=2
    copy_tokens = generate_random_integer_list(min_value, max_value, length)
    lm_tokens = copy_tokens.copy()
    copy_tokens = [bos_token_id] + copy_tokens + [bos_token_id] + copy_tokens + [eos_token_id] 

    # no same tokens prepend, test the LM acc%
    irrelevant_tokens = generate_random_integer_list(min_value, max_value, length)
    lm_tokens = [bos_token_id] + irrelevant_tokens + [bos_token_id] + lm_tokens + [eos_token_id] 
    mask = [0 for _ in range(1+length+1)] + [1 for _ in range(length)] + [0]
    return lm_tokens, copy_tokens, mask

def get_order_example(length, tokens):

    max_len = tokens.size(0)
    begin_loc = random.randint(0, max_len-length*3)
    end_loc = begin_loc + length

    bos_token_id=1
    eos_token_id=2

    copy_tokens = tokens[begin_loc:end_loc].tolist()
    copy_tokens = [bos_token_id] + copy_tokens + [bos_token_id] + copy_tokens + [eos_token_id] 
    mask = [0 for _ in range(1+length+1)] + [1 for _ in range(length)] + [0]

    # no same tokens prepend, test the LM acc%
    lm_tokens = tokens[begin_loc:end_loc].tolist()
    irrelevant_tokens = tokens[-length:].tolist()
    lm_tokens = [bos_token_id] + irrelevant_tokens + [bos_token_id] + lm_tokens + [eos_token_id] 


    return lm_tokens, copy_tokens, mask

def mem_forward(model, prompt, memory):
    work_size = memory.work_size
    s = prompt.size(1)
    # print(f"[DEBUG] len={s}, prompt shape:{prompt.shape}")
    all_ids = []
    with torch.no_grad(): 
        for beg in range(0, s, work_size):
            forward_prompt = prompt[:,beg:beg+work_size] 
            # print(f"[DEBUG] prompt[:,{beg}:{beg+work_size}] ")
            output = model(forward_prompt,past_key_values=memory)
            logits = output.logits
            predicted_token_ids = torch.argmax(logits, dim=-1)
            predicted_token_ids = predicted_token_ids.squeeze().tolist()
            all_ids += predicted_token_ids
    return all_ids
        
    

def get_acc(model, example, mask, length, memory=None):
    prompt = torch.tensor(example, dtype=torch.long)
    prompt = prompt[None,:].cuda()
    # [1,S]
    with torch.no_grad():
        if memory is None:
            output = model(prompt)
            logits = output.logits
            predicted_token_ids = torch.argmax(logits, dim=-1)
            predicted_token_ids = predicted_token_ids.squeeze().tolist()
        else:
            # clear the memory
            memory.reset_memory()
            predicted_token_ids = mem_forward(model, prompt, memory)

    # shift right
    predicted_token_ids = [example[0]] + predicted_token_ids[:-1]

    mask = torch.tensor(mask, dtype=torch.bool)
    src = torch.tensor(example, dtype=torch.long)[mask]
    tgt = torch.tensor(predicted_token_ids, dtype=torch.long)[mask]

    # pre 50% tokens for few-shot prompt, only calculate the accuracy of the last 50% token
    result = (src[-length//2:] == tgt[-length//2:]).float().mean().item()
    return result


def get_length_result(config, model, length, tokens=None, test_times=100, data_type="random", memory=None):

    lm_results = []
    copy_results = []
    print("-"*80)
    print(data_type)
    print(f"length:{length}")
    for _ in tqdm(range(test_times)):
        if data_type=="random":
            lm_tokens, copy_tokens, mask = get_random_example(length)
        else:
            lm_tokens, copy_tokens, mask = get_order_example(length, tokens)
        
        lm_acc = get_acc(model, lm_tokens, mask, length, memory)
        copy_acc = get_acc(model, copy_tokens, mask, length, memory)

        lm_results.append(lm_acc)
        copy_results.append(copy_acc)
    
    lm_mean = np.mean(lm_results)
    lm_std = np.std(lm_results)
    copy_mean = np.mean(copy_results)
    copy_std = np.std(copy_results)
    print(f"lm_mean:{lm_mean}, lm_std:{lm_std}")
    print(f"copy_mean:{copy_mean}, copy_std:{copy_std}")
    # print(results)
    return {
        "length":length, 
        "lm_mean":lm_mean, 
        "lm_std":lm_std,
        "copy_mean":copy_mean,
        "copy_std":copy_std
    }

def get_result(config, model, tokens=None, test_times=100, data_type="random", memory=None):

    # if os.path.exists(os.path.join(config["eval_config"]["save_dir"],f"{data_type}_work_memery_result.json")):
    #     with open(os.path.join(config["eval_config"]["save_dir"],f"{data_type}_work_memery_result.json")) as f:
    #         result = json.load(f)
    #     return result

    if 'test_length' not in config["eval_config"]:
        test_length = 8192
    else:
        test_length = config["eval_config"]["test_length"]

    results = []
    for i in range(12000):
        length = int(2**i)
        if length > test_length:
            break
        
        if length >= 8192:
            # only test 10 for save time
            test_times = 10
        result = get_length_result(config, model, length=length, tokens=tokens, test_times=test_times, data_type=data_type, memory=memory)
        results.append(result)

    with open(os.path.join(config["eval_config"]["save_dir"],f"{data_type}_work_memery_result.json"), "w") as f:
        json.dump(results, f, indent=4)
    return results

def draw_work_memory(config, model, tokens=None, test_times=100, data_type="random", memory=None):

    results = get_result(config, model, tokens=tokens, test_times=test_times, data_type=data_type, memory=memory)
    length = [result['length'] for result in results]
    lm_acc = [result['lm_mean'] for result in results]
    lm_acc_upper = [result['lm_mean']+result['lm_std'] for result in results]
    lm_acc_lower = [result['lm_mean']-result['lm_std'] for result in results]

    copy_acc = [result['copy_mean'] for result in results]
    copy_acc_upper = [result['copy_mean']+result['copy_std'] for result in results]
    copy_acc_lower = [result['copy_mean']-result['copy_std'] for result in results]

    plt.figure(figsize=(10, 5))

    plt.semilogx(length, lm_acc, label='lm_accuracy', color='blue')
    plt.semilogx(length, copy_acc, label='copy_accuracy', color='orange')

    plt.fill_between(length, lm_acc_lower, lm_acc_upper, color='blue', alpha=0.1)
    plt.fill_between(length, copy_acc_lower, copy_acc_upper, color='orange', alpha=0.1)

    plt.axvline(x=config["training_config"]["segment_len"]//2, color='red', linestyle='--')
    plt.text(config["training_config"]["segment_len"]//2, 0.7, f'training_len//2={config["training_config"]["segment_len"]//2}', verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel("Squence Length(log-scale)")
    plt.ylabel("Accuracy")
    plt.title(config["eval_config"]["save_dir"].split('/')[-1])
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(config["eval_config"]["save_dir"],f'{data_type}_work_memory.png'))
    plt.close()

    

if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)


    args = parse_args()
    with open(args.config_path) as f:
        config=json.load(f)

    # the save_dir will save a copy of the config in training
    with open(os.path.join(config["eval_config"]["save_dir"],"config.json")) as f:
        config=json.load(f)

    training_config = config["training_config"]
    config["model_config"]["max_position_embeddings"] = 20000
    cfg = LlamaConfig(**config["model_config"])
    model = LlamaForCausalLM(cfg)
    state_dict = torch.load(os.path.join(training_config["save_dir"],"model.pt"))
    model.load_state_dict(state_dict)
    model.cuda()

    tokens_path = os.path.join(config["path"]["prepare_data_path"],'validation.pt')
    valid_tokens = torch.load(tokens_path)

    draw_work_memory(config, model, tokens=valid_tokens, test_times=100, data_type="order")
    draw_work_memory(config, model, tokens=valid_tokens, test_times=100, data_type="random")



"""
python /home/liuxinyu/zrs/forget-me-not/work_mem.py --config_path /home/liuxinyu/zrs/forget-me-not/configs/llama/seglen2048/config.json
"""