import json
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


def draw_loss(config):
    with open(os.path.join(config["eval_config"]["save_dir"],"info.json")) as f:
        info_list=json.load(f)

    loss_values = [entry['training_loss'] for entry in info_list]
    step_values = [entry['steps'] for entry in info_list]

    draw(config, step_values, loss_values, xlabel="steps", ylabel="training_loss")

def draw(config, x, y, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(config["eval_config"]["save_dir"],ylabel+'.png'))


def draw_ppl(config, model, tokens, stride, split):
    result = {}
    device = "cuda:0"

    max_length = model.config.max_position_embeddings
    seq_len = tokens.size(0)

    nlls = []
    prev_end_loc = 0
    positions = []
    positions_loss = []
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = tokens[begin_loc:end_loc].to(device)
        labels = input_ids.clone()
        labels[:-trg_len] = -100

        # print(f"len label:{len(labels)}")

        if prev_end_loc == 0:
            # input :  0,1,2|3 
            # label :0|1,2,3
            # cal prev_end_loc+1 ~ end_loc-1
            positions += [pos for pos in range(prev_end_loc+1, end_loc)]
        else:
            # input :  2,3,4|5 prev_end_loc=4 end_loc=6
            # label :2|3,4,5
            # cal prev_end_loc ~ end_loc-1
            positions += [pos for pos in range(prev_end_loc, end_loc)]
        # print(len([pos for pos in range(prev_end_loc+1, end_loc+1)]))
        # print(positions)

        input_ids = input_ids[None,:].to(device)
        labels = labels[None,:].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

            # calculate the ppl on every position 
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous() # [batch_size,seq_len-1,vocab_size]
            shift_labels = labels[..., 1:].contiguous()  # [batch_size,seq_len-1]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # print(loss.shape)
            other_loss = loss[:-trg_len]
            loss = loss[-trg_len:]
            # print("mask loss:",sum(other_loss))
            assert all(other_loss == 0), "calculated non-need tokens"
            assert all(loss > 0), "didn't calculate the need tokens"
            
             
            # print(loss)
            positions_loss += loss.squeeze().tolist()
            # print(loss.shape)
            # print(len(loss.squeeze().tolist()))
            

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    my_ppl = np.exp(np.mean(positions_loss))

    # The calculation method of ppl_hf comes from https://huggingface.co/docs/transformers/main/en/perplexity#calculating-ppl-with-fixed-length-models
    print(f"ppl_hf:{ppl}")
    print(f"my_ppl:{my_ppl}")
    result["ppl_hf"]=ppl
    result["my_ppl"]=my_ppl
    result["positions"] = positions
    result["loss"] = positions_loss
    positions_ppl = [np.exp(loss) for loss in positions_loss]

    
    draw(config, positions, positions_loss,xlabel="postion",ylabel=f"{split}_loss")
    draw(config, positions, positions_ppl,xlabel="postion",ylabel=f"{split}_ppl")

    # smoothing for good view
    def smoothing(positions, values, stride):
        smoothing_postions = []
        smoothing_value = []
        for i in range(0,len(positions),stride):
            if i+stride > len(positions):
                break
            smoothing_postions.append(positions[i+stride-1])
            smoothing_value.append(np.mean(values[i:i+stride]))
        return smoothing_postions, smoothing_value

    def draw_smoothing(stride):  
        smoothing_postions, smoothing_loss = smoothing(positions, positions_loss, stride=stride)
        draw(config, smoothing_postions, smoothing_loss, xlabel="postion",ylabel=f"{split}_loss(stride={stride})")
        smoothing_ppl = [np.exp(loss) for loss in smoothing_loss]
        draw(config, smoothing_postions, smoothing_ppl, xlabel="postion",ylabel=f"{split}_ppl(stride={stride})")

    for i in range(1,18):
        draw_smoothing(stride=int(2**i))

    
    

if __name__ == "__main__":

    device = "cuda:0"

    args = parse_args()
    with open(args.config_path) as f:
        config=json.load(f)

    # the save_dir will save a copy of the config in training
    with open(os.path.join(config["eval_config"]["save_dir"],"config.json")) as f:
        config=json.load(f)

    training_config = config["training_config"]

    cfg = LlamaConfig(**config["model_cofig"])
    model = LlamaForCausalLM(cfg)
    state_dict = torch.load(os.path.join(training_config["save_dir"],"model.pt"))
    # exit(0)
    model.load_state_dict(state_dict)
    model.to(device)

    tokens_path = os.path.join(config["path"]["prepare_data_path"],'validation.pt')
    valid_tokens = torch.load(tokens_path)

    tokens_path = os.path.join(config["path"]["prepare_data_path"],'test.pt')
    test_tokens = torch.load(tokens_path)

    draw_loss(config)
    draw_ppl(config, model, valid_tokens, stride=config["model_cofig"]["max_position_embeddings"]//2, split='valid')
    draw_ppl(config, model, test_tokens, stride=config["model_cofig"]["max_position_embeddings"]//2, split='test')

"""
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluate.py --config_path ./configs/llama/1/config.json
"""