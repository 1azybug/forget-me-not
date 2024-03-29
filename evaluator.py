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
import json
from evaluate.work_mem import draw_work_memory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

class Evaluator:

    def __init__(self, config, title=None, test_length=None):
        self.config = config

        self.config['training_len'] = config["training_config"]["segment_len"]

        if title is None:
            self.config['title'] = config["eval_config"]["save_dir"].split('/')[-1]
        else:
            self.config['title'] = title

        if test_length is None:
            self.config["model_config"]["max_position_embeddings"] = 20000
            self.config["eval_config"]["test_length"] = 20000
        else:
            self.config["model_config"]["max_position_embeddings"] = test_length
            self.config["eval_config"]["test_length"] = test_length

        cfg = LlamaConfig(**self.config["model_config"])
        self.model = LlamaForCausalLM(cfg)
        state_dict = torch.load(os.path.join(config["eval_config"]["save_dir"],"model.pt"))
        self.model.load_state_dict(state_dict)
        self.model.to('cuda')

    def draw_loss(self):
        with open(os.path.join(self.config["eval_config"]["save_dir"],"info.json")) as f:
            info_list=json.load(f)

        loss_values = [entry['training_loss'] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]
        lr_values = [entry['learning_rate'] for entry in info_list]

        self.draw(step_values, loss_values, xlabel="Step", ylabel="training loss", label_training_len=False)
        self.draw(step_values, lr_values, xlabel="Step", ylabel="learning rate", label_training_len= False)

    def draw(self, x, y, xlabel, ylabel, label_training_len=False):
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label=ylabel)

        if label_training_len:
            plt.axvline(x=self.config['training_len'], color='red', linestyle='--')
            plt.text(self.config['training_len'], 0.7, f"training_len={self.config['training_len']}", verticalalignment='bottom', horizontalalignment='right')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.config['title'])
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.config["eval_config"]["save_dir"], ylabel+'.png'))
        plt.close()

    def sliding_ppl(self, tokens, stride, split):

        if os.path.exists(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_sliding_cache.json")):
            with open(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_sliding_cache.json")) as f:
                result = json.load(f)
            return result

        max_length = self.config['training_len']
        seq_len = tokens.size(0)

        prev_end_loc = 0
        positions = []
        positions_loss = []
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = tokens[begin_loc:end_loc]
            labels = input_ids.clone()
            labels[:-trg_len] = -100

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

            # batch size = 1
            input_ids = input_ids[None,:].to('cuda')
            labels = labels[None,:].to('cuda')
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.

                # calculate the ppl on every position 
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous() # [batch_size,seq_len-1,vocab_size]
                shift_labels = labels[..., 1:].contiguous()  # [batch_size,seq_len-1]
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="none")
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                loss = loss[-trg_len:] # [batch_size,trg_len]  /  [batch_size,trg_len-1]
                positions_loss += loss.squeeze().tolist()
                


            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        my_ppl = np.exp(np.mean(positions_loss))

        # The calculation method of ppl_hf comes from https://huggingface.co/docs/transformers/main/en/perplexity#calculating-ppl-with-fixed-length-models
        print(f"{split}_sliding_my_ppl:{my_ppl}")

        result = {}
        result[f"sliding_ppl"]=my_ppl
        with open(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_sliding_result.json"), "w") as f:
            json.dump(result, f, indent=4)

        result[f"sliding_positions"] = positions
        result[f"sliding_loss"] = positions_loss
        with open(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_sliding_cache.json"), "w") as f:
            json.dump(result, f, indent=4)

        return result

    def draw_sliding_ppl(self, result, split):
        
        positions = result["sliding_positions"]
        positions_loss = result["sliding_loss"]

        # prefix_segment_ppl
        def prefix(positions, values, stride):

            # the first position(<bos>) prob is 1
            seg_lens = []
            prefix_value = []
            prefix_sum = 0 
            prefix_len = 0

            # here position begin with the second token
            for i in tqdm(range(0,len(positions),stride)):
                seg_lens.append(i+stride)
                prefix_sum += np.sum(values[i:i+stride])
                prefix_len += len(values[i:i+stride])
                prefix_value.append(prefix_sum/prefix_len)
            return seg_lens, prefix_value

        def draw_prefix(stride):  
            # trunc_len = 64
            seg_lens, prefix_loss = prefix(positions, positions_loss, stride=stride)
            # self.draw(seg_lens, prefix_loss, xlabel="Sequence_Length",ylabel=f"{split}_prefix_loss(stride={stride})")
            # self.draw(prefix_postions[:trunc_len], prefix_loss[:trunc_len], xlabel="Sequence_Length",ylabel=f"{split}_prefix_loss(stride={stride}|trunc={trunc_len})")
            prefix_ppl = [np.exp(loss) for loss in prefix_loss]
            self.draw(seg_lens, prefix_ppl, xlabel="Sequence_Length",ylabel=f"{split}_prefix_ppl(stride={stride})",label_training_len=False)
            # self.draw(prefix_postions[:trunc_len], prefix_ppl[:trunc_len], xlabel="Sequence_Length",ylabel=f"{split}_prefix_ppl(stride={stride}|trunc={trunc_len})")

        draw_prefix(stride = 1024)


    def standard_ppl(self, tokens, stride, split):

        if os.path.exists(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_standard_cache.json")):
            with open(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_standard_cache.json")) as f:
                result = json.load(f)
            return result

        seq_lens = []
        seq_lens_loss = []
        for seq_len in tqdm(range(stride, self.model.config.max_position_embeddings, stride)):
            input_ids = tokens[:seq_len]
            input_ids = input_ids[None,:].to('cuda')
            labels = input_ids.clone().to('cuda')
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
            seq_lens.append(seq_len)
            seq_lens_loss.append(outputs.loss.item())

        result = {}
        result["standard_seq_lens"] = seq_lens
        result["standard_loss"] = seq_lens_loss 

        with open(os.path.join(self.config["eval_config"]["save_dir"],f"{split}_standard_cache.json"), "w") as f:
            json.dump(result, f, indent=4)
        return result

    def draw_standard_ppl(self, result, split):
        seq_len = result["standard_seq_lens"]
        loss = result["standard_loss"]
        ppl = [np.exp(l) for l in loss]
        # self.draw(seq_len, loss, xlabel="Sequence_Length",ylabel=f"{split}_standard_loss")
        self.draw(seq_len, ppl, xlabel="Sequence Length", ylabel=f"{split}_standard_ppl", label_training_len=True)
     

    def evaluate(self, tokens, split):

        result = self.sliding_ppl(tokens, stride=self.config['training_len']//2, split=split)
        self.draw_sliding_ppl(result, split=split)

        result = self.standard_ppl(tokens, stride=128, split=split)
        self.draw_standard_ppl(result, split=split)

        

    def run(self):
        
        # draw training loss
        self.draw_loss()

        tokens_path = os.path.join(config["path"]["prepare_data_path"],'validation.pt')
        valid_tokens = torch.load(tokens_path)
        tokens_path = os.path.join(config["path"]["prepare_data_path"],'test.pt')
        test_tokens = torch.load(tokens_path)

        self.evaluate(valid_tokens, split='valid')
        self.evaluate(test_tokens, split='test')
        draw_work_memory(self.config, self.model, tokens=test_tokens, test_times=100, data_type="order")


if __name__ == "__main__":

    args = parse_args()
    with open(args.config_path) as f:
        config=json.load(f)

    # the save_dir will save a copy of the config in training
    config["eval_config"]["save_dir"] += '/'+ args.config_path.split('/')[-2]
    with open(os.path.join(config["eval_config"]["save_dir"],"config.json")) as f:
        config=json.load(f)

    evaluator = Evaluator(config)
    evaluator.run()


"""
CUDA_VISIBLE_DEVICES=3 python ./evaluator.py --config_path ./configs/llama/1/config.json
"""