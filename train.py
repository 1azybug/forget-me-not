import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import os
import time
import json
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class LMOrderedDataset(IterableDataset):
    def __init__(self, data, batch_size, seg_len):
        super(LMOrderedDataset).__init__()

        self.batch_size = batch_size
        self.seg_len = seg_len
        self.data = data.view(batch_size, -1).t().contiguous()  # data: [Seq_len,Batch_size]

    def __iter__(self):

        for beg_idx in range(0, self.data.size(0) - 1, self.seg_len):
            end_idx = beg_idx + self.seg_len
            yield self.data[beg_idx:end_idx].t().contiguous()  # [batch_size,seg_len]

    # def __len__(self):
    #     return self.data.size(0)//self.seg_len




# Initialize process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def training_step(ddp_model, inputs, rank, accumulation_steps):
    inputs = inputs.to(rank)
    labels = inputs.clone().detach()
    outputs = ddp_model(input_ids=inputs,labels=labels)
    loss = outputs[0]
    return_loss = loss.item()
    loss /= accumulation_steps
    loss.backward()
    return return_loss




def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for name,p in model.named_parameters() if p.requires_grad and ('lm_head' in name or "emb" in name))
    print(f"Total parameters: {params}")
    print(f"Embedding parameters: {embedding_params}")
    print(f"non-Embedding parameters: {params-embedding_params}")



# Training process
def train(rank, world_size):
    
    with open("config.json") as f:
        config=json.load(f)

    setup(rank, world_size)
    torch.cuda.set_device(rank)

    training_config = config["training_config"]
    if not os.path.exists(training_config["save_dir"]):
        os.makedirs(training_config["save_dir"])

    assert world_size == training_config["device_count"], "device_count wrong"
    assert training_config["total_batch_size"] == training_config['batch_size_per_device']*training_config["device_count"]*training_config["gradient_accumulation_steps"]

    tokens_path = os.path.join(config["path"]["prepare_data_path"],'train.pt')
    tokens = torch.load(tokens_path)
    # tokens = torch.arange(0, 32000, dtype=torch.int64) # for check 

    # cal the total step
    batch_tokens_num = training_config["total_batch_size"] * training_config["segment_len"]
    training_steps = tokens.shape[0]//batch_tokens_num

    # drop last tokens
    tokens = tokens.narrow(0,0,batch_tokens_num*training_steps)

    tokens_per_gpu = tokens.shape[0] // training_config["device_count"]
    start_index = rank * tokens_per_gpu
    end_index = start_index + tokens_per_gpu
    tokens = tokens[start_index:end_index]

    if rank==0:
        print(f"[INFO] batch_tokens:{batch_tokens_num} | training_steps:{training_steps}")
    print(f"[INFO] rank{rank} training tokens[{start_index}:{end_index}] | total_tokens:{tokens.shape[0]} | batch_tokens_num_per_gpu:{tokens.shape[0]//training_steps} | training_steps:{training_steps}")

    # Instantiate the model and move it to the corresponding GPU
    cfg = LlamaConfig(**config["model_cofig"])
    model = LlamaForCausalLM(cfg).to(rank)

    if rank == 0:
        count_parameters(model)
    
    ddp_model = DDP(model, device_ids=[rank])

    # Instantiate the data loader
    dataset = LMOrderedDataset(tokens,training_config['batch_size_per_device'], training_config["segment_len"])

    loader = DataLoader(dataset, batch_size=None)
    # print(len(loader))

    # Instantiate  optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    scheduler = None
    memory = None



    accumulation_steps = training_config["gradient_accumulation_steps"]
    step_num = 0
    optimizer.zero_grad()
    info_list = []
    start_time = time.time()
    for epoch in range(1):

        def save():
            if rank!=0:
                return
            with open(os.path.join(training_config["save_dir"],"info.json"),'w') as f:
                json.dump(info_list,f,indent=4)

            torch.save(ddp_model.state_dict(),os.path.join(training_config["save_dir"],"model.pt"))
            torch.save(optimizer.state_dict(),os.path.join(training_config["save_dir"],"optimizer.pt"))
            if scheduler is not None:
                torch.save(scheduler.state_dict(),os.path.join(training_config["save_dir"],"scheduler.pt"))
            if memory is not None:
                torch.save(memory,os.path.join(training_config["save_dir"],"memory.pt"))

        for inputs in tqdm(loader,total=training_steps*accumulation_steps):
            step_num += 1

            # print(f"\n{'-'*80}\nstep{step_num}\ndevice:[{rank}]\ninputs:{inputs}\n{'-'*80}")  # for check

            if step_num % accumulation_steps == 0:
                loss = training_step(ddp_model,inputs,rank,accumulation_steps)
            else:
                with ddp_model.no_sync():
                    loss = training_step(ddp_model,inputs,rank,accumulation_steps)

            info_list.append({
                "run_time(hours)":(time.time()- start_time)/3600,
                "total_steps":training_steps,
                "steps":step_num/accumulation_steps, 
                "training_loss":loss, 
                "ppl":np.exp(loss),
                "learning_rate":optimizer.param_groups[0]['lr']})
            
            if step_num % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            
            if step_num % (training_config["log_step"]*accumulation_steps) == 0:
                if rank == 0:
                    print(info_list[-1])
            if step_num % (training_config["save_step"]*accumulation_steps) == 0:
                save()

        save()






# Launch multi-process training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)

"""
# 用 > train.log 无法实时查看输出
CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/liuxinyu/zrs/forget-me-not/train.py 
"""




