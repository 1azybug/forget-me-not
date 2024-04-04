from transformers.cache_utils import Cache
import torch
import os
class TrainingCache(Cache):

    def __init__(self, cache_config):
        self.key_cache = []
        self.value_cache = []
        print(cache_config)
        self.config = cache_config
        self.shift = self.config["shift"]
        self.stride = self.config["stride"]
        self.work_size = self.config["work_size"]
        self.cache_size = self.config["cache_window_len"]
        self.max_layer = self.config["num_layer"]

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs = None,
    ):
        """
        Input:
        key_states, value_states:[batch_size, seq_len, kv_head_num, head_dim]

        return : 
        key_states, value_states:[batch_size, cache_len+seq_len, kv_head_num, head_dim]
        position_ids:[cache_len+seq_len]
        """
    
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            # position_ids = torch.arange(0,key_states.shape[1],device=key_states.device)
            return key_states, value_states
        else:
            # get the cache first, when return_layer_idx == layer_idx, key_cache[return_layer_idx] will store the key_states
            return_layer_idx = min(self.max_layer-1, layer_idx+self.shift)
            k_cache = self.key_cache[return_layer_idx] 
            v_cache = self.value_cache[return_layer_idx]
            # [batch_size, q_len, head_num, head_dim]
            into_key_cache = key_states[:,::self.stride]
            into_value_cache = value_states[:,::self.stride]

            # now_cache_size + extend_size - forget_num <= max_cache_size -> forget_num >= now_cache_size + extend_size - max_cache_size
            forget_num = max(0, self.key_cache[layer_idx].size(1)+into_key_cache.size(1)-self.cache_size)
            

            # I'm not sure index_copy_ whether track the gradient in original element, so I use torch.cat
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx][:,forget_num:], into_key_cache], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:,forget_num:], into_value_cache], dim=1)

            # position_ids = torch.arange(0,self.key_cache[return_layer_idx].shape[1]+key_states.shape[1],device=key_states.device)
            return torch.cat([k_cache, key_states], dim=1), torch.cat([v_cache, value_states], dim=1)



    
    def stop_gradient(self):
        for i in range(self.max_layer):
            self.key_cache[i] = self.key_cache[i].detach()
            self.value_cache[i] = self.value_cache[i].detach()

    def get_seq_length(self, layer_idx = 0):
            """Returns the sequence length of the cached states. A layer index can be optionally passed."""
            if len(self.key_cache) <= layer_idx:
                return 0
            return self.key_cache[layer_idx].shape[1]
        
    
    def state_dict(self):
        return {
            "config":self.config,
            "key":self.key_cache,
            "value":self.value_cache,
        }
        
    def load_state_dict(self,state_dict,path=None):
        self.config = state_dict["config"]
        self.key_cache = state_dict["key"]
        self.value_cache = state_dict["value"]
        
        self.shift = self.config["shift"]
        self.stride = self.config["stride"]
        self.work_size = self.config["work_size"]
        self.cache_size = self.config["cache_window_len"]
        self.max_layer = self.config["num_layer"]
        if path is not None:
            self.mem_path = path
        assert self.mem_path is not None
    
    # batch_size memories to 1 memories
    def reduce2one(self):
        for i in range(self.max_layer):
            # [batch_size, q_len, head_num, head_dim] -> [1, q_len, head_num, head_dim]
            self.key_cache[i] = self.key_cache[i][0:1]
            self.value_cache[i] = self.value_cache[i][0:1]
        
    def to(self,device):
        for i in range(self.max_layer):
            self.key_cache[i] = self.key_cache[i].to(device)
            self.value_cache[i] = self.value_cache[i].to(device)
        
    def restore_memory(self):
        assert self.mem_path is not None
        if os.path.exists(self.mem_path):
            self.load_state_dict(torch.load(self.mem_path))
            self.reduce2one()
            self.to('cuda')
            
    
    def reset_memory(self):
        self.key_cache = []
        self.value_cache = []
        
        
class EvalCache(TrainingCache):
    """
    Eval setting:
    teacher forcing and feed the segment like training
    """
    pass
    
    
    
def get_eval_cache(config):
    """
    config can be a whole config or only cache_config
    """
    if "cache_config" in config:
        config = config["cache_config"]
    
    # now config is cache_config
    if config["eval_cache_type"] == "normal":
        return EvalCache(config)
    else:
        return None