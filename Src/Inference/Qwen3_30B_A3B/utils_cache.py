from collections import deque, Counter, defaultdict
from typing import Optional, Union, List
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import math
import statistics



class CachePolicyWrapper:
    """
    CachePolicyWrapper is a class that wraps around different cache policies.
    It provides a unified interface for accessing and updating the cache.
    """

    def __init__(self, cache_policy: str = "lru", cache_size: int = 16, n_layers: int = 16, expert_path: str = "", locked_limit=10, locked_size = 3, beta=30.0, ffn_model_path: str = None, num_experts: int = 64):
        """
        Initialize the CachePolicyWrapper with a specific cache policy.

        :param cache_policy: The cache policy to be wrapped.
        :param ffn_model_path: Path to FFN eviction model weights for ML_CECAR cache policy.
        """
        self.cache_policy = None

        match cache_policy:
            case "lru":
                self.cache_policy = LRUCache(cache_size, n_layers, expert_path)
            case "lfu":
                self.cache_policy = LFUCache(cache_size, n_layers, expert_path)
            case "lifo":
                self.cache_policy = LIFOCache(cache_size, n_layers, expert_path)
            case "unified_lru":
                self.cache_policy = UnifiedLRUCache(cache_size, n_layers, expert_path)
            case "markov":
                self.cache_policy = MarkovCache(cache_size, n_layers)
            case "unified_markov":
                self.cache_policy = UnifiedMarkovCache(cache_size, n_layers)
            case "proactive":
                self.cache_policy = ProactiveCache(cache_size, n_layers)
            case "unified_proactive":
                self.cache_policy = UnifiedProactiveCache(cache_size, n_layers)
            case "arithmetic":
                self.cache_policy = ArithmeticCache(cache_size, n_layers, expert_path)
            case "unified_arithmetic":
                self.cache_policy = UnifiedArithmeticCache(cache_size, n_layers, expert_path)
            case "lockup":
                self.cache_policy = LockupCache(cache_size, n_layers, expert_path, locked_limit, locked_size)
            case "victim":
                self.cache_policy = VictimCache(cache_size, n_layers, expert_path)
            case "unified_lockup":
                self.cache_policy = UnifiedLockupCache(cache_size, n_layers, expert_path, locked_limit, locked_size)
            case "lru_lfu":
                self.cache_policy = LRULFUCache(cache_size, n_layers, expert_path, beta)
            case "ML_FlashMoE":
                self.cache_policy = MLCache(cache_size, n_layers, expert_path, num_experts=num_experts, ffn_model_path=ffn_model_path)
            case "ML_CECAR":
                self.cache_policy = MLCacheV2(cache_size, n_layers, expert_path, num_experts=num_experts, ffn_model_path=ffn_model_path)
            case "unified_ml":
                self.cache_policy = UnifiedMLCache(cache_size, n_layers, expert_path)
            case "mix":
                self.cache_policy = MixCache(cache_size, n_layers, expert_path)
            case "mix2":
                self.cache_policy = Mix2Cache(cache_size, n_layers, expert_path)
            case "rnn":
                self.cache_policy = RNNCache(cache_size, n_layers, expert_path)
            case "lecar":
                self.cache_policy = LeCaRCache(cache_size, n_layers, expert_path)
            case "arc":
                self.cache_policy = ARCCache(cache_size, n_layers, expert_path)
            case _:
                raise ValueError(f"Unknown cache policy: {cache_policy}")

    def get(self, layer_id, expert_id):
        return self.cache_policy.get(layer_id, expert_id)

    def add(self, layer_id, expert_id, expert_layer):
        return self.cache_policy.add(layer_id, expert_id, expert_layer)
    
    def update_history(self, layer_id, selected_experts):
        if hasattr(self.cache_policy, 'update_history'):
            self.cache_policy.update_history(layer_id, selected_experts)
        else:
            pass

    def update_arithmetic(self, layer_id, all_expert_info):
        if hasattr(self.cache_policy, 'update_arithmetic'):
            self.cache_policy.update_arithmetic(layer_id, all_expert_info)
        else:
            pass
    
    def evict(self, layer_id):
        return self.cache_policy.evict(layer_id)
    
    def replace(self, layer_id, expert_id, state_dict):
        return self.cache_policy.replace(layer_id, expert_id, state_dict)

    def is_cache_full(self, n_layers):
        return self.cache_policy.is_cache_full(n_layers)
    
    def get_layer_cache_size(self, layer_id):
        return self.cache_policy.get_layer_cache_size(layer_id)

    def get_eviction_score(self, layer_id, expert_id):
        return self.cache_policy.get_eviction_score(layer_id, expert_id)

    def get_eviction_scores_batch(self, layer_id, expert_ids):
        """
        Vectorized batch score retrieval. Falls back to loop for non-optimized policies.
        """
        if hasattr(self.cache_policy, 'get_eviction_scores_batch'):
            return self.cache_policy.get_eviction_scores_batch(layer_id, expert_ids)
        else:
            # Fallback for other cache policies
            import torch
            if isinstance(expert_ids, list):
                scores = [self.cache_policy.get_eviction_score(layer_id, eid) for eid in expert_ids]
                return torch.tensor(scores, device='cuda')
            else:
                scores = [self.cache_policy.get_eviction_score(layer_id, eid.item()) for eid in expert_ids]
                return torch.tensor(scores, device='cuda')

    def get_cached_expert_ids_tensor(self, layer_idx):
        """
        Return cached expert IDs as GPU tensor for vectorized operations.
        """
        if hasattr(self.cache_policy, 'get_cached_expert_ids_tensor'):
            return self.cache_policy.get_cached_expert_ids_tensor(layer_idx)
        else:
            import torch
            ids = self.get_cached_expert_ids(layer_idx)
            return torch.tensor(ids, device='cuda', dtype=torch.long)

    def contains(self, layer_id, expert_id):
        return expert_id in self.policy[layer_id]

    def get_cached_expert_ids(self, layer_idx):
        """
        현재 layer_idx에 cache에 존재하는 Expert ID 리스트 반환
        """
        if hasattr(self.cache_policy, "cache") and layer_idx < len(self.cache_policy.cache):
            return list(self.cache_policy.cache[layer_idx].keys())
        else:
            return []




class LRUCache:
    """
    LRUCache is a class that implements a Least Recently Used (LRU) cache. 
    It uses a dictionary to store the cached items and a list to maintain the order of usage.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size
        self.expert_path = expert_path
        self.order = [[] for _ in range(n_layers)]  
        self.cache = [{} for _ in range(n_layers)]  

    def is_cache_full(self, layer_id):
        return len(self.order[layer_id]) >= self.cache_size

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            self.order[layer_id].remove(expert_id)
            self.order[layer_id].append(expert_id)
            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order[layer_id].append(expert_id)
        self.cache[layer_id][expert_id] = expert_layer

    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])

    def replace(self, layer_id, expert_id, expert_layer):
        if not self.order[layer_id]:
            return 

        to_evict = self.order[layer_id].pop(0)
        self.order[layer_id].append(expert_id)
        evicted = self.cache[layer_id].pop(to_evict)
        del evicted
        torch.cuda.empty_cache()
        self.cache[layer_id][expert_id] = expert_layer

        return expert_layer
    
    def evict(self, layer_id):
        oldest = self.order[layer_id].pop(0)
        del self.cache[layer_id][oldest]

    def get_eviction_score(self, layer_id, expert_id):
            """
            Returns an eviction score for the expert.
            In LRU, the lower the index in self.order[layer_id], the more recently it was used.
            Higher index = older = higher eviction score.
            """
            if expert_id in self.order[layer_id]:
                return self.order[layer_id].index(expert_id)
            else:
                return len(self.order[layer_id])


from collections import defaultdict

class LFUCache:
    """
    LFUCache is a class that implements a Least Frequently Used (LFU) cache.
    It uses dictionaries to store cached items and their usage frequencies.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size
        self.expert_path = expert_path
        self.cache = [{} for _ in range(n_layers)] 
        self.freq = [{} for _ in range(n_layers)]  

    def is_cache_full(self, layer_id):
        return len(self.cache[layer_id]) >= self.cache_size

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            self.freq[layer_id][expert_id] += 1
            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.cache[layer_id][expert_id] = expert_layer
        self.freq[layer_id][expert_id] = 1

    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])

    def replace(self, layer_id, expert_id, expert_layer):
        if not self.cache[layer_id]:
            return

        min_freq = min(self.freq[layer_id].values())
        candidates = [eid for eid, f in self.freq[layer_id].items() if f == min_freq]
        to_evict = candidates[0]

        del self.cache[layer_id][to_evict]
        del self.freq[layer_id][to_evict]
        torch.cuda.empty_cache()

        self.cache[layer_id][expert_id] = expert_layer
        self.freq[layer_id][expert_id] = 1

        return expert_layer

    def evict(self, layer_id):
        if not self.cache[layer_id]:
            return
        min_freq = min(self.freq[layer_id].values())
        candidates = [eid for eid, f in self.freq[layer_id].items() if f == min_freq]
        to_evict = candidates[0]
        del self.cache[layer_id][to_evict]
        del self.freq[layer_id][to_evict]

class LIFOCache:
    """
    LIFOCache is a class that implements a Last-In First-Out (LIFO) cache.
    It uses a stack (list) to track insertion order.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size
        self.expert_path = expert_path
        self.stack = [[] for _ in range(n_layers)]  
        self.cache = [{} for _ in range(n_layers)]  

    def is_cache_full(self, layer_id):
        return len(self.cache[layer_id]) >= self.cache_size

    def get(self, layer_id, expert_id):
        return self.cache[layer_id].get(expert_id, None)

    def add(self, layer_id, expert_id, expert_layer):
        self.cache[layer_id][expert_id] = expert_layer
        self.stack[layer_id].append(expert_id)

    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])

    def replace(self, layer_id, expert_id, expert_layer):
        if not self.stack[layer_id]:
            return

        to_evict = self.stack[layer_id].pop()
        del self.cache[layer_id][to_evict]
        torch.cuda.empty_cache()

        self.cache[layer_id][expert_id] = expert_layer
        self.stack[layer_id].append(expert_id)

        return expert_layer

    def evict(self, layer_id):
        if not self.stack[layer_id]:
            return
        to_evict = self.stack[layer_id].pop()
        del self.cache[layer_id][to_evict]



class UnifiedLRUCache:
    """
    UnifiedLRUCache is a class that implements a unified LRU cache for both CPU and GPU.
    It uses the LRUCache class to manage the cache and provides methods to access and update the cache.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size * n_layers
        self.order = []  
        self.cache = {}  
        self.expert_path = expert_path

    def is_cache_full(self, layer_id):
        return len(self.order) >= self.cache_size

    def get(self, layer_id, expert_id):
        if (layer_id, expert_id) in self.cache:
            self.order.remove((layer_id, expert_id))
            self.order.append((layer_id, expert_id))
            return self.cache[(layer_id, expert_id)]
        return None
    
    def get_layer_cache_size(self, layer_id):
        return len([key for key in self.cache if key[0] == layer_id])

    def add(self, layer_id, expert_id, expert_layer):
        self.order.append((layer_id, expert_id))
        self.cache[(layer_id, expert_id)] = expert_layer

    def replace(self, layer_id, expert_id):
        if not self.order:
            return 

        to_evict = self.order.pop(0)

        self.order.append((layer_id, expert_id))

        expert_layer = self.cache.pop(to_evict)

        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.cache[(layer_id, expert_id)] = expert_layer

        return expert_layer

    def evict(self, layer_id):
        oldest = self.order.pop(0)
        del self.cache[oldest]


class UnifiedLRUCache:
    """
    UnifiedLRUCache is a class that implements a unified LRU cache for both CPU and GPU.
    It uses the LRUCache class to manage the cache and provides methods to access and update the cache.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size * n_layers
        self.order = []  
        self.cache = {} 
        self.expert_path = expert_path

    def is_cache_full(self, layer_id):
        return len(self.order) >= self.cache_size

    def get(self, layer_id, expert_id):
        if (layer_id, expert_id) in self.cache:
            self.order.remove((layer_id, expert_id))
            self.order.append((layer_id, expert_id))
            return self.cache[(layer_id, expert_id)]
        return None
    
    def get_layer_cache_size(self, layer_id):
        return len([key for key in self.cache if key[0] == layer_id])

    def add(self, layer_id, expert_id, expert_layer):
        self.order.append((layer_id, expert_id))
        self.cache[(layer_id, expert_id)] = expert_layer

    def replace(self, layer_id, expert_id):
        if not self.order:
            return 

        to_evict = self.order.pop(0)

        self.order.append((layer_id, expert_id))

        expert_layer = self.cache.pop(to_evict)

        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.cache[(layer_id, expert_id)] = expert_layer

        return expert_layer

    def evict(self, layer_id):
        oldest = self.order.pop(0)
        del self.cache[oldest]

class MarkovCache:
    """
    MarkovCache is a class that implements a Markov chain-based cache.
    It uses the LRUCache class to manage the cache and provides methods to access and update the cache.
    """

    def __init__(self, cache_size: int = 16, n_layers: int = 16):
        self.cache_size = cache_size * n_layers
        self.order = []  
        self.cache = {}  

    def is_cache_full(self, layer_id):
        return len(self.order) >= self.cache_size

    def get(self, layer_id, expert_id):
        if (layer_id, expert_id) in self.cache:
            self.order.remove((layer_id, expert_id))
            self.order.append((layer_id, expert_id))
            return self.cache[(layer_id, expert_id)]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order.append((layer_id, expert_id))
        self.cache[(layer_id, expert_id)] = expert_layer

    def evict(self, layer_id):
        oldest = self.order.pop(0)
        del self.cache[oldest]



class UnifiedMarkovCache:
    """
    UnifiedMarkovCache is a class that implements a unified Markov chain-based cache for both CPU and GPU.
    It uses the MarkovCache class to manage the cache and provides methods to access and update the cache.
    """

    def __init__(self, cache_size: int = 16, n_layers: int = 16):
        self.cache_size = cache_size * n_layers
        self.n_layers = n_layers
        self.n = 5  
        self.order = []
        self.cache = {}  
        self.markov_model = {i: defaultdict(Counter) for i in range(n_layers)}
        self.history = {i: deque(maxlen=self.n) for i in range(n_layers)}

    def is_cache_full(self, layer_id):
        """Check if cache for a specific layer is full."""
        return len(self.cache) >= self.cache_size

    def get(self, layer_id, expert_id):
        """Get an expert layer from cache."""
        if (layer_id, expert_id) in self.order:
            self.order.remove((layer_id, expert_id))
            self.order.append((layer_id, expert_id))
            return self.cache[(layer_id, expert_id)]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        """Add an expert layer to the cache."""
        self.order.append((layer_id, expert_id))
        self.cache[(layer_id, expert_id)] = expert_layer

    def update_history(self, layer_id, selected_experts):
        """Update the history of expert usage (deque handles length automatically)."""
        hist = self.history[layer_id]

        if len(hist) == self.n - 1:
            context = tuple(tuple(x) for x in hist)
            next_step = tuple(selected_experts)
            self.markov_model[layer_id][context][next_step] += 1

        hist.append(selected_experts)  

    def evict(self, layer_id):
        """Evict at least one expert layer using Markov prediction or fallback to LRU."""
        min_future_access_page = min(
            (expert for expert in self.cache if expert[0] == layer_id and expert[1] not in self.history[layer_id][0]),
            key=lambda expert: self.predict_next_usage(layer_id, expert, self.n),
            default=None
        )
        
        if min_future_access_page:
            key = min_future_access_page
            del self.cache[key]
            if key in self.order:
                self.order.remove(key)

        if not min_future_access_page and self.order:
            print("lru evict")
            key = self.order.pop(0)
            del self.cache[key]

    def predict_next_usage(self, layer_id, expert, n):
        """
        Predict the next usage of expert layers using Markov model.
        Return the likelihood of the expert's next usage.
        """
        context = tuple(self.history[layer_id])[-(self.n-1):] 
        if layer_id in self.markov_model and context in self.markov_model[layer_id]:
            next_usage = self.markov_model[layer_id][context]
            return next_usage.get(expert[1], 0)  
        return float('inf') 




class ProactiveCache:
    """
    ProactiveCache is a class that implements a proactive caching strategy.
    It uses the LRUCache class to manage the cache and provides methods to access and update the cache.
    """

    def __init__(self, cache_size: int = 16):
        """
        Initialize the ProactiveCache with a maximum size.

        :param cache_size: The maximum number of items in the cache. Default is 100.
        """
        self.cache = LRUCache(cache_size)


class UnifiedProactiveCache:
    """
    UnifiedProactiveCache is a class that implements a unified proactive caching strategy for both CPU and GPU.
    It uses the ProactiveCache class to manage the cache and provides methods to access and update the cache.
    """

    def __init__(self, cache_size: int = 16):
        """
        Initialize the UnifiedProactiveCache with a maximum size.

        :param cache_size: The maximum number of items in the cache. Default is 100.
        """
        self.cpu_cache = ProactiveCache(cache_size)
        self.gpu_cache = ProactiveCache(cache_size)

class ArithmeticCache:
    """
    ArithmeticCache is a class that implements an arithmetic caching strategy.
    It uses a deque to maintain the history of expert usage and provides methods to access and update the cache.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size
        self.n = 10
        self.history = {i: deque(maxlen=self.n) for i in range(n_layers)}
        self.order = [[] for _ in range(n_layers)] 
        self.cache = [{} for _ in range(n_layers)]  
        self.expert_path = expert_path

    def is_cache_full(self, layer_id):
        return len(self.order[layer_id]) >= self.cache_size
    
    def update_arithmetic(self, layer_id, all_expert_info):
        hist = self.history[layer_id]
        hist.append(all_expert_info)

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            self.order[layer_id].remove(expert_id)
            self.order[layer_id].append(expert_id)
            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order[layer_id].append(expert_id)
        self.cache[layer_id][expert_id] = expert_layer

    def replace(self, layer_id, expert_id):
        if not self.order[layer_id]:
            return 

        hist = self.history[layer_id]
        if not hist:
            to_evict = self.order[layer_id].pop(0)

        else:
            sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()  
            
            min_importance = float('inf')
            to_evict = None
            for expert in self.order[layer_id]:
                importance = sum_vec[expert]
                if importance < min_importance:
                    min_importance = importance
                    to_evict = expert

        self.order[layer_id].remove(to_evict)
        self.order[layer_id].append(expert_id)

        expert_layer = self.cache[layer_id].pop(to_evict)
        
        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.cache[layer_id][expert_id] = expert_layer

        return expert_layer
    
    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])



    def evict(self, layer_id):
        if not self.order[layer_id]:
            return  

        hist = self.history[layer_id]
        if not hist:
            oldest = self.order[layer_id].pop(0)
            del self.cache[layer_id][oldest]
            return

        sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()  
        
        min_importance = float('inf')
        to_evict = None
        for expert in self.order[layer_id]:
            importance = sum_vec[expert]
            if importance < min_importance:
                min_importance = importance
                to_evict = expert

        self.order[layer_id].remove(to_evict)
        del self.cache[layer_id][to_evict]




class UnifiedArithmeticCache:
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size * n_layers
        self.n = 7
        self.n_layers = n_layers
        self.history = {i: deque(maxlen=self.n) for i in range(n_layers)}
        self.expert_path = expert_path

        self.global_order = []  
        self.global_cache = {}  

    def update_arithmetic(self, layer_id, all_expert_info):
        self.history[layer_id].append(all_expert_info)

    def get(self, layer_id, expert_id):
        key = (layer_id, expert_id)
        if key in self.global_cache:
            self.global_order.remove(key)
            self.global_order.append(key)
            return self.global_cache[key]
        return None
    
    def is_cache_full(self, layer_id):
        return len(self.global_cache) >= self.cache_size


    def add(self, layer_id, expert_id, expert_layer):
        key = (layer_id, expert_id)
        self.global_order.append(key)
        self.global_cache[key] = expert_layer

    def get_layer_cache_size(self, layer_id):
        return len([key for key in self.global_cache if key[0] == layer_id])

    def replace(self, layer_id, expert_id):
        if not self.global_order:
            return

        key = (layer_id, expert_id)

        importance_scores = {}
        for layer_idx in range(self.n_layers):
            hist = self.history[layer_idx]
            if not hist:
                continue
            sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()
            for expert_idx in range(len(sum_vec)):
                importance_scores[(layer_idx, expert_idx)] = sum_vec[expert_idx]

        min_importance = float('inf')
        to_evict = None
        for key_idx in self.global_order:
            if importance_scores[key_idx] < min_importance:
                min_importance=importance_scores[key_idx]
                to_evict = key_idx

        self.global_order.remove(to_evict)
        self.global_order.append(key)

        expert_layer = self.global_cache.pop(to_evict)

        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.global_cache[key] = expert_layer

        return expert_layer

    def evict(self, layer_id):
        if not self.global_order:
            return

        importance_scores = {}
        for layer_id in range(self.n_layers):
            hist = self.history[layer_id]
            if not hist:
                continue
            sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()
            for expert_id in range(len(sum_vec)):
                importance_scores[(layer_id, expert_id)] = sum_vec[expert_id]

        min_importance = float('inf')
        to_evict = None
        for key in self.global_order:
            if importance_scores[key] < min_importance:
                min_importance=importance_scores[key]
                to_evict = key

        self.global_order.remove(to_evict)
        del self.global_cache[to_evict]



class ArithmeticCache:
    """
    ArithmeticCache is a class that implements an arithmetic caching strategy.
    It uses a deque to maintain the history of expert usage and provides methods to access and update the cache.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size
        self.n = 10
        self.history = {i: deque(maxlen=self.n) for i in range(n_layers)}
        self.order = [[] for _ in range(n_layers)]  
        self.cache = [{} for _ in range(n_layers)]  
        self.expert_path = expert_path

    def is_cache_full(self, layer_id):
        return len(self.order[layer_id]) >= self.cache_size
    
    def update_arithmetic(self, layer_id, all_expert_info):
        hist = self.history[layer_id]
        hist.append(all_expert_info)

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            self.order[layer_id].remove(expert_id)
            self.order[layer_id].append(expert_id)
            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order[layer_id].append(expert_id)
        self.cache[layer_id][expert_id] = expert_layer

    def replace(self, layer_id, expert_id):
        if not self.order[layer_id]:
            return  

        hist = self.history[layer_id]
        if not hist:
            to_evict = self.order[layer_id].pop(0)

        else:
            sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()  
            
            min_importance = float('inf')
            to_evict = None
            for expert in self.order[layer_id]:
                importance = sum_vec[expert]
                if importance < min_importance:
                    min_importance = importance
                    to_evict = expert

        self.order[layer_id].remove(to_evict)
        self.order[layer_id].append(expert_id)

        expert_layer = self.cache[layer_id].pop(to_evict)
        
        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.cache[layer_id][expert_id] = expert_layer

        return expert_layer
    
    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])



    def evict(self, layer_id):
        if not self.order[layer_id]:
            return 

        hist = self.history[layer_id]
        if not hist:
            oldest = self.order[layer_id].pop(0)
            del self.cache[layer_id][oldest]
            return

        sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()  
        
        min_importance = float('inf')
        to_evict = None
        for expert in self.order[layer_id]:
            importance = sum_vec[expert]
            if importance < min_importance:
                min_importance = importance
                to_evict = expert

        self.order[layer_id].remove(to_evict)
        del self.cache[layer_id][to_evict]



class MixtureCache:
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        self.cache_size = cache_size
        self.n = 10
        self.history = {i: deque(maxlen=self.n) for i in range(n_layers)}
        self.order = [[] for _ in range(n_layers)]
        self.cache = [{} for _ in range(n_layers)]
        self.expert_path = expert_path
        
    def is_cache_full(self, layer_id):
        return len(self.order[layer_id]) >= self.cache_size
    
    def update_arithmetic(self, layer_id, all_expert_info):
        hist = self.history[layer_id]
        hist.append(all_expert_info)

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            self.order[layer_id].remove(expert_id)
            self.order[layer_id].append(expert_id)
            return self.cache[layer_id][expert_id]
        return None
    
    def add(self, layer_id, expert_id, expert_layer):
        self.order[layer_id].append(expert_id)
        self.cache[layer_id][expert_id] = expert_layer


    def replace(self, layer_id, expert_id):
        if not self.order[layer_id]:
            return
        
        hist = self.history[layer_id]
        if not hist:
            to_evict = self.order[layer_id].pop(0)
        else:
            sum_vec = torch.stack(list(hist), dim=0).sum(dim=0).squeeze(0).tolist()  

            min_importance = float('inf')   
            to_evict = None
            for expert in self.order[layer_id]:
                importance = sum_vec[expert]
                if importance < min_importance:
                    min_importance = importance
                    to_evict = expert
        self.order[layer_id].remove(to_evict)
        self.order[layer_id].append(expert_id)
        expert_layer = self.cache[layer_id].pop(to_evict)
        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)
        self.cache[layer_id][expert_id] = expert_layer

        return expert_layer
    
    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])
    
    def evict(self, layer_id):
        if not self.order[layer_id]:
            return
        hist = self.history[layer_id]
        if not hist:
            oldest = self.order[layer_id].pop(0)
            del self.cache[layer_id][oldest]
            return



class VictimCache:
    def __init__(self, total_cache_size: int = 16, n_layers: int = 16, expert_path: str = ""):
        assert total_cache_size % 2 == 0, "total_cache_size must be divisible by 2"

        self.n_layers = n_layers
        
        self.main_cache_size = total_cache_size // 4
        self.victim_cache_size = 3* total_cache_size // 4

        self.expert_path = expert_path

        self.main_order = [[] for _ in range(n_layers)]
        self.victim_order = []

        self.cache = {}

    def get_layer_cache_size(self, layer_id):
        return self.main_cache_size +len([key for key in self.cache if key[0] == layer_id])

    def get(self, layer_id, expert_id):
        if expert_id in self.main_order[layer_id]:
            self.main_order[layer_id].remove(expert_id)
            self.main_order[layer_id].append(expert_id)
            return self.cache[(layer_id, expert_id)]

        elif (layer_id, expert_id) in self.victim_order:
            expert_layer = self.cache[(layer_id, expert_id)]
            self.victim_order.remove((layer_id, expert_id))

            if len(self.main_order[layer_id]) >= self.main_cache_size:
                old_expert_id = self.main_order[layer_id].pop(0)                
                assert len(self.victim_order) <= self.victim_cache_size * self.n_layers
                self.victim_order.append((layer_id, old_expert_id))

            self.main_order[layer_id].append(expert_id)
            return expert_layer

        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.cache[(layer_id,expert_id)] = expert_layer
        self.main_order[layer_id].append(expert_id)
        if len(self.main_order[layer_id]) > self.main_cache_size:
            to_evict = self.main_order[layer_id].pop(0)
            self.victim_order.append((layer_id, to_evict))

    def is_cache_full(self, layer_id):
        return (
            len(self.main_order[layer_id]) >= self.main_cache_size and
            len(self.victim_order) >= self.victim_cache_size * self.n_layers
        )

    def replace(self, layer_id, expert_id):

        to_evict = self.victim_order.pop(0)
        expert_layer = self.cache.pop(to_evict)

        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.add(layer_id, expert_id, expert_layer)
        return expert_layer


from collections import deque, Counter

class LockupCache:
    def __init__(self, cache_size=16, n_layers=16, expert_path="", locked_limit=3, locked_size=10):
        self.cache_size = cache_size
        self.expert_path = expert_path
        self.order = [[] for _ in range(n_layers)]
        self.cache = [{} for _ in range(n_layers)]
        self.locked = [set() for _ in range(n_layers)]
        self.locked_limit = locked_limit
        self.locked_size = locked_size

        self.usage_history = [deque(maxlen=20) for _ in range(n_layers)]

        self.cache_size = [37,32,30,27,27,26,19,19,22,21,24,26,22,27,23,20,23,26,18,17,20,21,23,25,23,28,23,20,23,25,19,17,19,20,24,26,24,28,25,22,26,29,27,24,24,23,24,34]

    def is_cache_full(self, layer_id):
        return len(self.order[layer_id]) >= self.cache_size[layer_id]

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            self.usage_history[layer_id].append(expert_id)
            self._update_locked_from_usage(layer_id)

            self.order[layer_id].remove(expert_id)
            self.order[layer_id].append(expert_id)

            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order[layer_id].append(expert_id)
        self.cache[layer_id][expert_id] = expert_layer

    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])

    def replace(self, layer_id, expert_id):
        if not self.order[layer_id]:
            return
        
        evict_candidate = None
        for candidate in self.order[layer_id]:
            if candidate not in self.locked[layer_id]:
                evict_candidate = candidate
                break
        
        if evict_candidate is None:
            evict_candidate = self.order[layer_id][0]

        self.order[layer_id].remove(evict_candidate)
        self.order[layer_id].append(expert_id)

    
        expert_layer = self.cache[layer_id].pop(evict_candidate)

        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.cache[layer_id][expert_id] = expert_layer

        return expert_layer


    def _update_locked_from_usage(self, layer_id):
        counts = Counter(self.usage_history[layer_id])
        total = len(self.usage_history[layer_id])
        locked_set = set()
        for expert_id, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt > self.locked_limit and len(locked_set) < self.locked_size:
                locked_set.add(expert_id)

        self.locked[layer_id] = locked_set


class UnifiedLockupCache:
    def __init__(self, cache_size=16, n_layers=16, expert_path="", locked_limit=3, locked_size=10):
        self.cache_size = cache_size * n_layers
        self.expert_path = expert_path
        self.order = []
        self.cache = {}
        self.locked = set ()
        self.locked_limit = locked_limit * n_layers
        self.locked_size = locked_size

        self.usage_history = deque(maxlen=20)

    def is_cache_full(self, layer_id):
        return len(self.cache) >= self.cache_size

    def get(self, layer_id, expert_id):
        if (layer_id, expert_id) in self.cache:
            self.usage_history.append((layer_id, expert_id))
            self._update_locked_from_usage()

            self.order.remove((layer_id, expert_id))
            self.order.append((layer_id, expert_id))

            return self.cache[(layer_id, expert_id)]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order.append((layer_id, expert_id))
        self.cache[(layer_id, expert_id)] = expert_layer

    def get_layer_cache_size(self, layer_id):
        return sum(1 for k in self.cache if k[0] == layer_id)

    def replace(self, layer_id, expert_id):
        if not self.order:
            return
        
        evict_candidate = None
        for candidate in self.order:
            if candidate not in self.locked:
                evict_candidate = candidate
                break
        
        if evict_candidate is None:
            evict_candidate = self.order[0]

        self.order.remove(evict_candidate)
        self.order.append((layer_id, expert_id))

    
        expert_layer = self.cache.pop(evict_candidate)

        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        expert_layer.load_state_dict(state_dict)

        self.cache[(layer_id, expert_id)] = expert_layer

        return expert_layer


    def _update_locked_from_usage(self):
        counts = Counter(self.usage_history)
        total = len(self.usage_history)
        locked_set = set()
        for (layer_id, expert_id), cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt > self.locked_limit and len(locked_set) < self.locked_size:
                locked_set.add((layer_id, expert_id))

        self.locked = locked_set

class LRULFUCache:
    """
    LRULFUCache is a class that implements a Least Recently Used (LRU) cache. 
    It uses a dictionary to store the cached items and a list to maintain the order of usage.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = "", beta: int= 30.0):
        self.cache_size = cache_size
        self.expert_path = expert_path
        self.order = [[] for _ in range(n_layers)]  
        self.cache = [{} for _ in range(n_layers)]  
        self.recent_access = [deque() for _ in range(n_layers)]  
        self.freq_counter = [defaultdict(int) for _ in range(n_layers)]
        self.alpha = 1.0
        self.beta = [100, 7.5, 7.5, 3.75, 7.5,
                    7.5, 15, 7.5, 7.5, 7.5,
                    7.5, 7.5, 7.5, 3.75, 3.75,
                    3.75, 15, 7.5, 7.5, 7.5,
                    1.9, 3.75, 7.5, 3.75, 7.5,
                    7.5, 7.5, 7.5, 15, 3.75,
                    7.5, 3.75, 7.5, 7.5, 3.75,
                    7.5, 7.5, 7.5, 7.5, 7.5,
                    15, 3.75, 7.5, 15, 3.75,
                    7.5, 15, 100] 
        self.window_size = [float('inf'), float('inf'), 320, 160, 160,
                            160, 160, 160, 160, 160,
                            160, 160, 160, 160, 160,
                            160, 320, 160, 160, 160,
                            160, 160, 160, 160, 160,
                            160, 160, 160, 160, 160,
                            160, 160, 160, 160, 160,
                            160, 160, 160, 160, 160,
                            160, 160, 160, 320, 640,
                            float('inf'), 640, 1280] 

    def is_cache_full(self, layer_id):
        return len(self.order[layer_id]) >= self.cache_size

    def get(self, layer_id, expert_id):
        self.update_recent_access(layer_id, expert_id)
        if expert_id in self.cache[layer_id]:
            self.order[layer_id].remove(expert_id)
            self.order[layer_id].append(expert_id)
            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.order[layer_id].append(expert_id)
        self.cache[layer_id][expert_id] = expert_layer

    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])
    

    def replace(self, layer_id, expert_id, state_dict):
        if not self.order[layer_id]:
            return 

        score_list = []

        for idx, expert in enumerate(self.order[layer_id]):
            freq = self.freq_counter[layer_id][expert]
            score = self.alpha * idx + self.beta[layer_id] * freq
            score_list.append((expert, score))

        to_evict, _ = min(score_list, key=lambda x: x[1])
        

        self.order[layer_id].append(expert_id)
        self.order[layer_id].remove(to_evict)

        expert_layer = self.cache[layer_id].pop(to_evict)

        expert_layer.load_state_dict(state_dict)

        self.cache[layer_id][expert_id] = expert_layer

        return expert_layer
    
    def evict(self, layer_id):
        oldest = self.order[layer_id].pop(0)
        del self.cache[layer_id][oldest]

    def update_recent_access(self, layer_id, expert_id):
        self.recent_access[layer_id].append(expert_id)
        self.freq_counter[layer_id][expert_id] += 1
        if len(self.recent_access[layer_id]) > self.window_size[layer_id]:
            removed = self.recent_access[layer_id].popleft()
            self.freq_counter[layer_id][removed] -= 1
            if self.freq_counter[layer_id][removed] == 0:
                del self.freq_counter[layer_id][removed]

class MLCache:
    """
    MLCache is a class that implements a Least Recently Used (LRU) cache.
    It uses a dictionary to store the cached items and a list to maintain the order of usage.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = "", num_experts: int = 128, ffn_model_path: str = None):
        self.expert_path = expert_path
        self.num_experts = int(num_experts)
        self.ffn_model_path = ffn_model_path
        self.cache = [{} for _ in range(n_layers)] 
        self.cache_size = [cache_size] * n_layers
        self.eviction_models = [EvictionScorer(num_experts=self.num_experts).to("cuda") for _ in range(n_layers)]
        self.n_layers = n_layers

        self.lru_scores = [torch.zeros((1, self.num_experts), device='cuda') for _ in range(n_layers)]
        self.lfu_scores = [torch.zeros((1, self.num_experts), device='cuda') for _ in range(n_layers)]
        self.scores = torch.zeros((1,self.num_experts))
        self.init_eviction_models()
        self.sorted_experts = [deque() for _ in range(n_layers)]

    @torch.no_grad()
    def update_arithmetic(self, layer_id, routed_experts):
        lru = self.lru_scores[layer_id]
        lfu = self.lfu_scores[layer_id]
    
        lru += 1.0
        lru[0, routed_experts] = 1.0
        lfu[0, routed_experts] += 1.0
    
        lru_score = 1.0 / (lru + 1e-6)              
        lfu_score = lfu / (lfu.max() + 1e-6)        

        x = torch.cat([lru_score.squeeze(0), lfu_score.squeeze(0)], dim=0).to(torch.float32).cuda()  
    
        scores = self.eviction_models[layer_id](x)  
        sorted_eids = torch.argsort(scores, descending=True).tolist()
        self.sorted_experts[layer_id] = deque(sorted_eids)


    def init_eviction_models(self):
        for i in range(len(self.eviction_models)):
            path = f"{self.ffn_model_path}/base_layer{i}.pt"
            state_dict = torch.load(path, map_location="cuda", weights_only=True)
            self.eviction_models[i].load_state_dict(state_dict)
            self.eviction_models[i].eval()

    def is_cache_full(self, layer_id):
        return len(self.cache[layer_id]) >= self.cache_size[layer_id]

    def get(self, layer_id, expert_id):
        if expert_id in self.cache[layer_id]:
            return self.cache[layer_id][expert_id]
        return None

    def add(self, layer_id, expert_id, expert_layer):
        self.cache[layer_id][expert_id] = expert_layer

    def get_layer_cache_size(self, layer_id):
        return len(self.cache[layer_id])

    def replace(self, layer_id, expert_id, expert_layer):
        while self.sorted_experts[layer_id]:
            candidate = self.sorted_experts[layer_id].popleft()
            if candidate in self.cache[layer_id]:
                to_evict = candidate
                break
        else:
            to_evict = next(iter(self.cache[layer_id]))

        evicted = self.cache[layer_id].pop(to_evict)
        del evicted
        torch.cuda.empty_cache()
        self.cache[layer_id][expert_id] = expert_layer



class MLCacheV2:
    """
    MLCacheV2 using MultiEvictionScorer from Evaluation_modified2.py.
    Uses two FFNs (short-term and long-term) with weight blending based on cache size.
    Optimized: GPU-resident scores, cached blended scores, batch operations.
    """
    def __init__(self, cache_size: int = 16, n_layers: int = 16, expert_path: str = "", num_experts: int = 128, ffn_model_path: str = None):
        self.expert_path = expert_path
        self.ffn_model_path = ffn_model_path
        self.cache = [{} for _ in range(n_layers)] 
        self.cache_size = [int(cache_size)] * int(n_layers)
        self.num_experts = int(num_experts)
        self.n_layers = int(n_layers)
        self.first_k_dense_replace = 0 

        self.eviction_models = [
            MultiEvictionScorer(input_dim=2, hidden_dim=32, num_layers=3).to("cuda")
            for _ in range(self.n_layers)
        ]

        self.lru_scores = [torch.zeros((self.num_experts,), device="cuda") for _ in range(self.n_layers)]
        self.lfu_scores = [torch.zeros((self.num_experts,), device="cuda") for _ in range(self.n_layers)]

        self.score_short = [torch.zeros((self.num_experts,), device="cuda") for _ in range(self.n_layers)]
        self.score_long  = [torch.zeros((self.num_experts,), device="cuda") for _ in range(self.n_layers)]
        self.blended_scores = [torch.zeros((self.num_experts,), device="cuda") for _ in range(self.n_layers)]

        self.blend_weights = []
        for cs in self.cache_size:
            if cs >= 24:
                w1, w2 = 0.5, 0.5
            else:
                w2 = 0.5 * (cs / 24)
                w1 = 1.0 - w2
            self.blend_weights.append((float(w1), float(w2)))

        self.cache_mask = [
            torch.zeros((self.num_experts,), dtype=torch.bool, device="cuda")
            for _ in range(self.n_layers)
        ]

        self.sorted_experts = [None for _ in range(self.n_layers)]

        self.init_eviction_models()

    @torch.no_grad()
    def update_arithmetic(self, layer_id: int, routed_experts: Union[torch.Tensor, List[int]]):
        """
        Original semantics:
          1) Update LRU/LFU counters
          2) Compute lru_score/lfu_score
          3) Run eviction model -> (score_short, score_long)
          4) blended = w1*short + w2*long
          5) argsort over ALL experts (no masking here)
        """
        layer_id = int(layer_id)
        lru = self.lru_scores[layer_id]
        lfu = self.lfu_scores[layer_id]

        lru += 1.0

        if isinstance(routed_experts, torch.Tensor):
            idx = routed_experts.to(device="cuda", dtype=torch.long).flatten()
        else:
            idx = torch.as_tensor(routed_experts, device="cuda", dtype=torch.long).flatten()

        if idx.numel() > 0:
            lru[idx] = 1.0
            lfu[idx] += 1.0

        lru_score = 1.0 / (lru + 1e-6)          
        lfu_max = lfu.max() + 1e-6             
        lfu_score = lfu / lfu_max              

        features = torch.stack([lru_score, lfu_score], dim=-1).to(torch.float32)  

        score_short, score_long = self.eviction_models[layer_id](features)

        self.score_short[layer_id] = score_short
        self.score_long[layer_id] = score_long

        w1, w2 = self.blend_weights[layer_id]
        blended = w1 * score_short + w2 * score_long
        self.blended_scores[layer_id] = blended

        self.sorted_experts[layer_id] = torch.argsort(blended, descending=True)

    def init_eviction_models(self):
        """
        Keep your existing loading logic if you want.
        You said "path/load isn't what you mean by identical behavior".
        So below is a minimal safe default that does NOTHING if ffn_model_path is None.
        Replace with your actual loader.
        """
        import os
        import json

        if self.ffn_model_path is None:
            raise ValueError("ffn_model_path must be provided for ML_CECAR cache policy")

        config_path = f"../../Evaluation/fiddler_model/Config/Qwen3_30B_A3B_config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"FFN model config not found: {config_path}\n"
                f"Please create a config.json with 'n_layers' and 'num_experts' fields."
            )

        with open(config_path, 'r') as f:
            ffn_config = json.load(f)

        config_n_layers = ffn_config.get('n_layers')
        config_num_experts = ffn_config.get('num_experts')
        self.first_k_dense_replace = ffn_config.get('first_k_dense_replace', 0)

        if config_n_layers is None or config_num_experts is None:
            raise ValueError(
                f"config.json must contain 'n_layers' and 'num_experts' fields. "
                f"Found: {ffn_config}"
            )

        expected_moe_layers = self.n_layers - self.first_k_dense_replace
        if config_n_layers != expected_moe_layers:
            raise ValueError(
                f"FFN model layer count mismatch: config has {config_n_layers} MoE layers, "
                f"but model has {expected_moe_layers} MoE layers (total {self.n_layers} - {self.first_k_dense_replace} dense). "
                f"FFN model path: {self.ffn_model_path}"
            )

        if config_num_experts != self.num_experts:
            raise ValueError(
                f"FFN model expert count mismatch: config has {config_num_experts} experts, "
                f"but model has {self.num_experts} experts. "
                f"FFN model path: {self.ffn_model_path}"
            )

        for i in range(len(self.eviction_models)):
            if i < self.first_k_dense_replace:
                continue
            file_idx = i - self.first_k_dense_replace
            path = f"{self.ffn_model_path}/base_layer{i}.pt"
            if not os.path.exists(path):
                raise FileNotFoundError(f"FFN eviction model not found: {path}")
            state_dict = torch.load(path, map_location="cuda", weights_only=True)
            self.eviction_models[i].load_state_dict(state_dict)
            self.eviction_models[i].eval()

    def is_cache_full(self, layer_id: int) -> bool:
        layer_id = int(layer_id)
        return len(self.cache[layer_id]) >= self.cache_size[layer_id]

    def get(self, layer_id: int, expert_id: int):
        layer_id = int(layer_id)
        expert_id = int(expert_id)
        return self.cache[layer_id].get(expert_id, None)

    def add(self, layer_id: int, expert_id: int, expert_layer):
        layer_id = int(layer_id)
        expert_id = int(expert_id)
        self.cache[layer_id][expert_id] = expert_layer
        self.cache_mask[layer_id][expert_id] = True

    def get_layer_cache_size(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        return len(self.cache[layer_id])

    def replace(self, layer_id: int, expert_id: int, expert_layer):
        """
        Original semantics:
          - iterate sorted_experts (overall ranking) until find candidate that is in cache (dict membership)
          - if none found, fallback to next(iter(cache))
        """
        layer_id = int(layer_id)
        expert_id = int(expert_id)

        if len(self.cache[layer_id]) == 0:
            self.cache[layer_id][expert_id] = expert_layer
            self.cache_mask[layer_id][expert_id] = True
            return

        sorted_indices = self.sorted_experts[layer_id]
        to_evict = None

        if sorted_indices is not None and sorted_indices.numel() > 0:
            for cand in sorted_indices.tolist():
                if cand in self.cache[layer_id]:
                    to_evict = cand
                    break

        if to_evict is None:
            to_evict = next(iter(self.cache[layer_id]))

        self.cache_mask[layer_id][to_evict] = False
        evicted = self.cache[layer_id].pop(to_evict)
        del evicted


        self.cache[layer_id][expert_id] = expert_layer
        self.cache_mask[layer_id][expert_id] = True

    def get_eviction_score(self, layer_id: int, expert_id: int) -> float:
        """
        Mirror original flow: recompute blended from stored short/long each call.
        (Still identical to blended_scores if weights unchanged.)
        """
        layer_id = int(layer_id)
        expert_id = int(expert_id)
        w1, w2 = self.blend_weights[layer_id]
        total = w1 * self.score_short[layer_id] + w2 * self.score_long[layer_id]
        return float(total[expert_id].item())

    def get_eviction_scores_batch(self, layer_id: int, expert_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Return blended scores for a batch of expert_ids (GPU tensor).
        """
        layer_id = int(layer_id)
        if isinstance(expert_ids, list):
            expert_ids = torch.tensor(expert_ids, device="cuda", dtype=torch.long)
        else:
            expert_ids = expert_ids.to(device="cuda", dtype=torch.long)
        return self.blended_scores[layer_id][expert_ids]

    def get_cached_expert_ids_tensor(self, layer_id: int) -> torch.Tensor:
        layer_id = int(layer_id)
        return self.cache_mask[layer_id].nonzero(as_tuple=True)[0]



class EvictionScorer(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_experts = int(num_experts)
        input_dim = 2 * self.num_experts
        out_dim = self.num_experts

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = self.ffn(x)                  
        return torch.abs(y).squeeze(0)   


class MultiEvictionScorer(nn.Module):
    """
    MultiEvictionScorer from Evaluation_modified2.py
    Two separate FFNs for short-term and long-term eviction scoring.
    """
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=3):
        super().__init__()
        layers1 = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers1.append(nn.Linear(hidden_dim, 1))
        self.ffn_short = nn.Sequential(*layers1)

        layers2 = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers2 += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers2.append(nn.Linear(hidden_dim, 1))
        self.ffn_long = nn.Sequential(*layers2)

    def forward(self, x):  
        score_short = self.ffn_short(x) 
        score_long = self.ffn_long(x)    
        return torch.abs(score_short.squeeze(-1)), torch.abs(score_long.squeeze(-1)) 


import os
from collections import deque
from typing import Iterable, Optional
import torch
import torch.nn as nn

# -------------------------
# MultiEvictionScorer (same as training)
# -------------------------
class MultiEvictionScorer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=3):
        super().__init__()
        layers1 = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, num_layers - 2)):
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers1.append(nn.Linear(hidden_dim, 1))
        self.ffn_short = nn.Sequential(*layers1)

        layers2 = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, num_layers - 2)):
            layers2 += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers2.append(nn.Linear(hidden_dim, 1))
        self.ffn_long = nn.Sequential(*layers2)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            BTE, E, F = x.shape
            flat = x.view(BTE * E, F)
            s = self.ffn_short(flat).view(BTE, E)
            l = self.ffn_long(flat).view(BTE, E)
            return torch.abs(s), torch.abs(l)
        else:
            flat = x  
            s = self.ffn_short(flat).squeeze(-1)
            l = self.ffn_long(flat).squeeze(-1)
            return torch.abs(s), torch.abs(l)


class VirtualCache:
    """
    Virtual cache for simulation mode.
    Tracks cache state without storing actual expert weights.
    Supports batch-wise independent cache states.
    """

    def __init__(self, cache_policy: str = "lru", cache_size: int = 16,
                 n_layers: int = 48, batch_size: int = 1, num_experts: int = 128,
                 ffn_model_path: str = None):
        self.cache_policy = cache_policy
        self.cache_size = cache_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.num_experts = num_experts
        self.ffn_model_path = ffn_model_path

        self.cache_state = [
            [set() for _ in range(n_layers)]
            for _ in range(batch_size)
        ]

        self.lru_order = [
            [[] for _ in range(n_layers)]
            for _ in range(batch_size)
        ]

        self.lfu_counts = [
            [{} for _ in range(n_layers)]
            for _ in range(batch_size)
        ]

        self.hit_count = [[0] * n_layers for _ in range(batch_size)]
        self.total_count = [[0] * n_layers for _ in range(batch_size)]

        if cache_policy == "ML_CECAR":
            self.lru_scores = [
                [torch.zeros((1, num_experts), device='cpu') for _ in range(n_layers)]
                for _ in range(batch_size)
            ]
            self.lfu_scores = [
                [torch.zeros((1, num_experts), device='cpu') for _ in range(n_layers)]
                for _ in range(batch_size)
            ]
            self.sorted_experts = [
                [deque() for _ in range(n_layers)]
                for _ in range(batch_size)
            ]
            self.eviction_models = [
                MultiEvictionScorer(input_dim=2, hidden_dim=32, num_layers=3).to("cuda")
                for _ in range(n_layers)
            ]
            self._load_eviction_models()

    def _load_eviction_models(self):
        """Load pretrained eviction models"""
        import os
        import json

        if self.ffn_model_path is None:
            raise ValueError("ffn_model_path must be provided for ML_CECAR cache policy")

        config_path = f"{self.ffn_model_path}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"FFN model config not found: {config_path}\n"
                f"Please create a config.json with 'n_layers' and 'num_experts' fields."
            )

        with open(config_path, 'r') as f:
            ffn_config = json.load(f)

        config_n_layers = ffn_config.get('n_layers')
        config_num_experts = ffn_config.get('num_experts')

        if config_n_layers is None or config_num_experts is None:
            raise ValueError(
                f"config.json must contain 'n_layers' and 'num_experts' fields. "
                f"Found: {ffn_config}"
            )

        if config_n_layers != self.n_layers:
            raise ValueError(
                f"FFN model layer count mismatch: config has {config_n_layers} layers, "
                f"but model has {self.n_layers} layers. "
                f"FFN model path: {self.ffn_model_path}"
            )

        if config_num_experts != self.num_experts:
            raise ValueError(
                f"FFN model expert count mismatch: config has {config_num_experts} experts, "
                f"but model has {self.num_experts} experts. "
                f"FFN model path: {self.ffn_model_path}"
            )

        loaded_count = 0
        for layer_id in range(self.n_layers):
            model_path = f"{self.ffn_model_path}/layer_{layer_id:02d}_model.pt"

            if os.path.exists(model_path):
                self.eviction_models[layer_id].load_state_dict(
                    torch.load(model_path, map_location="cuda", weights_only=True)
                )
                self.eviction_models[layer_id].eval()
                loaded_count += 1
            else:
                raise FileNotFoundError(f"FFN eviction model not found: {model_path}")

    def access(self, batch_idx: int, layer_id: int, expert_id: int) -> bool:
        """
        Simulate cache access.
        Returns True if hit, False if miss.
        Updates cache state accordingly.
        """
        is_hit = expert_id in self.cache_state[batch_idx][layer_id]
        self.total_count[batch_idx][layer_id] += 1

        if is_hit:
            self.hit_count[batch_idx][layer_id] += 1
            self._update_on_hit(batch_idx, layer_id, expert_id)
        else:
            self._update_on_miss(batch_idx, layer_id, expert_id)

        return is_hit

    def _update_on_hit(self, batch_idx: int, layer_id: int, expert_id: int):
        """Update state on cache hit"""
        if self.cache_policy == "lru":
            self.lru_order[batch_idx][layer_id].remove(expert_id)
            self.lru_order[batch_idx][layer_id].append(expert_id)
        elif self.cache_policy == "lfu":
            self.lfu_counts[batch_idx][layer_id][expert_id] += 1
        elif self.cache_policy == "ML_CECAR":
            self._update_ml_scores(batch_idx, layer_id, expert_id)

    def _update_on_miss(self, batch_idx: int, layer_id: int, expert_id: int):
        """Update state on cache miss (evict if needed, then add)"""
        if len(self.cache_state[batch_idx][layer_id]) >= self.cache_size:
            self._evict(batch_idx, layer_id)

        self.cache_state[batch_idx][layer_id].add(expert_id)

        if self.cache_policy == "lru":
            self.lru_order[batch_idx][layer_id].append(expert_id)
        elif self.cache_policy == "lfu":
            self.lfu_counts[batch_idx][layer_id][expert_id] = 1
        elif self.cache_policy == "ML_CECAR":
            self._update_ml_scores(batch_idx, layer_id, expert_id)

    def _evict(self, batch_idx: int, layer_id: int):
        """Evict one expert based on policy"""
        if self.cache_policy == "lru":
            to_evict = self.lru_order[batch_idx][layer_id].pop(0)
            self.cache_state[batch_idx][layer_id].remove(to_evict)

        elif self.cache_policy == "lfu":
            min_count = min(self.lfu_counts[batch_idx][layer_id].values())
            for eid, cnt in self.lfu_counts[batch_idx][layer_id].items():
                if cnt == min_count and eid in self.cache_state[batch_idx][layer_id]:
                    self.cache_state[batch_idx][layer_id].remove(eid)
                    del self.lfu_counts[batch_idx][layer_id][eid]
                    break

        elif self.cache_policy == "ML_CECAR":
            while self.sorted_experts[batch_idx][layer_id]:
                candidate = self.sorted_experts[batch_idx][layer_id].popleft()
                if candidate in self.cache_state[batch_idx][layer_id]:
                    self.cache_state[batch_idx][layer_id].remove(candidate)
                    return
            to_evict = next(iter(self.cache_state[batch_idx][layer_id]))
            self.cache_state[batch_idx][layer_id].remove(to_evict)

    def _update_ml_scores(self, batch_idx: int, layer_id: int, expert_id: int):
        """Update ML-based eviction scores"""
        lru = self.lru_scores[batch_idx][layer_id]
        lfu = self.lfu_scores[batch_idx][layer_id]

        lru += 1.0
        lru[0, expert_id] = 1.0
        lfu[0, expert_id] += 1.0

        lru_score = 1.0 / (lru + 1e-6)
        lfu_score = lfu / (lfu.max() + 1e-6)

        features = torch.stack([
            lru_score.squeeze(0),
            lfu_score.squeeze(0)
        ], dim=-1).to(torch.float32).cuda()

        with torch.no_grad():
            score_short, score_long = self.eviction_models[layer_id](features)

        if self.cache_size >= 24:
            w1, w2 = 0.5, 0.5
        else:
            w2 = 0.5 * (self.cache_size / 24)
            w1 = 1.0 - w2

        total_score = w1 * score_short.cpu() + w2 * score_long.cpu()
        sorted_eids = torch.argsort(total_score, descending=True).tolist()
        self.sorted_experts[batch_idx][layer_id] = deque(sorted_eids)

        self.lru_scores[batch_idx][layer_id] = lru
        self.lfu_scores[batch_idx][layer_id] = lfu

    def add_to_cache(self, batch_idx: int, layer_id: int, expert_id: int):
        """Directly add to cache (for prefill initialization)"""
        if expert_id not in self.cache_state[batch_idx][layer_id]:
            if len(self.cache_state[batch_idx][layer_id]) >= self.cache_size:
                self._evict(batch_idx, layer_id)

            self.cache_state[batch_idx][layer_id].add(expert_id)

            if self.cache_policy == "lru":
                self.lru_order[batch_idx][layer_id].append(expert_id)
            elif self.cache_policy == "lfu":
                self.lfu_counts[batch_idx][layer_id][expert_id] = 1

    def is_cached(self, batch_idx: int, layer_id: int, expert_id: int) -> bool:
        """Check if expert is in cache without updating state"""
        return expert_id in self.cache_state[batch_idx][layer_id]

    def get_cached_expert_ids(self, batch_idx: int, layer_id: int) -> list:
        """Get list of cached expert IDs for a batch/layer"""
        return list(self.cache_state[batch_idx][layer_id])

    def get_eviction_score(self, batch_idx: int, layer_id: int, expert_id: int) -> float:
        """
        Get eviction score for a specific expert.
        Higher score = more likely to be evicted.
        Used by banpick bonus strategy.
        """
        if self.cache_policy == "lru":
            # LRU: position in order (0 = oldest = highest eviction priority)
            if expert_id in self.lru_order[batch_idx][layer_id]:
                idx = self.lru_order[batch_idx][layer_id].index(expert_id)
                # Invert so that oldest has highest score
                return len(self.lru_order[batch_idx][layer_id]) - idx
            return 0.0

        elif self.cache_policy == "lfu":
            # LFU: inverse of frequency (lower freq = higher eviction score)
            if expert_id in self.lfu_counts[batch_idx][layer_id]:
                freq = self.lfu_counts[batch_idx][layer_id][expert_id]
                max_freq = max(self.lfu_counts[batch_idx][layer_id].values()) if self.lfu_counts[batch_idx][layer_id] else 1
                return 1.0 - (freq / max_freq)
            return 1.0

        elif self.cache_policy == "ML_CECAR":
            # ML_CECAR: use the sorted position
            sorted_list = list(self.sorted_experts[batch_idx][layer_id])
            if expert_id in sorted_list:
                idx = sorted_list.index(expert_id)
                return 1.0 - (idx / len(sorted_list)) if sorted_list else 0.0
            return 0.0

        return 0.0

    def reset(self, batch_idx: int = None):
        """Reset cache state. If batch_idx is None, reset all batches."""
        if batch_idx is not None:
            for layer_id in range(self.n_layers):
                self.cache_state[batch_idx][layer_id].clear()
                self.lru_order[batch_idx][layer_id].clear()
                self.lfu_counts[batch_idx][layer_id].clear()
                self.hit_count[batch_idx][layer_id] = 0
                self.total_count[batch_idx][layer_id] = 0
                if self.cache_policy == "ML_CECAR":
                    self.lru_scores[batch_idx][layer_id].zero_()
                    self.lfu_scores[batch_idx][layer_id].zero_()
                    self.sorted_experts[batch_idx][layer_id].clear()
        else:
            for b in range(self.batch_size):
                self.reset(b)

    def get_hit_rate(self, batch_idx: int = None) -> float:
        """Get hit rate. If batch_idx is None, return aggregate."""
        if batch_idx is not None:
            total = sum(self.total_count[batch_idx])
            hits = sum(self.hit_count[batch_idx])
            return hits / total if total > 0 else 0.0
        else:
            total = sum(sum(tc) for tc in self.total_count)
            hits = sum(sum(hc) for hc in self.hit_count)
            return hits / total if total > 0 else 0.0

    def get_hit_rate_by_layer(self, batch_idx: int = None) -> list:
        """Get hit rate per layer"""
        if batch_idx is not None:
            return [
                self.hit_count[batch_idx][l] / self.total_count[batch_idx][l]
                if self.total_count[batch_idx][l] > 0 else 0.0
                for l in range(self.n_layers)
            ]
        else:
            result = []
            for l in range(self.n_layers):
                total = sum(self.total_count[b][l] for b in range(self.batch_size))
                hits = sum(self.hit_count[b][l] for b in range(self.batch_size))
                result.append(hits / total if total > 0 else 0.0)
            return result

