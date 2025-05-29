"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist # Added for Tensor Parallelism

# --- Tensor Parallel Utilities (New) ---
# Based on concepts from Megatron-LM and other TP implementations

def _ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"

def _split_along_last_dim(input_, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension."""
    _ensure_divisibility(input_.size(-1), num_partitions)
    # Get the size and dimension.
    last_dim = input_.dim() - 1
    last_dim_size = input_.size(last_dim) // num_partitions
    # Split.
    input_list = torch.split(input_, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in input_list)
    return input_list

class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""
    @staticmethod
    def symbolic(graph, input_, tp_group_): # tp_group_ is not used in symbolic
        return input_
    
    @staticmethod
    def forward(ctx, input_, tp_group_):
        ctx.tp_group = tp_group_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tp_group is not None and dist.get_world_size(group=ctx.tp_group) > 1:
            dist.all_reduce(grad_output, group=ctx.tp_group)
        return grad_output, None

class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""
    @staticmethod
    def symbolic(graph, input_, tp_group_): # tp_group_ is not used in symbolic
        #象征性地，它只是传递张量，all_reduce将在实际执行中发生
        return input_

    @staticmethod
    def forward(ctx, input_, tp_group_):
        if tp_group_ is not None and dist.get_world_size(group=tp_group_) > 1:
            dist.all_reduce(input_, group=tp_group_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None # Gradient is passed straight-through

class _GatherToParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""
    @staticmethod
    def symbolic(graph, input_, tp_group_, dim_): # tp_group_ is not used in symbolic
        # This is tricky for ONNX export if dims change. For now, assume it's handled by the execution.
        return input_ # Simplified for now

    @staticmethod
    def forward(ctx, input_, tp_group_, dim_):
        ctx.tp_group = tp_group_
        ctx.dim = dim_
        world_size = dist.get_world_size(group=tp_group_) if tp_group_ is not None else 1
        if world_size > 1:
            # Create a list of tensors to gather.
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            dist.all_gather(tensor_list, input_.contiguous(), group=tp_group_) # input_ must be contiguous
            # Concatenate along the specified dimension.
            output = torch.cat(tensor_list, dim=dim_)
        else:
            output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size(group=ctx.tp_group) if ctx.tp_group is not None else 1
        if world_size > 1:
            # Split the gradient along the gathering dimension.
            # This is the reverse of the cat operation.
            # Each process gets its corresponding chunk of the gradient.
            # The dim might be different from the original split dim in some cases,
            # so careful handling or using _ReduceFromParallelRegion might be more robust for grads.
            # For now, assume grad can be split like the forward input was effectively "split" for each rank.
            rank = dist.get_rank(group=ctx.tp_group)
            # Assuming grad_output needs to be split to match the original input_ to the forward of this rank
            # This part is complex and depends on how the output was used.
            # A simpler conceptual backward for gather might be to just pass the gradient and use reduce elsewhere,
            # or each rank takes its "slice" of the gradient if the gathered output was used in a way that
            # maintains spatial correspondence for the gradient.
            # Let's assume the gradient is split along the same dimension.
            # This is often not what you want directly. Often, the gradient for a gather op
            # would be a scatter operation (the adjoint).
            # A common pattern is that the layer consuming the gathered output will produce a gradient
            # that should be all-reduced. Or, if the gathered output goes into a RowParallelLinear,
            # then the gradient path handles itself appropriately.
            # For simplicity in this example, we'll assume the gradient is reduced by the upstream op
            # or this layer isn't the one needing to split for backprop directly for TP.
            # This needs a more specific TP pattern to be correct.
            # A common approach: the backward of all_gather is reduce_scatter or split + identity.
            # Let's use the split + identity for the gradient.
            # It needs to be split in the same way the original tensor was split before gather.
            # This is a simplification. The true adjoint of gather is split.
            output_grad_chunks = _split_along_last_dim(grad_output, world_size) # Assuming last dim for gather here too
            return output_grad_chunks[rank], None, None

        return grad_output, None, None


# --- New Tensor Parallel Linear Layers ---
class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension (columns) as A = [A_1, A_2, ..., A_p].
    X is broadcast to all TP ranks. Each rank computes Y_i = XA_i.
    The outputs Y_i are concatenated along the second dimension to produce Y.
    """
    def __init__(self, in_features, out_features, bias=True, config=None,
                 keep_master_weight=False): # keep_master_weight for potential future use
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallel_size = config.tensor_parallel_size if config and hasattr(config, 'tensor_parallel_size') else 1
        self.tp_group = config.tp_group if config and hasattr(config, 'tp_group') else None

        _ensure_divisibility(self.out_features, self.tensor_parallel_size)
        self.output_features_per_partition = self.out_features // self.tensor_parallel_size

        self.weight = nn.Parameter(torch.empty(
            self.output_features_per_partition, self.in_features,
            device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu', # Get current device
            dtype=config.dtype if config and hasattr(config, 'dtype') else torch.float32
        ))
        # Initialize weight with specific method if needed (e.g. from GPT._init_weights)
        # For now, use default Kaiming uniform for empty tensor, or allow GPT init to handle it.
        # GPT._init_weights will handle this via self.apply if this module is part of GPT.

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_features_per_partition,
                device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
                dtype=config.dtype if config and hasattr(config, 'dtype') else torch.float32
            ))
            # Initialize bias (e.g., to zero)
            with torch.no_grad():
                self.bias.zero_() # Common initialization
        else:
            self.register_parameter('bias', None)
        
        # For weight sharding during load/save if master_weight is used
        self.master_weight = None # Placeholder

    def forward(self, input_):
        # Set input to be parallel (broadcast to all TP ranks)
        # The input is already on the correct device due to model.to(device)
        # and DDP/torchrun handling of inputs.
        # For TP, input needs to be identical across TP ranks for ColumnParallel.
        # This is usually handled by the data loader or the layer before it.
        # If not, we might need: input_parallel = _CopyToParallelRegion.apply(input_, self.tp_group)
        # but often, the input is already replicated.

        # Matrix multiply.
        output_parallel = F.linear(input_, self.weight, self.bias)
        if self.tensor_parallel_size > 1:
            # Gather the output if it's the last layer in a sequence or needs to be full size
            # For MLP, this output feeds into GELU, which is element-wise, so can stay sharded.
            # output = _GatherToParallelRegion.apply(output_parallel, self.tp_group, -1)
            # For typical MLP (fc -> gelu -> proj), output_parallel can be directly used by GELU
            # The subsequent RowParallelLinear will handle the sharded input.
            return output_parallel 
        else:
            return output_parallel


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension (rows) as A = [A_1^T, A_2^T, ..., A_p^T]^T.
    X is split along its second dimension as X = [X_1, X_2, ..., X_p]. Each rank i
    computes Y_i = X_i A_i. The results Y_i are summed using an all-reduce operation.
    Bias b is added to the output Y on rank 0 (or replicated and only rank 0 adds).
    """
    def __init__(self, in_features, out_features, bias=True, config=None,
                 input_is_parallel=True, keep_master_weight=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tensor_parallel_size = config.tensor_parallel_size if config and hasattr(config, 'tensor_parallel_size') else 1
        self.tp_group = config.tp_group if config and hasattr(config, 'tp_group') else None

        _ensure_divisibility(self.in_features, self.tensor_parallel_size)
        self.input_features_per_partition = self.in_features // self.tensor_parallel_size
        
        self.weight = nn.Parameter(torch.empty(
            self.out_features, self.input_features_per_partition,
            device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
            dtype=config.dtype if config and hasattr(config, 'dtype') else torch.float32
        ))
        # GPT._init_weights will handle this.

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.out_features,
                device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
                dtype=config.dtype if config and hasattr(config, 'dtype') else torch.float32
            ))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Input: Sharded along the last dimension if input_is_parallel is True
        # Or, input is full and needs to be sharded here (less common for RowParallel input)
        if self.input_is_parallel and self.tensor_parallel_size > 1:
            input_parallel = input_
        else:
            # If input is not parallel, it should be split.
            # This path might not be typical for RowParallelLinear in an MLP.
            # Usually, it receives sharded input from ColumnParallelLinear -> GELU.
            # If we were to split it:
            # input_shards = _split_along_last_dim(input_, self.tensor_parallel_size)
            # rank = dist.get_rank(group=self.tp_group)
            # input_parallel = input_shards[rank]
            # For now, assume input_is_parallel is True if tp_size > 1
            input_parallel = input_


        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # output_parallel is partial sum. All-reduce to get the full sum.
        if self.tensor_parallel_size > 1:
            output = _ReduceFromParallelRegion.apply(output_parallel, self.tp_group)
        else:
            output = output_parallel

        if self.bias is not None:
            output = output + self.bias
        return output

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config # Store config
        self.use_tp = config.tensor_parallel_size > 1 if hasattr(config, 'tensor_parallel_size') else False
        self.tensor_parallel_size = config.tensor_parallel_size if self.use_tp else 1
        self.tp_group = config.tp_group if self.use_tp else None

        if self.use_tp:
            _ensure_divisibility(config.n_embd, self.tensor_parallel_size)
            # Heads are split among TP ranks
            _ensure_divisibility(config.n_head, self.tensor_parallel_size)
            self.num_heads_per_partition = config.n_head // self.tensor_parallel_size
            self.hidden_size_per_partition = config.n_embd // self.tensor_parallel_size
            
            # Key, query, value projections for local heads
            self.c_attn = ColumnParallelLinear(
                config.n_embd, 
                3 * config.n_embd, # Still project to 3 * full_embd, sharding is on output_features
                bias=config.bias, 
                config=config
            )
            # Output projection
            self.c_proj = RowParallelLinear(
                config.n_embd, # Input is full_embd (after gathering from heads if TP)
                               # Or if sharded, this needs to be input_features_per_partition
                               # Let's assume input to c_proj is full n_embd after head concat
                config.n_embd, 
                bias=config.bias, 
                config=config,
                input_is_parallel=False # Output of attention heads will be gathered before c_proj
            )
        else:
            # key, query, value projections for all heads, but in a batch
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            # output projection
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.n_head = config.n_head # Total number of heads
        self.n_embd = config.n_embd # Total embedding dimensionality

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout # for flash attention
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # For TP, the mask might need to be handled per TP rank if sequence length is also sharded (not doing that here)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values
        if self.use_tp:
            # c_attn is ColumnParallelLinear, output is sharded: (B, T, 3 * n_embd / tp_size)
            qkv_parallel = self.c_attn(x)
            # Split qkv_parallel into q_parallel, k_parallel, v_parallel
            # Each is (B, T, n_embd / tp_size)
            # This requires careful splitting based on the sharded dimension
            # The output of ColumnParallelLinear for c_attn is (B, T, 3 * hidden_size_per_partition)
            # So we split into three (B, T, hidden_size_per_partition) tensors for Q, K, V for the local heads
            q_local, k_local, v_local = torch.split(qkv_parallel, self.hidden_size_per_partition, dim=2)
            
            # Reshape for multi-head attention for local heads
            # (B, T, num_heads_per_partition, head_dim) -> (B, num_heads_per_partition, T, head_dim)
            # where head_dim = hidden_size_per_partition / num_heads_per_partition (NO, head_dim = C // total_n_head)
            head_dim = self.n_embd // self.n_head # head_dim is based on total n_embd and total n_head
            
            q_local = q_local.view(B, T, self.num_heads_per_partition, head_dim).transpose(1, 2)
            k_local = k_local.view(B, T, self.num_heads_per_partition, head_dim).transpose(1, 2)
            v_local = v_local.view(B, T, self.num_heads_per_partition, head_dim).transpose(1, 2)
            
            # Local attention computation
            if self.flash:
                y_local = torch.nn.functional.scaled_dot_product_attention(q_local, k_local, v_local, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att_local = (q_local @ k_local.transpose(-2, -1)) * (1.0 / math.sqrt(k_local.size(-1)))
                att_local = att_local.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att_local = F.softmax(att_local, dim=-1)
                att_local = self.attn_dropout(att_local)
                y_local = att_local @ v_local
            
            # y_local is (B, num_heads_per_partition, T, head_dim)
            # Transpose and reshape back: (B, T, num_heads_per_partition * head_dim) = (B, T, hidden_size_per_partition)
            y_local_reshaped = y_local.transpose(1, 2).contiguous().view(B, T, self.hidden_size_per_partition)
            
            # Gather attention outputs from all TP ranks
            # Each rank has y_local_reshaped of size (B, T, hidden_size_per_partition)
            # We need to gather these along the last dimension to get (B, T, n_embd)
            # Create a tensor list for all_gather
            tensor_list = [torch.empty_like(y_local_reshaped) for _ in range(self.tensor_parallel_size)]
            dist.all_gather(tensor_list, y_local_reshaped.contiguous(), group=self.tp_group)
            y_gathered = torch.cat(tensor_list, dim=-1) # Concatenate along the feature dimension

            # Output projection (RowParallelLinear, expects full input unless input_is_parallel=True for it)
            # Since we gathered, input to c_proj is full (B, T, C)
            # c_proj (RowParallelLinear) will internally handle sharding its input if input_is_parallel=True was set for it.
            # But we set input_is_parallel=False for c_proj, so it expects full input and all_reduces its output.
            y = self.c_proj(y_gathered)

        else: # Original non-TP path
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)

        y = self.resid_dropout(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_tp = config.tensor_parallel_size > 1 if hasattr(config, 'tensor_parallel_size') else False
        if self.use_tp:
            self.c_fc = ColumnParallelLinear(config.n_embd, 4 * config.n_embd, bias=config.bias, config=config)
            self.gelu = nn.GELU() # GELU is element-wise, works on sharded input
            self.c_proj = RowParallelLinear(4 * config.n_embd, config.n_embd, bias=config.bias, config=config, input_is_parallel=True)
        else:
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu    = nn.GELU()
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # --- New fields for Tensor Parallelism ---
    tensor_parallel_size: int = 1 # Number of GPUs for tensor parallelism
    tp_group = None # PyTorch distributed group for tensor parallelism
    # Add dtype to config for TP layers to access it consistently
    dtype: torch.dtype = torch.float32 # Default, will be updated from training script

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # print(f"Initializing nn.Linear: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            # Initialize sharded weights. For simplicity, initialize them as if they are smaller distinct layers.
            # A more sophisticated approach would involve initializing the full weight on one rank
            # and then scattering/sharding, or using a carefully seeded initialization across ranks.
            # print(f"Initializing TP Linear: {type(module).__name__} with weight shape {module.weight.shape}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Init sharded weight
            if module.bias is not None:
                # For ColumnParallelLinear, bias is sharded. For RowParallelLinear, bias is full (added after all_reduce).
                # The current bias initialization (zeros) is fine for both cases as sharded zeros are still zeros,
                # and full bias initialized to zeros is also fine.
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # print(f"Initializing nn.Embedding: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # This is the complete 'forward' method to replace the existing one in nanoGPT/model.py
# Ensure 'import torch.utils.checkpoint' is at the top of model.py

    def forward(self, idx, targets=None, use_recompute=False): # Added use_recompute flag
        device = idx.device
        b, t = idx.size()

        # Check if sequence length t > block_size and crop if necessary
        # This logic is to ensure compatibility with how nanoGPT handles block_size.
        if t > self.config.block_size:
            idx = idx[:, :self.config.block_size]
            if targets is not None: # Also crop targets if they exist
                targets = targets[:, :self.config.block_size]
            t = self.config.block_size # Update t to the new cropped length

        # The original assert t <= self.config.block_size can be kept or removed
        # if the cropping logic above handles it. For robustness, it's fine.
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        # Get current dropout probability from the nn.Dropout layer
        current_dropout_p = 0.0
        if hasattr(self.transformer, 'drop') and isinstance(self.transformer.drop, torch.nn.Dropout):
            current_dropout_p = self.transformer.drop.p

        # Apply dropout only if p > 0.0
        if current_dropout_p > 0.0:
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = tok_emb + pos_emb

        for block_idx, block in enumerate(self.transformer.h): # self.transformer.h is the ModuleList of Blocks
            if use_recompute and self.training:
                # Ensure the 'block' itself is a callable module that takes 'x'
                # and returns the transformed 'x'.
                # use_reentrant=False is generally recommended for newer PyTorch versions
                # if your model allows for it (no in-place ops that checkpointing might break).
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x) # Final LayerNorm

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # Output logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def get_pp_stages(self, num_stages: int) -> list[nn.Module]:
        """
        Splits the model into a specified number of pipeline stages.
        Returns a list of nn.Sequential modules, each representing a stage.
        """
        if num_stages <= 0:
            raise ValueError("Number of pipeline stages must be positive.")

        # Ensure n_layer is divisible by num_stages if num_stages > 1 and n_layer > 0 for even distribution
        # Or handle uneven distribution. For simplicity, let's try to distribute as evenly as possible.
        # if self.config.n_layer > 0 and num_stages > 1 and self.config.n_layer % num_stages != 0:
        #     print(f"Warning: n_layer ({self.config.n_layer}) is not perfectly divisible by num_stages ({num_stages})."
        #           f" Block distribution might be slightly uneven.")

        stages = []
        all_blocks = list(self.transformer.h) # Convert ModuleList to list
        n_layers_total = len(all_blocks)
        
        layers_per_stage = [n_layers_total // num_stages + (1 if i < n_layers_total % num_stages else 0) for i in range(num_stages)]
        
        current_block_idx = 0
        for i in range(num_stages):
            stage_layers = []
            # First stage gets embeddings and dropout
            if i == 0:
                stage_layers.extend([
                    self.transformer.wte,
                    self.transformer.wpe,
                    self.transformer.drop
                ])
            
            # Add assigned blocks to this stage
            num_blocks_for_stage = layers_per_stage[i]
            stage_layers.extend(all_blocks[current_block_idx : current_block_idx + num_blocks_for_stage])
            current_block_idx += num_blocks_for_stage
            
            # Last stage gets final LayerNorm and LM head
            if i == num_stages - 1:
                stage_layers.extend([
                    self.transformer.ln_f,
                    self.lm_head
                ])
            
            stages.append(nn.Sequential(*stage_layers))
            
        # Sanity check: ensure all blocks are assigned
        if current_block_idx != n_layers_total:
             raise RuntimeError(f"Error in model splitting: Not all transformer blocks were assigned to stages. Assigned: {current_block_idx}, Total: {n_layers_total}")

        # Assign devices to stages will happen in the training script
        return stages

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
