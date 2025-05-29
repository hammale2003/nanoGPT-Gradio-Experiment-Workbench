# train_core.py
import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import subprocess # Make sure this is imported
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

# --- Path Setup (Assuming this_studio contains train_core.py and the nanoGPT folder) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# The 'nanoGPT' directory is expected to be a subdirectory within current_script_dir
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir) # Add current_script_dir to find nanoGPT package

try:
    from nanoGPT.model import GPTConfig, GPT
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from 'nanoGPT.model'.")
    print(f"Ensure 'train_core.py' is in a directory (e.g., 'THIS_STUDIO'),")
    print(f"and a folder named 'nanoGPT' (the cloned repository) is also directly inside that same directory.")
    print(f"Directory of train_core.py: {current_script_dir}")
    print(f"Expected 'nanoGPT' folder at: {os.path.join(current_script_dir, 'nanoGPT')}")
    print(f"sys.path: {sys.path}")
    print(f"Original ImportError: {e}")
    raise

# --- get_lr function ---
def get_lr(it, warmup_iters_local, learning_rate_local, lr_decay_iters_local, min_lr_local):
    if it < warmup_iters_local: return learning_rate_local * it / warmup_iters_local
    if it > lr_decay_iters_local: return min_lr_local
    decay_ratio = (it - warmup_iters_local) / (lr_decay_iters_local - warmup_iters_local)
    assert 0 <= decay_ratio <= 1; coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr_local + coeff * (learning_rate_local - min_lr_local)

# --- Default Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' if torch.cuda.is_available() else 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# --- Data Loading ---
def get_batch(split, block_size_local, batch_size_local, train_data_local, val_data_local, device_local):
    data = train_data_local if split == 'train' else val_data_local
    if batch_size_local <= 0: raise ValueError("batch_size_local must be positive.")
    current_data_len = len(data)
    if current_data_len <= block_size_local:
        if current_data_len == 0: raise ValueError(f"Data for '{split}' split is empty.")
        # print(f"Warning: Data length ({current_data_len}) for '{split}' split is <= block_size ({block_size_local}). Adjusting block_size for this batch to {max(1, current_data_len - 1)}.")
        block_size_local = max(1, current_data_len - 1)
        if block_size_local == 0 : raise ValueError(f"Cannot form a sequence for '{split}' split as adjusted block_size is 0 (data length: {current_data_len}).")
    ix = torch.randint(current_data_len - block_size_local, (batch_size_local,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size_local]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size_local]).astype(np.int64)) for i in ix])
    if device_local == 'cuda': x, y = x.pin_memory().to(device_local, non_blocking=True), y.pin_memory().to(device_local, non_blocking=True)
    else: x, y = x.to(device_local), y.to(device_local)
    return x, y

# --- Loss Estimation ---
@torch.no_grad()
def estimate_loss(model_local, block_size_local, micro_batch_size_eval, train_data_local, val_data_local, device_local, eval_iters_local, ctx_local, use_recompute_during_eval=False):
    out = {}; model_local.eval()
    for split in ['train', 'val']:
        current_data_split = val_data_local if split == 'val' else train_data_local
        if len(current_data_split) == 0: out[split] = float('nan'); continue
        losses = torch.zeros(eval_iters_local)
        for k in range(eval_iters_local):
            try: X, Y = get_batch(split, block_size_local, micro_batch_size_eval, train_data_local, val_data_local, device_local)
            except ValueError as e: losses[k] = float('nan'); continue # Should ideally log this error or handle more gracefully
            
            with ctx_local:
                _, loss_output = model_local(X, Y, use_recompute=use_recompute_during_eval)
            
            # Ensure scalar loss by averaging if it came from DataParallel
            # .mean() is a no-op if loss_output is already a scalar.
            scalar_loss = loss_output.mean()
            losses[k] = scalar_loss.item()

        valid_losses = losses[~torch.isnan(losses)]; out[split] = valid_losses.mean().item() if len(valid_losses) > 0 else float('nan')
    model_local.train(); return out

# --- Main Training Function ---
def run_nanoGPT_training(
    block_size: int, vocab_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float, bias: bool,
    batch_size_effective: int, micro_batch_size_ui: int, learning_rate: float, max_iters: int,
    weight_decay: float, beta1: float, beta2: float, grad_clip: float, decay_lr: bool,
    warmup_iters: int, lr_decay_iters: int, min_lr: float,
    grad_accumulation_steps_ui: int, use_recompute_ui: bool, use_data_parallel_ui: bool,
    use_pipeline_parallel_ui: bool, use_tensor_parallel_ui: bool,
    dataset_name: str, out_dir_ui: str, log_interval_ui: int, eval_interval_ui: int, eval_iters_ui: int
):
    tokens_per_iter = grad_accumulation_steps_ui * micro_batch_size_ui * block_size
    print(f"--- Training Run Start ---")
    print(f"Target Effective Batch Size (for info): {batch_size_effective}")
    print(f"Micro-Batch Size (per fwd pass): {micro_batch_size_ui}")
    print(f"Gradient Accumulation Steps (direct from UI): {grad_accumulation_steps_ui}")
    print(f"Tokens per effective iter: {tokens_per_iter:,}")

    path_to_nanogpt_repo_root = os.path.join(current_script_dir, "nanoGPT")
    data_dir_for_dataset = os.path.join(path_to_nanogpt_repo_root, "data", dataset_name)
    train_bin_path = os.path.join(data_dir_for_dataset, 'train.bin')
    val_bin_path = os.path.join(data_dir_for_dataset, 'val.bin')

    if not (os.path.exists(train_bin_path) and os.path.exists(val_bin_path)):
        yield {"type": "info", "message": f"Data files for '{dataset_name}' not found. Attempting automatic preparation..."}
        prepare_script_path = os.path.join(data_dir_for_dataset, "prepare.py")
        if os.path.exists(prepare_script_path):
            if dataset_name not in ["shakespeare_char", "shakespeare"]:
                warning_msg = f"Automatic preparation for '{dataset_name}' might take a very long time. Consider running it manually."
                yield {"type": "info", "message": warning_msg}
            python_executable = sys.executable
            try:
                process = subprocess.run([python_executable, 'prepare.py'], cwd=data_dir_for_dataset, check=True, capture_output=True, text=True, timeout=3600)
                yield {"type": "info", "message": f"Preparation for '{dataset_name}' completed."}
            except subprocess.CalledProcessError as e:
                error_msg = f"Error during automatic preparation of '{dataset_name}': {e.stderr}"
                yield {"type": "error", "message": error_msg}; return
            except FileNotFoundError:
                error_msg = f"Error: Python interpreter '{python_executable}' or script '{prepare_script_path}' not found."
                yield {"type": "error", "message": error_msg}; return
            except subprocess.TimeoutExpired:
                error_msg = f"Timeout: Data preparation for '{dataset_name}' took too long and was stopped."
                yield {"type": "error", "message": error_msg}; return
        else:
            error_msg = f"prepare.py script not found in {data_dir_for_dataset} for dataset '{dataset_name}'. Cannot prepare automatically."
            yield {"type": "error", "message": error_msg}; return
        if not (os.path.exists(train_bin_path) and os.path.exists(val_bin_path)):
            error_msg = f"Automatic preparation for '{dataset_name}' failed. train.bin or val.bin still not found."
            yield {"type": "error", "message": error_msg}; return

    script_execution_dir = os.getcwd()
    absolute_out_dir_ui = os.path.join(script_execution_dir, out_dir_ui)
    os.makedirs(absolute_out_dir_ui, exist_ok=True)
    print(f"Checkpoints and logs will be saved to: {absolute_out_dir_ui}")
    torch.manual_seed(1337)

    try:
        train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
        val_data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
    except FileNotFoundError:
        error_msg = f"FATAL: Data files {train_bin_path} or {val_bin_path} not found even after preparation attempt."
        yield {"type": "error", "message": error_msg}; return

    meta_path = os.path.join(data_dir_for_dataset, 'meta.pkl')
    actual_vocab_size = vocab_size
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f: meta = pickle.load(f)
        actual_vocab_size = meta['vocab_size']
        print(f"Loaded vocab_size from meta.pkl: {actual_vocab_size}")
    else:
        print(f"Warning: meta.pkl not found in {data_dir_for_dataset}. Using UI-provided vocab_size: {vocab_size}.")
    
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=actual_vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args); model = GPT(gptconf)
    num_gpus = torch.cuda.device_count()

    if use_tensor_parallel_ui: yield {"type": "info", "message": "Tensor Parallelism flag ON (placeholder)."}
    if use_pipeline_parallel_ui: yield {"type": "info", "message": "Pipeline Parallelism flag ON (placeholder)."}
    
    model_for_optimizer = model
    if use_data_parallel_ui and not use_pipeline_parallel_ui and device == 'cuda' and num_gpus > 1 :
        model = DataParallel(model); model_for_optimizer = model.module
        print(f"Using DataParallel across {num_gpus} GPUs.")
    model.to(device)
    
    optimizer = model_for_optimizer.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    
    iter_num = 0; best_val_loss = 1e9
    overall_start_time = time.time(); t0 = time.time()
    max_mem_gb = 0.0 
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()

    for iter_num_current_loop in range(max_iters + 1):
        iter_num = iter_num_current_loop
        lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        current_max_mem_gb_eval = 0.0
        if device == 'cuda': current_max_mem_gb_eval = torch.cuda.max_memory_allocated() / (1024 ** 3); max_mem_gb = max(max_mem_gb, current_max_mem_gb_eval)
        
        if iter_num > 0 and iter_num % eval_interval_ui == 0:
            losses = estimate_loss(model, block_size, micro_batch_size_ui, train_data, val_data, device, eval_iters_ui, ctx)
            val_loss_current = losses.get('val', float('nan')); train_loss_est_current = losses.get('train', float('nan'))
            yield {"type": "eval", "iter": iter_num, "val_loss": val_loss_current,
                   "train_loss_est": train_loss_est_current, "best_val_loss": best_val_loss, "lr": lr,
                   "time_elapsed_total": time.time() - overall_start_time,
                   "gpu_mem_gb_max": max_mem_gb, "gpu_mem_gb_current_eval": current_max_mem_gb_eval}
            if not np.isnan(val_loss_current) and val_loss_current < best_val_loss:
                best_val_loss = val_loss_current; raw_model_state_dict = model_for_optimizer.state_dict()
                checkpoint = {'model': raw_model_state_dict, 'optimizer': optimizer.state_dict(), 'model_args': model_args,
                              'iter_num': iter_num, 'best_val_loss': best_val_loss, 'config': gptconf}
                ckpt_path = os.path.join(absolute_out_dir_ui, 'ckpt.pt')
                print(f"Iter {iter_num}: Saving checkpoint to {ckpt_path}"); torch.save(checkpoint, ckpt_path)
        
        if iter_num == max_iters:
             if max_iters % eval_interval_ui != 0 and max_iters > 0: # Ensure final eval if not already done
                losses = estimate_loss(model, block_size, micro_batch_size_ui, train_data, val_data, device, eval_iters_ui, ctx)
                if device == 'cuda': current_max_mem_gb_eval = torch.cuda.max_memory_allocated() / (1024 ** 3)
                yield {"type": "eval", "iter": iter_num, "val_loss": losses.get('val', float('nan')), "train_loss_est": losses.get('train', float('nan')),
                       "best_val_loss": best_val_loss, "lr": lr, "time_elapsed_total": time.time() - overall_start_time, "gpu_mem_gb_max": current_max_mem_gb_eval} # use current_max_mem_gb_eval here
             break

        optimizer.zero_grad(set_to_none=True); accumulated_loss_for_iter = 0.0
        for micro_step in range(grad_accumulation_steps_ui):
            try:
                X, Y = get_batch('train', block_size, micro_batch_size_ui, train_data, val_data, device)
            except ValueError as e:
                yield {"type": "error", "message": f"Training halted due to data error: {e}"}; return
            
            with ctx:
                logits, loss_from_model = model(X, Y, use_recompute=use_recompute_ui) # loss_from_model can be a tensor from DataParallel
            
            # Ensure loss_from_model is a scalar by averaging if it came from DataParallel.
            # .mean() is a no-op if loss_from_model is already a scalar (e.g., single GPU/CPU).
            actual_loss_this_micro_batch = loss_from_model.mean()
            
            # Now, scale this scalar loss for gradient accumulation
            loss_for_backward_and_accumulation = actual_loss_this_micro_batch / grad_accumulation_steps_ui
            
            loss_for_backward_and_accumulation.backward()
            accumulated_loss_for_iter += loss_for_backward_and_accumulation.item() # Summing the scaled losses

        if grad_clip != 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        dt = time.time() - t0; t0 = time.time()
        
        current_max_mem_gb_iter = 0.0
        if device == 'cuda': current_max_mem_gb_iter = torch.cuda.max_memory_allocated() / (1024 ** 3); max_mem_gb = max(max_mem_gb, current_max_mem_gb_iter)
        
        if iter_num % log_interval_ui == 0:
            yield {"type": "train_iter", "iter": iter_num, "loss": accumulated_loss_for_iter, "lr": lr, # accumulated_loss_for_iter is sum of scaled losses
                   "time_per_iter_ms": dt * 1000, "time_elapsed_total": time.time() - overall_start_time,
                   "gpu_mem_gb_max_iter": max_mem_gb, "gpu_mem_gb_current_iter": current_max_mem_gb_iter}
    
    final_elapsed_time = time.time() - overall_start_time
    final_max_mem = max_mem_gb if device == 'cuda' else 0.0
    yield {"type": "finished", "message": f"Training finished!", "best_val_loss": best_val_loss,
           "time_elapsed_total": final_elapsed_time, "gpu_mem_gb_max_overall": final_max_mem}
