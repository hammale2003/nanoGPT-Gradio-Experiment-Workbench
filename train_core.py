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
    # Try to import the modified model with all parallelism support
    from model_with_checkpointing import GPTConfig, GPT, GPTWithSingleProcessPipeline
    print("✅ Using advanced GPT model with REAL implementations of:")
    print("   - Gradient Checkpointing (Activation Recomputation)")
    print("   - Pipeline Parallelism") 
    print("   - Tensor Parallelism")
except ImportError:
    try:
        from nanoGPT.model import GPTConfig, GPT
        GPTWithSingleProcessPipeline = None
        print("⚠️  Warning: Using original nanoGPT model without advanced parallelism support!")
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

# --- Helper function to collect GPU memory per GPU for all parallelism types ---
def get_gpu_memory_per_gpu():
    """Collect GPU memory usage for each GPU when using any parallelism type"""
    if not torch.cuda.is_available():
        return {}
    
    gpu_memory = {}
    current_device = torch.cuda.current_device()
    
    for gpu_id in range(torch.cuda.device_count()):
        try:
            # Set device context and collect memory
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize(gpu_id)  # Make sure all operations are finished
            
            # Get both allocated and reserved memory
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)  # Current allocation in GB
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)  # Reserved memory in GB
            
            # Use the higher of allocated or reserved memory for better tracking
            gpu_memory[gpu_id] = max(memory_allocated, memory_reserved)
            
            # For pipeline parallelism, we want to track even small amounts of memory usage
            # since different GPUs might have different amounts of the model
            
        except Exception as e:
            # If we can't access a GPU, record 0 memory usage
            gpu_memory[gpu_id] = 0.0
    
    # Restore original device
    try:
        torch.cuda.set_device(current_device)
    except:
        pass  # Ignore errors when restoring device
    
    return gpu_memory

# --- Helper function to initialize distributed for tensor parallelism ---
def initialize_distributed_for_tensor_parallelism(num_gpus):
    """
    Initialize PyTorch distributed for tensor parallelism if not already initialized.
    This is needed for tensor parallelism to work with multi-GPU communication.
    """
    try:
        # Check if distributed is already initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print("✅ PyTorch distributed already initialized")
            return True
        
        if not torch.distributed.is_available():
            print("⚠️  PyTorch distributed not available in this installation")
            print("   Install PyTorch with NCCL support for tensor parallelism")
            return False
        
        if num_gpus < 2:
            print("⚠️  Need at least 2 GPUs for tensor parallelism")
            return False
        
        # Initialize distributed with NCCL backend for GPU communication
        print("🔄 Initializing PyTorch distributed for tensor parallelism...")
        
        # Set environment variables for single-machine multi-GPU
        import os
        
        # Use a unique port to avoid conflicts
        import random
        master_port = str(12355 + random.randint(0, 1000))
        
        # Set required environment variables
        env_vars = {
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': master_port,
            'WORLD_SIZE': '1',
            'RANK': '0',
            'LOCAL_RANK': '0',
            'NCCL_DEBUG': 'WARN'  # For debugging NCCL issues
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        print(f"   Using MASTER_PORT: {master_port}")
        
        # Initialize the process group with timeout
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0,
            timeout=torch.distributed.default_pg_timeout  # Use default timeout
        )
        
        # Verify initialization worked
        if torch.distributed.is_initialized():
            print("✅ PyTorch distributed initialized successfully for tensor parallelism")
            
            # Test basic communication
            test_tensor = torch.zeros(1).cuda()
            torch.distributed.all_reduce(test_tensor)
            print("✅ Distributed communication test successful")
            
            return True
        else:
            print("❌ Distributed initialization appeared to succeed but is not actually initialized")
            return False
        
    except Exception as e:
        print(f"❌ Failed to initialize PyTorch distributed: {e}")
        
        # Provide helpful error messages for common issues
        error_str = str(e).lower()
        if 'nccl' in error_str:
            print("   💡 NCCL error - this might be a driver/CUDA compatibility issue")
            print("   Try: export NCCL_DEBUG=INFO for more details")
        elif 'timeout' in error_str:
            print("   💡 Timeout error - GPUs might not be able to communicate")
        elif 'address already in use' in error_str:
            print("   💡 Port conflict - will try different port on next attempt")
        
        print("   Tensor parallelism will fall back to single-process mode")
        return False

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
def estimate_loss(model_local, block_size_local, micro_batch_size_eval, train_data_local, val_data_local, device_local, eval_iters_local, ctx_local, use_recompute_local=False):
    out = {}; model_local.eval()
    for split in ['train', 'val']:
        current_data_split = val_data_local if split == 'val' else train_data_local
        if len(current_data_split) == 0: out[split] = float('nan'); continue
        losses = torch.zeros(eval_iters_local)
        for k in range(eval_iters_local):
            try: X, Y = get_batch(split, block_size_local, micro_batch_size_eval, train_data_local, val_data_local, device_local)
            except ValueError as e: losses[k] = float('nan'); continue # Should ideally log this error or handle more gracefully
            
            with ctx_local:
                # Pass use_recompute=False during evaluation for faster inference
                result = model_local(X, Y, use_recompute=False)
                
                # Handle the case where model returns None or invalid result
                if result is None or (isinstance(result, tuple) and len(result) < 2):
                    losses[k] = float('nan')
                    continue
                
                if isinstance(result, tuple):
                    _, loss_output = result
                else:
                    # If model only returns loss (some configurations)
                    loss_output = result
                
                # Check if loss_output is valid
                if loss_output is None:
                    losses[k] = float('nan')
                    continue
            
            # Ensure scalar loss by averaging if it came from DataParallel
            # .mean() is a no-op if loss_output is already a scalar.
            scalar_loss = loss_output.mean() if hasattr(loss_output, 'mean') else loss_output
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

    # Advanced Parallelism Configuration
    num_gpus = torch.cuda.device_count()
    
    # Check if distributed is available and initialized for advanced parallelism
    dist_available = False
    try:
        dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
    except:
        dist_available = False
    
    # If tensor parallelism is requested and distributed is not initialized, try to initialize it
    if use_tensor_parallel_ui and num_gpus > 1 and not dist_available:
        print("🔄 Tensor Parallelism requested - attempting to initialize PyTorch distributed...")
        dist_available = initialize_distributed_for_tensor_parallelism(num_gpus)
    
    # Configure parallelism - allow pipeline parallelism even without distributed
    pipeline_parallel_size = min(num_gpus, 2) if use_pipeline_parallel_ui and num_gpus > 1 else 1
    
    # Tensor parallelism configuration - only enable if distributed is truly available
    if dist_available and use_tensor_parallel_ui and num_gpus > 1:
        try:
            # Double-check that distributed is working properly
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                tensor_parallel_size = min(num_gpus, 2)
                print("✅ Tensor Parallelism enabled with distributed training")
            else:
                tensor_parallel_size = 1
                print("⚠️  Tensor Parallelism disabled: distributed not properly initialized")
        except Exception as e:
            tensor_parallel_size = 1
            print(f"⚠️  Tensor Parallelism disabled due to distributed error: {e}")
    else:
        tensor_parallel_size = 1
        if use_tensor_parallel_ui and num_gpus > 1:
            print("⚠️  Tensor Parallelism requested but distributed initialization failed")
            print("   Falling back to single-process mode (no tensor parallelism)")
    
    # Ensure we don't exceed available GPUs
    total_parallel_gpus = pipeline_parallel_size * tensor_parallel_size
    if total_parallel_gpus > num_gpus:
        print(f"⚠️  Warning: Total parallelism ({total_parallel_gpus}) exceeds available GPUs ({num_gpus})")
        print(f"   Adjusting to use maximum available GPUs...")
        if use_pipeline_parallel_ui:
            pipeline_parallel_size = min(num_gpus, 2)
            tensor_parallel_size = 1
        elif use_tensor_parallel_ui:
            tensor_parallel_size = min(num_gpus, 2)
            pipeline_parallel_size = 1

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
    
    # Create model configuration with advanced parallelism support
    model_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
        bias=bias, vocab_size=actual_vocab_size, dropout=dropout,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size
    )
    
    gptconf = GPTConfig(**model_args)
    
    # Initialize model with parallelism configuration
    try:
        # Choose the appropriate model based on parallelism configuration
        if pipeline_parallel_size > 1 and not dist_available and GPTWithSingleProcessPipeline is not None:
            # Use single-process pipeline parallelism
            print("🔄 Using GPTWithSingleProcessPipeline for pipeline parallelism")
            tensor_parallel_rank = 0
            model = GPTWithSingleProcessPipeline(gptconf, tensor_parallel_rank=tensor_parallel_rank)
        elif (tensor_parallel_size > 1 or pipeline_parallel_size > 1) and dist_available:
            # Use advanced model with distributed parallelism
            print("🔄 Using GPTWithAdvancedParallelism for distributed tensor/pipeline parallelism")
            pipeline_rank = 0  # For now, we'll use rank 0 (single-process training)
            tensor_parallel_rank = 0
            from model_with_checkpointing import GPTWithAdvancedParallelism
            model = GPTWithAdvancedParallelism(gptconf, pipeline_rank=pipeline_rank, tensor_parallel_rank=tensor_parallel_rank)
        else:
            # Use standard model for single-GPU or when advanced features are not available
            print("🔄 Using standard GPT model")
            pipeline_rank = 0
            tensor_parallel_rank = 0
            model = GPT(gptconf, pipeline_rank=pipeline_rank, tensor_parallel_rank=tensor_parallel_rank)
    except TypeError as e:
        print(f"⚠️  Model initialization error: {e}")
        # Fallback to original model if advanced features not available
        try:
            model = GPT(gptconf)
            print("✅ Using standard model initialization (no parallelism features)")
        except Exception as e2:
            # Last resort: try importing the original nanoGPT model
            print(f"❌ Standard model failed too: {e2}")
            try:
                from nanoGPT.model import GPT as OriginalGPT
                model = OriginalGPT(gptconf)
                print("✅ Using original nanoGPT model (emergency fallback)")
            except Exception as e3:
                yield {"type": "error", "message": f"All model initialization attempts failed: {e3}"}
                return
    except Exception as e:
        print(f"❌ Unexpected model initialization error: {e}")
        yield {"type": "error", "message": f"Model initialization failed: {e}"}
        return

    # Parallelism status messages
    if use_tensor_parallel_ui: 
        if tensor_parallel_size > 1 and dist_available:
            yield {"type": "info", "message": f"✅ Tensor Parallelism ENABLED - Using {tensor_parallel_size} GPUs for tensor operations"}
            yield {"type": "info", "message": f"🔗 PyTorch distributed initialized for multi-GPU communication"}
        else:
            # Tensor parallelism was requested but couldn't be enabled
            reason = "unknown reason"
            if num_gpus <= 1:
                reason = "only 1 GPU available"
            elif not dist_available:
                reason = "PyTorch distributed initialization failed"
            elif tensor_parallel_size <= 1:
                reason = "distributed not properly initialized"
            
            yield {"type": "info", "message": f"⚠️  Tensor Parallelism requested but disabled due to: {reason}"}
            yield {"type": "info", "message": "   Using standard training instead. Model will use regular (non-tensor-parallel) components."}
            yield {"type": "info", "message": "   💡 For tensor parallelism: ensure multiple GPUs and proper NCCL/CUDA setup"}
    
    if use_pipeline_parallel_ui: 
        if pipeline_parallel_size > 1:
            mode = "distributed" if dist_available else "single-process"
            yield {"type": "info", "message": f"✅ Pipeline Parallelism ENABLED - Using {pipeline_parallel_size} GPUs in {mode} mode"}
            
            # Report detailed pipeline device information
            if hasattr(model, 'get_pipeline_device_info'):
                device_info = model.get_pipeline_device_info()
                yield {"type": "info", "message": f"📋 Pipeline device placement: {device_info}"}
            elif hasattr(model, '_verify_device_placement'):
                is_multi_gpu = model._verify_device_placement()
                if is_multi_gpu:
                    yield {"type": "info", "message": f"✅ Pipeline parallelism confirmed - model distributed across {pipeline_parallel_size} GPUs"}
                else:
                    yield {"type": "info", "message": f"⚠️  Pipeline parallelism setup issue - model appears to be on single device"}
            
            # Check weight tying status for pipeline parallelism
            if hasattr(model, 'weights_tied'):
                tying_status = "enabled" if model.weights_tied else "disabled (normal for pipeline parallelism)"
                yield {"type": "info", "message": f"🔗 Weight tying {tying_status}"}
        else:
            yield {"type": "info", "message": "⚠️  Pipeline Parallelism requested but using single GPU (need more GPUs)"}
    
    if use_recompute_ui: 
        yield {"type": "info", "message": "✅ Gradient Checkpointing ENABLED - Memory usage will be reduced during training!"}
    
    # Traditional data parallelism (can be combined with other parallelism types)
    model_for_optimizer = model
    if use_data_parallel_ui and not use_pipeline_parallel_ui and device == 'cuda' and num_gpus > 1:
        model = DataParallel(model)
        model_for_optimizer = model.module
        print(f"Using DataParallel across {num_gpus} GPUs.")
        yield {"type": "info", "message": f"Data Parallelism enabled with {num_gpus} GPUs."}
    
    # General message about per-GPU tracking for all parallelism types
    if (use_data_parallel_ui or use_pipeline_parallel_ui or use_tensor_parallel_ui) and num_gpus > 1:
        yield {"type": "info", "message": f"📊 Per-GPU memory tracking enabled for {num_gpus} GPUs - Individual GPU lines will appear in plots!"}
    
    # Move model to device (for single-process pipeline, this is handled in the model initialization)
    if not (pipeline_parallel_size > 1 and not dist_available):
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
        gpu_mem_per_gpu_eval = {}
        if device == 'cuda': 
            current_max_mem_gb_eval = torch.cuda.max_memory_allocated() / (1024 ** 3); max_mem_gb = max(max_mem_gb, current_max_mem_gb_eval)
            # Collect per-GPU memory for ALL parallelism types (not just data parallelism)
            if (use_data_parallel_ui or use_pipeline_parallel_ui or use_tensor_parallel_ui) and num_gpus > 1:
                gpu_mem_per_gpu_eval = get_gpu_memory_per_gpu()
        
        if iter_num > 0 and iter_num % eval_interval_ui == 0:
            losses = estimate_loss(model, block_size, micro_batch_size_ui, train_data, val_data, device, eval_iters_ui, ctx, use_recompute_ui)
            val_loss_current = losses.get('val', float('nan')); train_loss_est_current = losses.get('train', float('nan'))
            
            eval_metrics = {
                "type": "eval", "iter": iter_num, "val_loss": val_loss_current,
                "train_loss_est": train_loss_est_current, "best_val_loss": best_val_loss, "lr": lr,
                "time_elapsed_total": time.time() - overall_start_time,
                "gpu_mem_gb_max": max_mem_gb, "gpu_mem_gb_current_eval": current_max_mem_gb_eval
            }
            
            # Add per-GPU memory data for ANY parallelism type
            if (use_data_parallel_ui or use_pipeline_parallel_ui or use_tensor_parallel_ui) and num_gpus > 1 and gpu_mem_per_gpu_eval:
                eval_metrics["gpu_mem_per_gpu_eval"] = gpu_mem_per_gpu_eval
                
            yield eval_metrics
            if not np.isnan(val_loss_current) and val_loss_current < best_val_loss:
                best_val_loss = val_loss_current; raw_model_state_dict = model_for_optimizer.state_dict()
                checkpoint = {'model': raw_model_state_dict, 'optimizer': optimizer.state_dict(), 'model_args': model_args,
                              'iter_num': iter_num, 'best_val_loss': best_val_loss, 'config': gptconf}
                ckpt_path = os.path.join(absolute_out_dir_ui, 'ckpt.pt')
                print(f"Iter {iter_num}: Saving checkpoint to {ckpt_path}"); torch.save(checkpoint, ckpt_path)
        
        if iter_num == max_iters:
             if max_iters % eval_interval_ui != 0 and max_iters > 0: # Ensure final eval if not already done
                losses = estimate_loss(model, block_size, micro_batch_size_ui, train_data, val_data, device, eval_iters_ui, ctx, use_recompute_ui)
                if device == 'cuda': current_max_mem_gb_eval = torch.cuda.max_memory_allocated() / (1024 ** 3)
                yield {"type": "eval", "iter": iter_num, "val_loss": losses.get('val', float('nan')), "train_loss_est": losses.get('train', float('nan')),
                       "best_val_loss": best_val_loss, "lr": lr, "time_elapsed_total": time.time() - overall_start_time, "gpu_mem_gb_max": current_max_mem_gb_eval}
             break

        optimizer.zero_grad(set_to_none=True); accumulated_loss_for_iter = 0.0
        for micro_step in range(grad_accumulation_steps_ui):
            try:
                X, Y = get_batch('train', block_size, micro_batch_size_ui, train_data, val_data, device)
            except ValueError as e:
                yield {"type": "error", "message": f"Training halted due to data error: {e}"}; return
            
            with ctx:
                # Use advanced forward pass with all parallelism types
                if hasattr(model, 'pipeline_forward') and (use_pipeline_parallel_ui and pipeline_parallel_size > 1):
                    result = model.pipeline_forward(X, Y, use_recompute=use_recompute_ui, 
                                                   micro_batch_size=micro_batch_size_ui)
                else:
                    result = model(X, Y, use_recompute=use_recompute_ui)
                
                # Handle the case where model returns None or invalid result
                if result is None or (isinstance(result, tuple) and len(result) < 2):
                    yield {"type": "error", "message": "Model returned None or invalid result. Check parallelism configuration."}
                    return
                
                if isinstance(result, tuple):
                    logits, loss_from_model = result
                else:
                    # If model only returns loss (some configurations)
                    logits, loss_from_model = None, result
            
            # Ensure loss_from_model is valid and convert to scalar
            if loss_from_model is None:
                yield {"type": "error", "message": "Model returned None loss. Check model configuration and parallelism settings."}
                return
            
            # Ensure loss_from_model is a scalar by averaging if it came from DataParallel.
            actual_loss_this_micro_batch = loss_from_model.mean() if hasattr(loss_from_model, 'mean') else loss_from_model
            
            # Scale loss for gradient accumulation
            loss_for_backward_and_accumulation = actual_loss_this_micro_batch / grad_accumulation_steps_ui
            
            loss_for_backward_and_accumulation.backward()
            accumulated_loss_for_iter += loss_for_backward_and_accumulation.item()

        if grad_clip != 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        dt = time.time() - t0; t0 = time.time()
        
        current_max_mem_gb_iter = 0.0
        gpu_mem_per_gpu_iter = {}
        if device == 'cuda': 
            current_max_mem_gb_iter = torch.cuda.max_memory_allocated() / (1024 ** 3); max_mem_gb = max(max_mem_gb, current_max_mem_gb_iter)
            # Collect per-GPU memory for ALL parallelism types (not just data parallelism)
            if (use_data_parallel_ui or use_pipeline_parallel_ui or use_tensor_parallel_ui) and num_gpus > 1:
                gpu_mem_per_gpu_iter = get_gpu_memory_per_gpu()
        
        if iter_num % log_interval_ui == 0:
            train_metrics = {
                "type": "train_iter", "iter": iter_num, "loss": accumulated_loss_for_iter, "lr": lr,
                "time_per_iter_ms": dt * 1000, "time_elapsed_total": time.time() - overall_start_time,
                "gpu_mem_gb_max_iter": max_mem_gb, "gpu_mem_gb_current_iter": current_max_mem_gb_iter
            }
            
            # Add per-GPU memory data for ANY parallelism type
            if (use_data_parallel_ui or use_pipeline_parallel_ui or use_tensor_parallel_ui) and num_gpus > 1 and gpu_mem_per_gpu_iter:
                train_metrics["gpu_mem_per_gpu"] = gpu_mem_per_gpu_iter
                
            yield train_metrics
    
    final_elapsed_time = time.time() - overall_start_time
    final_max_mem = max_mem_gb if device == 'cuda' else 0.0
    yield {"type": "finished", "message": f"Training finished!", "best_val_loss": best_val_loss,
           "time_elapsed_total": final_elapsed_time, "gpu_mem_gb_max_overall": final_max_mem}
