# gradio_app.py
import gradio as gr
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import sys # For path manipulation

# --- Path Setup for train_core and to help train_core find nanoGPT ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

try:
    from train_core import run_nanoGPT_training
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'run_nanoGPT_training' from 'train_core.py'.")
    print(f"Ensure 'train_core.py' is in the same directory as 'gradio_app.py': {current_script_dir}")
    print(f"Current sys.path: {sys.path}")
    print(f"Original ImportError: {e}")
    if 'nanoGPT' in str(e).lower() or 'model' in str(e).lower():
        print("\n---\nError might be from 'train_core.py' trying to import 'nanoGPT'. Ensure structure:\n"
              f"1. Current directory: {current_script_dir}\n"
              f"2. 'train_core.py' is here.\n"
              f"3. 'nanoGPT' directory (cloned repo) is also here: {os.path.join(current_script_dir, 'nanoGPT')}\n---\n")
    raise

# --- History and Visualization (Global state for the app session) ---
history = {
    "metrics_list": [],
    "config_list": []
}

def visualize_results_gradio(metrics_list_global, config_list_global):
    if not metrics_list_global:
        fig, axs = plt.subplots(4, 1, figsize=(12, 20))
        titles = ["Loss Curves vs. Iteration", "Time per Evaluation Interval", "Peak GPU Memory at Eval Points", "Final Validation Loss per Run"]
        for i in range(4):
            axs[i].text(0.5, 0.5, 'No data yet. Run training first.', ha='center', va='center', fontsize=9)
            axs[i].set_title(titles[i], fontsize=12)
        plt.tight_layout()
        return fig, "No data to visualize yet. Please complete a training run."
    
    fig, axs = plt.subplots(4, 1, figsize=(14, 20))
    num_runs = len(metrics_list_global)
    try: colors = plt.cm.get_cmap('tab10', max(10, num_runs))
    except ValueError: colors = plt.cm.get_cmap('viridis', max(10, num_runs))

    # Plot 1: Loss Curves
    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        eval_data = [m for m in run_metrics if m['type'] == 'eval' and 'iter' in m and 'val_loss' in m and 'train_loss_est' in m]
        iters = [m['iter'] for m in eval_data]
        val_l = [m['val_loss'] for m in eval_data]
        train_l_est = [m['train_loss_est'] for m in eval_data]
        cfg_lbl = (f"R{i+1} μB{run_config_summary.get('micro_batch_size_ui', '?')},"
                   f"GA{run_config_summary.get('grad_accumulation_steps_direct_ui', '?')},"
                   f"RC{run_config_summary.get('use_recompute_ui', False)},"
                   f"DP{run_config_summary.get('use_data_parallel_ui', False)}")
        if iters and val_l: axs[0].plot(iters, val_l, label=f"{cfg_lbl} Val", color=colors(i % colors.N), linestyle='-', marker='.', markersize=5, alpha=0.9)
        if iters and train_l_est: axs[0].plot(iters, train_l_est, label=f"{cfg_lbl} TrainEst", color=colors(i % colors.N), linestyle='--', marker='x', markersize=5, alpha=0.9)
    axs[0].set_title("Loss Curves vs. Iteration", fontsize=14); axs[0].set_xlabel("Iteration", fontsize=12); axs[0].set_ylabel("Loss", fontsize=12)
    # Only add legend if there's actual data plotted
    if metrics_list_global and any(len([m for m in run_metrics if m['type'] == 'eval' and 'iter' in m and 'val_loss' in m]) > 0 for run_metrics, _ in zip(metrics_list_global, config_list_global)):
        axs[0].legend(fontsize='small', loc='best', ncol=1)
    axs[0].grid(True, linestyle=':', alpha=0.7); axs[0].tick_params(axis='both', which='major', labelsize=10)

    # Plot 2: Time per Evaluation Interval
    for i, (run_metrics, _) in enumerate(zip(metrics_list_global, config_list_global)):
        eval_data = [m for m in run_metrics if m['type'] == 'eval' and 'iter' in m and 'time_elapsed_total' in m]
        iters = [m['iter'] for m in eval_data]; times = [m['time_elapsed_total'] for m in eval_data]
        interval_t, valid_iters_interval = [], []
        if times:
            interval_t.append(times[0]); valid_iters_interval.append(iters[0])
            for j in range(1, len(times)): interval_t.append(times[j] - times[j-1]); valid_iters_interval.append(iters[j])
        if valid_iters_interval and interval_t: axs[1].plot(valid_iters_interval, interval_t, label=f"Run {i+1}", color=colors(i % colors.N), marker='.', markersize=5, alpha=0.9)
    axs[1].set_title("Time per Evaluation Interval", fontsize=14); axs[1].set_xlabel("Iteration (end of interval)", fontsize=12); axs[1].set_ylabel("Time (s)", fontsize=12)
    # Only add legend if there's actual data plotted
    if metrics_list_global and any(len([m for m in run_metrics if m['type'] == 'eval' and 'iter' in m and 'time_elapsed_total' in m]) > 0 for run_metrics, _ in zip(metrics_list_global, config_list_global)):
        axs[1].legend(fontsize='small')
    axs[1].grid(True, linestyle=':', alpha=0.7); axs[1].tick_params(axis='both', which='major', labelsize=10)

    # Plot 3: Peak GPU Memory - Enhanced for ALL types of Parallelism
    has_gpu_data = False
    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        # Check if ANY type of parallelism is enabled
        use_data_parallel = run_config_summary.get('use_data_parallel_ui', False)
        use_pipeline_parallel = run_config_summary.get('use_pipeline_parallel_ui', False)
        use_tensor_parallel = run_config_summary.get('use_tensor_parallel_ui', False)
        use_any_parallelism = use_data_parallel or use_pipeline_parallel or use_tensor_parallel
        
        if use_any_parallelism:
            # For any parallelism type, show individual GPU lines + average
            gpu_data = {}  # gpu_id -> {iters: [], mems: []}
            iters_avg, gpu_mems_avg = [], []
            
            for m in run_metrics:
                if m['type'] == 'train_iter' and 'gpu_mem_per_gpu' in m:
                    iter_num = m['iter']
                    gpu_mem_data = m['gpu_mem_per_gpu']  # dict: {gpu_id: mem_gb}
                    
                    # Collect data for each GPU
                    for gpu_id, mem_gb in gpu_mem_data.items():
                        if gpu_id not in gpu_data:
                            gpu_data[gpu_id] = {'iters': [], 'mems': []}
                        gpu_data[gpu_id]['iters'].append(iter_num)
                        gpu_data[gpu_id]['mems'].append(mem_gb)
                    
                    # Calculate average for this iteration
                    if gpu_mem_data:
                        avg_mem = sum(gpu_mem_data.values()) / len(gpu_mem_data)
                        iters_avg.append(iter_num)
                        gpu_mems_avg.append(avg_mem)
                        has_gpu_data = True
                
                elif m['type'] == 'eval' and 'gpu_mem_per_gpu_eval' in m:
                    iter_num = m['iter']
                    gpu_mem_data = m['gpu_mem_per_gpu_eval']  # dict: {gpu_id: mem_gb}
                    
                    # Collect data for each GPU
                    for gpu_id, mem_gb in gpu_mem_data.items():
                        if gpu_id not in gpu_data:
                            gpu_data[gpu_id] = {'iters': [], 'mems': []}
                        if iter_num not in gpu_data[gpu_id]['iters']:
                            gpu_data[gpu_id]['iters'].append(iter_num)
                            gpu_data[gpu_id]['mems'].append(mem_gb)
                    
                    # Calculate average for this iteration
                    if gpu_mem_data and iter_num not in iters_avg:
                        avg_mem = sum(gpu_mem_data.values()) / len(gpu_mem_data)
                        iters_avg.append(iter_num)
                        gpu_mems_avg.append(avg_mem)
                        has_gpu_data = True
            
            # Plot individual GPU lines
            gpu_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
            for gpu_idx, (gpu_id, data) in enumerate(gpu_data.items()):
                meaningful_iters = [it for idx, it in enumerate(data['iters']) if data['mems'][idx] > 0.001]
                meaningful_mem = [mem for mem in data['mems'] if mem > 0.001]
                if meaningful_iters:
                    gpu_color = gpu_colors[gpu_idx % len(gpu_colors)]
                    # Create label indicating which parallelism type is used
                    parallelism_types = []
                    if use_data_parallel: parallelism_types.append("DP")
                    if use_pipeline_parallel: parallelism_types.append("PP") 
                    if use_tensor_parallel: parallelism_types.append("TP")
                    parallelism_label = "+".join(parallelism_types)
                    
                    axs[2].plot(meaningful_iters, meaningful_mem, 
                               label=f"R{i+1} GPU{gpu_id} ({parallelism_label})", color=gpu_color, 
                               marker='.', markersize=2, alpha=0.6, linewidth=1, linestyle='-')
            
            # Plot average line
            meaningful_iters_avg = [it for idx, it in enumerate(iters_avg) if gpu_mems_avg[idx] > 0.001]
            meaningful_mem_avg = [mem for mem in gpu_mems_avg if mem > 0.001]
            if meaningful_iters_avg:
                parallelism_types = []
                if use_data_parallel: parallelism_types.append("DP")
                if use_pipeline_parallel: parallelism_types.append("PP")
                if use_tensor_parallel: parallelism_types.append("TP")
                parallelism_label = "+".join(parallelism_types)
                
                axs[2].plot(meaningful_iters_avg, meaningful_mem_avg, 
                           label=f"R{i+1} Average ({parallelism_label})", color=colors(i % colors.N), 
                           marker='o', markersize=4, alpha=0.9, linewidth=2, linestyle='--')
        else:
            # Standard single GPU or CPU mode
            iters_mem, gpu_mems = [], []
            for m in run_metrics:
                if m['type'] == 'train_iter' and 'gpu_mem_gb_max_iter' in m: 
                    iters_mem.append(m['iter']); gpu_mems.append(m['gpu_mem_gb_max_iter']); has_gpu_data = True
                elif m['type'] == 'eval' and 'gpu_mem_gb_current_eval' in m and m['iter'] not in iters_mem: 
                    iters_mem.append(m['iter']); gpu_mems.append(m['gpu_mem_gb_current_eval']); has_gpu_data = True
            
            if iters_mem:
                pts = sorted(zip(iters_mem, gpu_mems)); s_iters = [p[0] for p in pts]; s_mem = [p[1] for p in pts]
                meaningful_iters = [it for idx, it in enumerate(s_iters) if s_mem[idx] > 0.001]
                meaningful_mem = [mem for mem in s_mem if mem > 0.001]
                if meaningful_iters: 
                    axs[2].plot(meaningful_iters, meaningful_mem, label=f"Run {i+1} (Standard)", color=colors(i % colors.N), 
                               marker='.', markersize=3, alpha=0.7, linewidth=0.8)
    
    axs[2].set_title("Peak GPU Memory Usage", fontsize=14); axs[2].set_xlabel("Iteration", fontsize=12); axs[2].set_ylabel("Memory (GB)", fontsize=12)
    # Only add legend if there's actual GPU data and lines plotted
    if has_gpu_data and axs[2].get_lines():
        axs[2].legend(fontsize='x-small', loc='best')
    elif not has_gpu_data:
        axs[2].text(0.5, 0.5, 'No GPU Memory data or GPU not used.', ha='center', va='center', fontsize=9, color='gray', wrap=True)
    axs[2].grid(True, linestyle=':', alpha=0.7); axs[2].tick_params(axis='both', which='major', labelsize=10)

    # Plot 4: Final Validation Losses
    run_lbls, final_val_losses, valid_indices_bar = [], [], []
    for i, (run_metrics, run_cfg) in enumerate(zip(metrics_list_global, config_list_global)):
        eval_m = [m for m in run_metrics if m['type'] == 'eval' and 'val_loss' in m and not np.isnan(m['val_loss'])]
        
        # Create comprehensive label with all parallelism types
        parallelism_info = []
        if run_cfg.get('use_recompute_ui', False): parallelism_info.append("RC")
        if run_cfg.get('use_data_parallel_ui', False): parallelism_info.append("DP")
        if run_cfg.get('use_pipeline_parallel_ui', False): parallelism_info.append("PP")
        if run_cfg.get('use_tensor_parallel_ui', False): parallelism_info.append("TP")
        parallelism_label = "+".join(parallelism_info) if parallelism_info else "None"
        
        lbl_txt = (f"R{i+1}\nμB{run_cfg.get('micro_batch_size_ui','?')},"f"GA{run_cfg.get('grad_accumulation_steps_direct_ui','?')}\n"
                   f"Parallelism: {parallelism_label}")
        run_lbls.append(lbl_txt)
        if eval_m: final_val_losses.append(eval_m[-1]['val_loss']); valid_indices_bar.append(i)
        else: final_val_losses.append(np.nan)
    if valid_indices_bar:
        f_lbls = [run_lbls[i] for i in valid_indices_bar]; f_losses = [final_val_losses[i] for i in valid_indices_bar]
        x_idx = np.arange(len(f_lbls)); bar_w = max(0.1, min(0.6, 0.5 / (len(f_lbls) / 5 + 1)))
        bars = axs[3].bar(x_idx, f_losses, bar_w, color=[colors(i % colors.N) for i in valid_indices_bar])
        axs[3].set_xticks(x_idx); axs[3].set_xticklabels(f_lbls, rotation=45, ha="right", fontsize=8)
        if f_losses:
            max_l_val = max(filter(lambda x: not np.isnan(x), f_losses), default=1.0)
            for bar_idx, bar_item in enumerate(bars):
                yval = bar_item.get_height()
                if not np.isnan(yval): axs[3].text(bar_item.get_x() + bar_item.get_width()/2.0, yval + 0.02 * max_l_val , f'{yval:.3f}', ha='center', va='bottom', fontsize=8)
    else: axs[3].text(0.5, 0.5, 'No final validation loss data.', ha='center', va='center', fontsize=9, color='gray')
    axs[3].set_title("Final Validation Loss per Run", fontsize=14); axs[3].set_ylabel("Loss", fontsize=12)
    axs[3].grid(True, axis='y', linestyle=':', alpha=0.7); axs[3].tick_params(axis='y', which='major', labelsize=10)
    
    plt.tight_layout(pad=3.0, h_pad=3.0)
    return fig, "Plots updated based on all runs in history."

def analyze_results_gradio(metrics_list_global, config_list_global):
    if not metrics_list_global: return "No data to analyze. Please complete a training run."
    analysis = "## Comparative Analysis of Runs\n\n"
    for i, (run_metrics, run_cfg) in enumerate(zip(metrics_list_global, config_list_global)):
        analysis += f"### Run {i+1}:\n  - **Configuration**:\n"
        analysis += f"    - Dataset: `{run_cfg.get('dataset_name_ui', 'N/A')}`\n"
        analysis += f"    - Effective BS (calc): `{run_cfg.get('batch_size_effective_calculated', 'N/A')}`\n"
        analysis += f"    - Micro-BS: `{run_cfg.get('micro_batch_size_ui', 'N/A')}`\n"
        analysis += f"    - Grad Accum Steps: `{run_cfg.get('grad_accumulation_steps_direct_ui', 'N/A')}`\n"
        
        # Enhanced parallelism information
        parallelism_info = []
        if run_cfg.get('use_recompute_ui', False): parallelism_info.append("Gradient Checkpointing")
        if run_cfg.get('use_data_parallel_ui', False): parallelism_info.append("Data Parallelism")
        if run_cfg.get('use_pipeline_parallel_ui', False): parallelism_info.append("Pipeline Parallelism")
        if run_cfg.get('use_tensor_parallel_ui', False): parallelism_info.append("Tensor Parallelism")
        parallelism_used = ", ".join(parallelism_info) if parallelism_info else "None"
        
        analysis += f"    - Parallelism Techniques: `{parallelism_used}`\n"
        analysis += f"    - Max Iters: `{run_cfg.get('max_iters_ui', 'N/A')}`\n"
        eval_m = [m for m in run_metrics if m['type'] == 'eval' and 'val_loss' in m and not np.isnan(m['val_loss'])]
        final_prog_update = run_metrics[-1] if run_metrics else {}
        if eval_m: analysis += f"  - **Performance**:\n    - Best Val Loss: `{min(m['val_loss'] for m in eval_m):.4f}`\n    - Final Val Loss: `{eval_m[-1]['val_loss']:.4f}`\n"
        else: analysis += f"  - **Performance**: No valid evaluation data.\n"
        analysis += f"    - Total Training Time: `{final_prog_update.get('time_elapsed_total', 0):.2f}s`\n"
        max_gpu = 0.0;
        for m_item in run_metrics: max_gpu = max(max_gpu, m_item.get('gpu_mem_gb_max',0.0), m_item.get('gpu_mem_gb_max_iter',0.0), m_item.get('gpu_mem_gb_max_overall',0.0), m_item.get('gpu_mem_gb_current_eval',0.0))
        if max_gpu > 0.001: analysis += f"    - Peak GPU Memory: `{max_gpu:.2f} GB`\n"
        
        # Add information about per-GPU tracking if any parallelism was used
        if any([run_cfg.get('use_data_parallel_ui', False), run_cfg.get('use_pipeline_parallel_ui', False), run_cfg.get('use_tensor_parallel_ui', False)]):
            analysis += f"    - 📊 **Per-GPU memory tracking enabled** (see individual GPU lines in plots)\n"
        
        analysis += "\n---\n"
    return analysis

def create_gradio_interface():
    current_run_metrics_buffer = []
    def run_training_interface_generator(
        dataset_name_ui, block_size_ui, vocab_size_ui, n_layer_ui, n_head_ui, n_embd_ui, dropout_ui, bias_ui,
        micro_batch_size_ui, grad_accumulation_steps_direct_ui, learning_rate_ui, max_iters_ui,
        weight_decay_ui, beta1_ui, beta2_ui, grad_clip_ui, decay_lr_ui, warmup_iters_ui, lr_decay_iters_ui, min_lr_ui,
        use_recompute_ui, use_data_parallel_ui, use_pipeline_parallel_ui, use_tensor_parallel_ui,
        out_dir_gradio_ui, log_interval_gradio_ui, eval_interval_gradio_ui, eval_iters_gradio_ui
    ):
        nonlocal current_run_metrics_buffer; current_run_metrics_buffer = []
        try:
            params_numeric = {k: float(v) for k, v in {"lr": learning_rate_ui, "wd": weight_decay_ui, "b1": beta1_ui, "b2": beta2_ui, "gc": grad_clip_ui, "mlr": min_lr_ui, "drop": dropout_ui}.items()}
            params_int = {k: int(v) for k, v in {"bs": block_size_ui, "vs": vocab_size_ui, "nl": n_layer_ui, "nh": n_head_ui, "ne": n_embd_ui,
                                                 "mbs": micro_batch_size_ui, "gas": grad_accumulation_steps_direct_ui, "mi": max_iters_ui,
                                                 "wi": warmup_iters_ui, "lrdi": lr_decay_iters_ui, "logi": log_interval_gradio_ui,
                                                 "evali": eval_interval_gradio_ui, "evalits": eval_iters_gradio_ui}.items()}
        except ValueError as ve:
            error_msg = f"Input Error: Check numerical fields. Details: {ve}"; empty_fig, _ = visualize_results_gradio([],[])
            # CORRECTED YIELD (4 items)
            yield empty_fig, error_msg, f"Config Error: {error_msg}", f"Analysis not available due to config error: {error_msg}"; return

        if not params_int["mbs"] > 0: error_msg = "Error: Micro-Batch Size must be > 0."
        elif not params_int["gas"] > 0: error_msg = "Error: Grad Accumulation Steps must be > 0."
        else: error_msg = None
        if error_msg: empty_fig, _ = visualize_results_gradio([], []); yield empty_fig, error_msg, error_msg, error_msg; return # CORRECTED YIELD

        batch_size_effective_calculated = params_int["mbs"] * params_int["gas"]
        current_config_summary_dict = {
            "dataset_name_ui": str(dataset_name_ui), "block_size_ui": params_int["bs"], "vocab_size_ui": params_int["vs"],
            "n_layer_ui": params_int["nl"], "n_head_ui": params_int["nh"], "n_embd_ui": params_int["ne"], "dropout_ui": params_numeric["drop"],
            "bias_ui": bool(bias_ui), "batch_size_effective_calculated": batch_size_effective_calculated,
            "micro_batch_size_ui": params_int["mbs"], "grad_accumulation_steps_direct_ui": params_int["gas"],
            "learning_rate_ui": params_numeric["lr"], "max_iters_ui": params_int["mi"],
            "use_recompute_ui": bool(use_recompute_ui), "use_data_parallel_ui": bool(use_data_parallel_ui),
            "use_pipeline_parallel_ui": bool(use_pipeline_parallel_ui), "use_tensor_parallel_ui": bool(use_tensor_parallel_ui),
            "out_dir_gradio_ui": str(out_dir_gradio_ui),
        }
        
        empty_fig, _= visualize_results_gradio([], [])
        # CORRECTED YIELD (4 items)
        yield empty_fig, "Starting training...", "Initializing...", "Analysis will be available after the run."

        training_generator = run_nanoGPT_training(
            block_size=params_int["bs"], vocab_size=params_int["vs"], n_layer=params_int["nl"], n_head=params_int["nh"],
            n_embd=params_int["ne"], dropout=params_numeric["drop"], bias=bool(bias_ui),
            batch_size_effective=batch_size_effective_calculated, micro_batch_size_ui=params_int["mbs"],
            learning_rate=params_numeric["lr"], max_iters=params_int["mi"], weight_decay=params_numeric["wd"],
            beta1=params_numeric["b1"], beta2=params_numeric["b2"], grad_clip=params_numeric["gc"], decay_lr=bool(decay_lr_ui),
            warmup_iters=params_int["wi"], lr_decay_iters=int(min(params_int["lrdi"], params_int["mi"])), min_lr=params_numeric["mlr"],
            grad_accumulation_steps_ui=params_int["gas"], use_recompute_ui=bool(use_recompute_ui),
            use_data_parallel_ui=bool(use_data_parallel_ui), use_pipeline_parallel_ui=bool(use_pipeline_parallel_ui),
            use_tensor_parallel_ui=bool(use_tensor_parallel_ui), dataset_name=str(dataset_name_ui),
            out_dir_ui=str(out_dir_gradio_ui), log_interval_ui=params_int["logi"],
            eval_interval_ui=params_int["evali"], eval_iters_ui=params_int["evalits"]
        )

        log_stream_for_ui = ""; live_plot_fig = empty_fig
        for progress_update in training_generator:
            current_run_metrics_buffer.append(progress_update.copy()); status_label_text = "Running..."
            log_entry = ""
            if progress_update["type"] == "error":
                status_label_text = f"Error: {progress_update['message']}"; log_entry = f"[ERROR] {status_label_text}"
                empty_fig_err, _ = visualize_results_gradio([],[]); yield empty_fig_err, status_label_text, f"{log_entry}\n{log_stream_for_ui}", status_label_text; return
            elif progress_update["type"] == "eval":
                status_label_text = f"Iter {progress_update['iter']}: Val Loss {progress_update.get('val_loss', float('nan')):.4f}"
                log_entry = f"[EVAL] Iter {progress_update['iter']}: Val={progress_update.get('val_loss', float('nan')):.4f}, TrainEst={progress_update.get('train_loss_est', float('nan')):.4f}, LR={progress_update.get('lr', 0):.2e}, GPU Mem Max: {progress_update.get('gpu_mem_gb_max', 0.0):.2f}GB"
            elif progress_update["type"] == "train_iter":
                status_label_text = f"Iter {progress_update['iter']}: Train Loss {progress_update.get('loss', float('nan')):.4f}"
                log_entry = f"[TRAIN] Iter {progress_update['iter']}: Loss={progress_update.get('loss', float('nan')):.4f}, LR={progress_update.get('lr',0):.2e}, dt={progress_update.get('time_per_iter_ms',0):.0f}ms, GPU Mem Max: {progress_update.get('gpu_mem_gb_max_iter',0.0):.2f}GB"
            elif progress_update["type"] == "info": log_entry = f"[INFO] {progress_update['message']}"
            elif progress_update["type"] == "finished": status_label_text = progress_update["message"]; log_entry = f"[FINISHED] {progress_update['message']}"
            
            log_stream_for_ui = f"{log_entry}\n{log_stream_for_ui}"
            if len(log_stream_for_ui) > 4000: log_stream_for_ui = log_stream_for_ui[:log_stream_for_ui.rfind('\n', 0, 3800)] + "\n... (log truncated)"
            
            live_plot_fig, _ = visualize_results_gradio([current_run_metrics_buffer], [current_config_summary_dict])
            # This yield has 4 items, matching outputs
            yield live_plot_fig, status_label_text, log_stream_for_ui, "" # No analysis update during run

        if current_run_metrics_buffer and current_run_metrics_buffer[-1]["type"] != "error":
            history["metrics_list"].append(list(current_run_metrics_buffer)); history["config_list"].append(current_config_summary_dict)
            final_comparative_plot_fig, plot_msg = visualize_results_gradio(history["metrics_list"], history["config_list"])
            final_analysis_text = analyze_results_gradio(history["metrics_list"], history["config_list"])
            final_status = "Training Finished!"
            if current_run_metrics_buffer and current_run_metrics_buffer[-1]["type"] == "finished": final_status = current_run_metrics_buffer[-1]["message"]
            # This yield has 4 items
            yield final_comparative_plot_fig, final_status, log_stream_for_ui, final_analysis_text
        else:
            error_msg_final = "Training ended with error or no data."
            if current_run_metrics_buffer and current_run_metrics_buffer[-1]["type"] == "error": error_msg_final = current_run_metrics_buffer[-1]["message"]
            empty_fig_final, _ = visualize_results_gradio([], [])
            # This yield has 4 items
            yield empty_fig_final, error_msg_final, log_stream_for_ui, error_msg_final

    def clear_history_gradio():
        nonlocal current_run_metrics_buffer; history["metrics_list"] = []; history["config_list"] = []; current_run_metrics_buffer = []
        empty_fig_live, _ = visualize_results_gradio([], []); empty_fig_overall, _ = visualize_results_gradio([], [])
        # This return has 5 items, but clear_button.click outputs are 5. This is fine.
        # The warning was for train_button.click which has 4 outputs.
        return empty_fig_live, "History Cleared. Ready for new run.", "", "", empty_fig_overall # For live_plot, status, log, analysis, overall_plot

    def refresh_overall_plots_and_analysis_wrapper():
        fig, _ = visualize_results_gradio(history["metrics_list"], history["config_list"])
        analysis = analyze_results_gradio(history["metrics_list"], history["config_list"])
        return fig, analysis

    # CSS pour supprimer les bordures SAUF pour les graphiques
    custom_css = """
    /* SUPPRESSION DES BORDURES SAUF GRAPHIQUES */
    * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        animation: none !important;
        border-radius: 0 !important;
    }
    
    *:focus, *:hover, *:active, *:visited {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        animation: none !important;
        border-radius: 0 !important;
    }
    
    .gradio-container, .gradio-container * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        animation: none !important;
        border-radius: 0 !important;
    }
    
    .gradio-container *:focus, .gradio-container *:hover, .gradio-container *:active {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        animation: none !important;
        border-radius: 0 !important;
    }
    
    /* Suppression spécifique pour les éléments NON-graphiques */
    div:not([data-testid="plot"]):not(.plot-container), 
    span, section, article, main, aside, nav, header, footer, form, fieldset, legend, input, textarea, select, button, label {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }
    
    /* Cibler tous les éléments avec des classes contenant certains mots SAUF plots */
    [class*="border"]:not([data-testid="plot"]):not(.plot-container), 
    [class*="rounded"]:not([data-testid="plot"]):not(.plot-container), 
    [class*="shadow"]:not([data-testid="plot"]):not(.plot-container), 
    [class*="outline"]:not([data-testid="plot"]):not(.plot-container), 
    [class*="ring"]:not([data-testid="plot"]):not(.plot-container) {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }
    
    /* AJOUTER DES BORDURES POUR LES GRAPHIQUES SEULEMENT */
    .gradio-container [data-testid="plot"],
    .gradio-container .plot-container,
    .gradio-container [data-testid="plot"] > div,
    .gradio-container .plot-container > div {
        border: 2px solid #2563eb !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        padding: 8px !important;
        margin: 4px 0 !important;
    }
    
    /* Hover effect pour les graphiques */
    .gradio-container [data-testid="plot"]:hover,
    .gradio-container .plot-container:hover {
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    """
    
    with gr.Blocks(title="nanoGPT Parallelism Workbench", 
                   theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky),
                   css=custom_css) as demo:
        gr.Markdown("#  nanoGPT Parallelism Workbench")
        #gr.Markdown("## **Now with REAL implementations of all parallelism techniques!**")
        #gr.Markdown("✅ **Gradient Checkpointing** | ✅ **Pipeline Parallelism** | ✅ **Tensor Parallelism** | ✅ **Data Parallelism**")
        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Configuration & Live Training", id="config_tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("### nanoGPT Model & Dataset (Fixed)")
                        dataset_name_ui = gr.Dropdown(label="Dataset", choices=["shakespeare_char", "openwebtext", "shakespeare"], value="shakespeare_char", interactive=False)
                        block_size_ui = gr.Number(value=256, label="Block Size", interactive=False)
                        vocab_size_ui = gr.Number(value=50304, label="Vocab Size (overridden by meta.pkl if exists)", interactive=False)
                        n_layer_ui = gr.Number(value=6, label="Layers", interactive=False)
                        n_head_ui = gr.Number(value=6, label="Heads", interactive=False)
                        n_embd_ui = gr.Number(value=384, label="Embedding Dim", interactive=False)
                        dropout_ui = gr.Number(value=0.2, label="Dropout", interactive=False)
                        bias_ui = gr.Checkbox(value=False, label="Bias in Linear Layers?", interactive=False)
                        
                        gr.Markdown("### Training Hyperparameters (Adjustable)")
                        micro_batch_size_ui = gr.Slider(minimum=1, maximum=128, step=1, value=8, label="Micro-Batch Size (per GPU/fwd pass)", interactive=True)
                        grad_accumulation_steps_direct_ui = gr.Slider(minimum=1, maximum=64, step=1, value=5, label="Gradient Accumulation Steps", interactive=True)
                        effective_batch_size_display = gr.Textbox(label="Effective Batch Size (Calculated)", interactive=False, placeholder="Updates automatically")
                        learning_rate_ui = gr.Textbox(value="6e-4", label="Learning Rate", interactive=True)
                        max_iters_ui = gr.Slider(minimum=10, maximum=20000, step=10, value=2000, label="Max Iterations", interactive=True)
                        with gr.Accordion("Advanced Training Hyperparameters", open=False):
                            weight_decay_ui = gr.Textbox(value="1e-1", label="Weight Decay", interactive=True)
                            beta1_ui = gr.Textbox(value="0.9", label="Adam Beta1", interactive=True)
                            beta2_ui = gr.Textbox(value="0.95", label="Adam Beta2", interactive=True)
                            grad_clip_ui = gr.Textbox(value="1.0", label="Gradient Clipping (0 for none)", interactive=True)
                            decay_lr_ui = gr.Checkbox(value=True, label="Decay Learning Rate?", interactive=True)
                            warmup_iters_ui = gr.Slider(minimum=0, maximum=2000, step=10, value=100, label="Warmup Iterations", interactive=True)
                            lr_decay_iters_ui = gr.Slider(minimum=10, maximum=20000, step=10, value=2000, label="LR Decay Iterations (auto-capped at max_iters)", interactive=True)
                            min_lr_ui = gr.Textbox(value="6e-5", label="Minimum Learning Rate", interactive=True)

                        gr.Markdown("### Optimization Techniques (Adjustable)")
                       
                        use_recompute_ui = gr.Checkbox(label="Gradient Checkpointing (Activation Recomputation)", value=False, interactive=True)
                        
                        #gr.Markdown("💡 **Data Parallelism**: Replicates model across GPUs, each processes different batches.")
                        use_data_parallel_ui = gr.Checkbox(label="Data Parallelism (nn.DataParallel - enables per-GPU memory tracking)", value=torch.cuda.is_available() and torch.cuda.device_count() > 1, interactive=True)
                        
                        #gr.Markdown("💡 **Pipeline Parallelism**: Splits model layers across GPUs - each GPU processes different layers.")
                        use_pipeline_parallel_ui = gr.Checkbox(label="Pipeline Parallelism", value=False, interactive=True)
                        
                        #gr.Markdown("💡 **Tensor Parallelism**: Splits operations within layers across GPUs - each GPU processes part of each operation.")
                        use_tensor_parallel_ui = gr.Checkbox(label="Tensor Parallelism", value=False, interactive=True)
                        
                        gr.Markdown("### Logging & Output (Adjustable)")
                        out_dir_gradio_ui = gr.Textbox(value="out-gradio-run", label="Output SubDirectory (relative to app location)", interactive=False)
                        log_interval_gradio_ui = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Log Interval (Iterations)", interactive=True)
                        eval_interval_gradio_ui = gr.Slider(minimum=10, maximum=1000, step=10, value=100, label="Evaluation Interval (Iterations)", interactive=True)
                        eval_iters_gradio_ui = gr.Slider(minimum=1, maximum=200, step=1, value=20, label="Eval Iterations (for loss estimation)", interactive=True)
                        with gr.Row():
                            train_button = gr.Button("🚀 Launch nanoGPT Training", variant="primary", scale=2)
                            clear_button = gr.Button("🧹 Clear All Run History", scale=1)
                    with gr.Column(scale=2, min_width=600):
                        gr.Markdown("### Live Training Progress")
                        gr.Markdown("** GPU Memory Tracking**: When ANY parallelism is enabled (Data/Pipeline/Tensor), the GPU Memory plot will show individual lines for each GPU plus the average.")
                     
                        status_label_ui = gr.Label(value="Ready. Configure and Launch.", container=False)
                        live_plot_output_ui = gr.Plot(container=False) 
                        progress_log_ui = gr.Textbox(label="Training Log Stream (Latest First)", lines=15, max_lines=30, interactive=False, autoscroll=False, container=False)
            with gr.TabItem("📊 Comparative Results & Analysis (All Runs)", id="results_tab"):
                #gr.Markdown("### Aggregated Results Across All Runs in This Session")
                overall_plot_output_ui = gr.Plot(container=False) # This will be cleared by clear_button
                overall_analysis_output_ui = gr.Markdown(label="Automated Analysis of All Runs") # This will be cleared by clear_button via run_training_interface_generator's yield
                refresh_button = gr.Button("🔄 Refresh Overall Plots & Analysis")
                gr.Markdown("*Note: Plots and analysis here update after each run completes or when this refresh button is clicked.*")

        def update_effective_bs_display_wrapper(micro_bs, accum_steps):
            try: return str(int(micro_bs) * int(accum_steps))
            except ValueError: return "Invalid input"
        micro_batch_size_ui.change(update_effective_bs_display_wrapper, inputs=[micro_batch_size_ui, grad_accumulation_steps_direct_ui], outputs=effective_batch_size_display)
        grad_accumulation_steps_direct_ui.change(update_effective_bs_display_wrapper, inputs=[micro_batch_size_ui, grad_accumulation_steps_direct_ui], outputs=effective_batch_size_display)

        # train_button outputs 4 components
        train_button_inputs = [
            dataset_name_ui, block_size_ui, vocab_size_ui, n_layer_ui, n_head_ui, n_embd_ui, dropout_ui, bias_ui,
            micro_batch_size_ui, grad_accumulation_steps_direct_ui, learning_rate_ui, max_iters_ui,
            weight_decay_ui, beta1_ui, beta2_ui, grad_clip_ui, decay_lr_ui, warmup_iters_ui, lr_decay_iters_ui, min_lr_ui,
            use_recompute_ui, use_data_parallel_ui, use_pipeline_parallel_ui, use_tensor_parallel_ui,
            out_dir_gradio_ui, log_interval_gradio_ui, eval_interval_gradio_ui, eval_iters_gradio_ui
        ]
        train_button_outputs = [live_plot_output_ui, status_label_ui, progress_log_ui, overall_analysis_output_ui]
        train_button.click(fn=run_training_interface_generator, inputs=train_button_inputs, outputs=train_button_outputs)
        
        refresh_button.click(fn=refresh_overall_plots_and_analysis_wrapper, inputs=[], outputs=[overall_plot_output_ui, overall_analysis_output_ui])
        
        # clear_button outputs 5 components.
        # The outputs for clear_button need to match the 5 values it returns.
        # live_plot_output_ui, status_label_ui, progress_log_ui, overall_analysis_output_ui (from train_button)
        # AND overall_plot_output_ui needs to be cleared too.
        clear_button_outputs = [
            live_plot_output_ui,      # Cleared live plot
            status_label_ui,          # "History Cleared..."
            progress_log_ui,          # Cleared log
            overall_analysis_output_ui, # Cleared analysis text (via empty string from clear_history_gradio)
            overall_plot_output_ui    # Cleared overall plot
        ]
        clear_button.click(fn=clear_history_gradio, inputs=[], outputs=clear_button_outputs)
        return demo

if __name__ == "__main__":
    print("🚀 Starting Advanced nanoGPT Parallelism Workbench...")
    print("✅ REAL implementations now available for:")
    print("   - Gradient Checkpointing (Activation Recomputation)")
    print("   - Pipeline Parallelism (Layer distribution)")
    print("   - Tensor Parallelism (Operation splitting)")
    print("   - Data Parallelism (Model replication)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()): 
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"💡 With {torch.cuda.device_count()} GPUs, you can now use:")
        if torch.cuda.device_count() >= 2:
            print("   - Pipeline Parallelism (layers across GPUs)")
            print("   - Tensor Parallelism (operations across GPUs)")
        print("   - Data Parallelism (model replication)")
        print("   - Gradient Checkpointing (memory optimization)")
    else: 
        print("CUDA not available, running on CPU.")
        print("⚠️  Advanced parallelism features require CUDA GPUs.")
    
    gradio_app_instance = create_gradio_interface()
    try:
        gradio_app_instance.launch(share=True, debug=True)
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
