# gradio_app.py
import gradio as gr
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import sys # For path manipulation

# --- Path Setup for train_core and to help train_core find nanoGPT ---
# Get the directory where gradio_app.py is located
# e.g., /teamspace/studios/this_studio/
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Add this current_script_dir to sys.path.
# This allows Python to find train_core.py directly if it's in the same directory.
# It also helps train_core.py to correctly establish its own relative path to the
# nanoGPT subdirectory.
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

try:
    from train_core import run_nanoGPT_training
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'run_nanoGPT_training' from 'train_core.py'.")
    print(f"Ensure 'train_core.py' is in the same directory as 'gradio_app.py': {current_script_dir}")
    print(f"Current sys.path: {sys.path}")
    print(f"Original ImportError: {e}")
    # Attempt to give more specific advice if the error is about nanoGPT within train_core
    if 'nanoGPT' in str(e).lower() or 'model' in str(e).lower():
        print("\n---")
        print("The error might be originating from within 'train_core.py' when it tries to import from 'nanoGPT'.")
        print("Please ensure the following structure:")
        print(f"1. Your current directory: {current_script_dir}")
        print(f"2. 'train_core.py' is in this directory.")
        print(f"3. A directory named 'nanoGPT' (the cloned repository) is also directly inside this directory: {os.path.join(current_script_dir, 'nanoGPT')}")
        print("   The 'train_core.py' script expects to find 'nanoGPT/model.py' and other nanoGPT files there.")
        print("---\n")
    raise # Re-raise the exception to stop execution

# --- History and Visualization (Global state for the app session) ---
history = {
    "metrics_list": [], # List of lists (each inner list is metrics for one run)
    "config_list": []   # List of dicts (each dict is config for one run)
}

def visualize_results_gradio(metrics_list_global, config_list_global):
    if not metrics_list_global: # Check if the global list is empty
        fig, axs = plt.subplots(2, 2, figsize=(12, 8)) # Create empty plot structure
        titles = ["Loss Curves vs. Iteration", "Time per Evaluation Interval", "Peak GPU Memory at Eval Points", "Final Validation Loss per Run"]
        for i_row in range(2):
            for j_col in range(2):
                axs[i_row,j_col].text(0.5, 0.5, 'No data yet. Run training first.', ha='center', va='center', fontsize=9)
                axs[i_row,j_col].set_title(titles[i_row*2+j_col], fontsize=10)
        plt.tight_layout()
        return fig, "No data to visualize yet. Please complete a training run."
    
    fig, axs = plt.subplots(2, 2, figsize=(17, 12)) # Adjusted figsize for better legend space & labels
    num_runs = len(metrics_list_global)
    try:
        # Using a qualitative colormap for better distinction between lines
        colors = plt.cm.get_cmap('tab10', max(10, num_runs)) # tab10 is good for up to 10 distinct colors
    except ValueError: # Fallback if tab10 is not found or num_runs is problematic for it
        colors = plt.cm.get_cmap('viridis', max(10, num_runs))


    # --- 1. Loss Curves (Validation and Estimated Training Loss vs. Iteration) ---
    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        # Filter for 'eval' type metrics that contain the necessary keys
        eval_data_points = [m for m in run_metrics if m['type'] == 'eval' and 'iter' in m and 'val_loss' in m and 'train_loss_est' in m]
        
        iters_eval = [m['iter'] for m in eval_data_points]
        val_losses = [m['val_loss'] for m in eval_data_points]
        train_losses_est = [m['train_loss_est'] for m in eval_data_points]
        
        # Create a concise label for the legend
        config_label = (f"R{i+1} μB{run_config_summary.get('micro_batch_size_ui', '?')},"
                        f"GA{run_config_summary.get('grad_accumulation_steps_direct_ui', '?')}," # Updated key
                        f"RC{run_config_summary.get('use_recompute_ui', False)},"
                        f"DP{run_config_summary.get('use_data_parallel_ui', False)}")

        if iters_eval and val_losses:
            axs[0, 0].plot(iters_eval, val_losses, label=f"{config_label} Val", color=colors(i % colors.N), linestyle='-', marker='.', markersize=5, alpha=0.9)
        if iters_eval and train_losses_est:
            axs[0, 0].plot(iters_eval, train_losses_est, label=f"{config_label} TrainEst", color=colors(i % colors.N), linestyle='--', marker='x', markersize=5, alpha=0.9)
    
    axs[0, 0].set_title("Loss Curves vs. Iteration", fontsize=11)
    axs[0, 0].set_xlabel("Iteration", fontsize=10)
    axs[0, 0].set_ylabel("Loss", fontsize=10)
    if metrics_list_global : axs[0, 0].legend(fontsize='xx-small', loc='best', ncol=1) # ncol for multi-column legend if too many items
    axs[0, 0].grid(True, linestyle=':', alpha=0.7)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=9)


    # --- 2. Time per Evaluation Interval ---
    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        eval_data_points = [m for m in run_metrics if m['type'] == 'eval' and 'iter' in m and 'time_elapsed_total' in m]
        iters_eval = [m['iter'] for m in eval_data_points]
        time_elapsed_at_eval = [m['time_elapsed_total'] for m in eval_data_points]
        
        interval_times = []
        valid_iters_for_interval_plot = [] # Store iterations where interval_times are calculated
        if time_elapsed_at_eval:
            interval_times.append(time_elapsed_at_eval[0]) # Time for the first interval
            valid_iters_for_interval_plot.append(iters_eval[0])
            for j in range(1, len(time_elapsed_at_eval)):
                interval_times.append(time_elapsed_at_eval[j] - time_elapsed_at_eval[j-1])
                valid_iters_for_interval_plot.append(iters_eval[j])
        
        config_label = f"Run {i+1}" # Simpler label for this plot
        if valid_iters_for_interval_plot and interval_times:
            axs[0, 1].plot(valid_iters_for_interval_plot, interval_times, label=config_label, color=colors(i % colors.N), marker='.', markersize=5, alpha=0.9)

    axs[0, 1].set_title("Time per Evaluation Interval", fontsize=11)
    axs[0, 1].set_xlabel("Iteration (at end of interval)", fontsize=10)
    axs[0, 1].set_ylabel("Time (s)", fontsize=10)
    if metrics_list_global: axs[0, 1].legend(fontsize='xx-small')
    axs[0, 1].grid(True, linestyle=':', alpha=0.7)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=9)


    # --- 3. Max GPU Memory per Evaluation Interval (MODIFIED TO PLOT DATA) ---
    has_gpu_data = False
    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        # Check for 'gpu_mem_gb_current_eval' or 'gpu_mem_gb_max_iter' (more frequent)
        # Let's use 'gpu_mem_gb_max_iter' from 'train_iter' type for a denser plot,
        # or 'gpu_mem_gb_current_eval' from 'eval' if the other is not available.
        iters_with_mem = []
        gpu_mem_values = []

        for m in run_metrics:
            if m['type'] == 'train_iter' and 'gpu_mem_gb_max_iter' in m:
                iters_with_mem.append(m['iter'])
                gpu_mem_values.append(m['gpu_mem_gb_max_iter'])
                has_gpu_data = True
            elif m['type'] == 'eval' and 'gpu_mem_gb_current_eval' in m and m['iter'] not in iters_with_mem: # Avoid duplicate iter points
                iters_with_mem.append(m['iter'])
                gpu_mem_values.append(m['gpu_mem_gb_current_eval'])
                has_gpu_data = True
        
        # Sort by iteration before plotting, in case data points are out of order
        if iters_with_mem:
            sorted_points = sorted(zip(iters_with_mem, gpu_mem_values))
            sorted_iters = [p[0] for p in sorted_points]
            sorted_mem = [p[1] for p in sorted_points]

            config_label = f"Run {i+1}"
            # Filter out zero memory if not meaningful (e.g. CPU runs or no allocation yet)
            meaningful_iters = [it for idx, it in enumerate(sorted_iters) if sorted_mem[idx] > 0.001]
            meaningful_mem = [mem for mem in sorted_mem if mem > 0.001]
            if meaningful_iters:
                 axs[1, 0].plot(meaningful_iters, meaningful_mem, label=config_label, color=colors(i % colors.N), marker='.', markersize=3, alpha=0.7, linewidth=0.8)
    
    axs[1, 0].set_title("Peak GPU Memory Usage", fontsize=11) # Updated title
    axs[1, 0].set_xlabel("Iteration", fontsize=10)
    axs[1, 0].set_ylabel("Memory (GB)", fontsize=10)
    if has_gpu_data:
        axs[1, 0].legend(fontsize='xx-small')
    else:
        axs[1, 0].text(0.5, 0.5, 'No GPU Memory data collected\nor GPU not used / Memory not tracked.', ha='center', va='center', fontsize=9, color='gray', wrap=True)
    axs[1, 0].grid(True, linestyle=':', alpha=0.7)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=9)


    # --- 4. Final Validation Losses Comparison (Bar Chart) ---
    run_labels = []
    final_val_losses_all_runs = []
    valid_run_indices_for_bar = [] 

    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        eval_metrics_for_run = [m for m in run_metrics if m['type'] == 'eval' and 'val_loss' in m and not np.isnan(m['val_loss'])]
        
        label_text = (f"R{i+1}\nμB{run_config_summary.get('micro_batch_size_ui','?')},"
                      f"GA{run_config_summary.get('grad_accumulation_steps_direct_ui','?')}\n" # Updated key
                      f"RC:{run_config_summary.get('use_recompute_ui',False)},"
                      f"DP:{run_config_summary.get('use_data_parallel_ui',False)}")
        run_labels.append(label_text)

        if eval_metrics_for_run:
            final_val_losses_all_runs.append(eval_metrics_for_run[-1]['val_loss'])
            valid_run_indices_for_bar.append(i)
        else: 
            final_val_losses_all_runs.append(np.nan)

    if valid_run_indices_for_bar:
        filtered_labels = [run_labels[i] for i in valid_run_indices_for_bar]
        filtered_losses = [final_val_losses_all_runs[i] for i in valid_run_indices_for_bar]
        x_indices = np.arange(len(filtered_labels))
        bar_width = 0.5 / (len(filtered_labels) / 5 + 1) # Make bars thinner if many runs
        bar_width = max(0.1, min(0.6, bar_width))


        bars = axs[1, 1].bar(x_indices, filtered_losses, bar_width, color=[colors(i % colors.N) for i in valid_run_indices_for_bar])
        axs[1, 1].set_xticks(x_indices)
        axs[1, 1].set_xticklabels(filtered_labels, rotation=45, ha="right", fontsize=7) # Smaller font for labels
        
        # Add text labels on bars if there's space
        if filtered_losses: # Check if list is not empty before finding max
            max_loss_val = max(filter(lambda x: not np.isnan(x), filtered_losses), default=1.0) 
            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height()
                if not np.isnan(yval): 
                    axs[1,1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02 * max_loss_val , f'{yval:.3f}', ha='center', va='bottom', fontsize=6) # Smaller font for bar text
    else: 
        axs[1,1].text(0.5, 0.5, 'No final validation loss data to display.', ha='center', va='center', fontsize=9, color='gray')

    axs[1, 1].set_title("Final Validation Loss per Run", fontsize=11)
    axs[1, 1].set_ylabel("Loss", fontsize=10)
    axs[1, 1].grid(True, axis='y', linestyle=':', alpha=0.7)
    axs[1, 1].tick_params(axis='y', which='major', labelsize=9)

    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0) # Adjust padding
    return fig, "Plots updated based on all runs in history."

# --- analyze_results_gradio function (using grad_accumulation_steps_direct_ui) ---
def analyze_results_gradio(metrics_list_global, config_list_global):
    if not metrics_list_global:
        return "No data to analyze. Please complete a training run."
    
    analysis_text = "## Comparative Analysis of Runs\n\n"
    
    for i, (run_metrics, run_config_summary) in enumerate(zip(metrics_list_global, config_list_global)):
        analysis_text += f"### Run {i+1}:\n"
        analysis_text += f"  - **Configuration**:\n"
        analysis_text += f"    - Dataset: `{run_config_summary.get('dataset_name_ui', 'N/A')}`\n"
        analysis_text += f"    - Effective BS (calc): `{run_config_summary.get('batch_size_effective_calculated', 'N/A')}`\n" # Display calculated
        analysis_text += f"    - Micro-BS: `{run_config_summary.get('micro_batch_size_ui', 'N/A')}`\n"
        analysis_text += f"    - Grad Accum Steps: `{run_config_summary.get('grad_accumulation_steps_direct_ui', 'N/A')}`\n" # Display direct input
        analysis_text += f"    - Recompute: `{'On' if run_config_summary.get('use_recompute_ui', False) else 'Off'}`\n"
        analysis_text += f"    - DataParallel: `{'On' if run_config_summary.get('use_data_parallel_ui', False) else 'Off'}`\n"
        analysis_text += f"    - PipelineParallel (Flag): `{'On' if run_config_summary.get('use_pipeline_parallel_ui', False) else 'Off'}`\n"
        analysis_text += f"    - TensorParallel (Flag): `{'On' if run_config_summary.get('use_tensor_parallel_ui', False) else 'Off'}`\n"
        analysis_text += f"    - Max Iters: `{run_config_summary.get('max_iters_ui', 'N/A')}`\n"
        analysis_text += f"    - LR: `{run_config_summary.get('learning_rate_ui', 'N/A')}`\n"

        eval_metrics_for_run = [m for m in run_metrics if m['type'] == 'eval' and 'val_loss' in m and not np.isnan(m['val_loss'])]
        # Find the last 'finished' or 'error' or last eval metric for total time
        final_progress_update = {}
        if run_metrics:
            for m_idx in range(len(run_metrics) -1, -1, -1):
                if run_metrics[m_idx]['type'] in ['finished', 'error', 'eval']:
                    final_progress_update = run_metrics[m_idx]
                    break
        
        if eval_metrics_for_run:
            best_val_loss_this_run = min(m['val_loss'] for m in eval_metrics_for_run)
            final_val_loss_this_run = eval_metrics_for_run[-1]['val_loss']
            analysis_text += f"  - **Performance**:\n"
            analysis_text += f"    - Best Validation Loss: `{best_val_loss_this_run:.4f}`\n"
            analysis_text += f"    - Final Validation Loss: `{final_val_loss_this_run:.4f}`\n"
        else:
            analysis_text += f"  - **Performance**: No valid evaluation data recorded for this run.\n"

        total_time_s = final_progress_update.get('time_elapsed_total', 0)
        analysis_text += f"    - Total Training Time: `{total_time_s:.2f} seconds`\n"
        
        max_gpu_mem_overall = 0.0
        for m_item in run_metrics: # Find overall max GPU memory for the run
            max_gpu_mem_overall = max(max_gpu_mem_overall, m_item.get('gpu_mem_gb_max', 0.0))
            max_gpu_mem_overall = max(max_gpu_mem_overall, m_item.get('gpu_mem_gb_max_iter', 0.0))
            max_gpu_mem_overall = max(max_gpu_mem_overall, m_item.get('gpu_mem_gb_max_overall', 0.0)) # from finished
            max_gpu_mem_overall = max(max_gpu_mem_overall, m_item.get('gpu_mem_gb_current_eval',0.0))
        if max_gpu_mem_overall > 0.001 :
            analysis_text += f"    - Peak GPU Memory during run: `{max_gpu_mem_overall:.2f} GB`\n"

        analysis_text += "\n---\n"
            
    return analysis_text


# --- Gradio Interface Function ---
def create_gradio_interface():
    current_run_metrics_buffer = [] 

    # --- run_training_interface_generator ---
    # SIGNATURE MUST MATCH THE train_button.click() inputs list EXACTLY
    def run_training_interface_generator(
        dataset_name_ui, block_size_ui, vocab_size_ui, n_layer_ui, n_head_ui, n_embd_ui, dropout_ui, bias_ui,
        micro_batch_size_ui, grad_accumulation_steps_direct_ui, # Modified Grad Accum
        learning_rate_ui, max_iters_ui,
        weight_decay_ui, beta1_ui, beta2_ui, grad_clip_ui,
        decay_lr_ui, warmup_iters_ui, lr_decay_iters_ui, min_lr_ui,
        use_recompute_ui, use_data_parallel_ui,
        use_pipeline_parallel_ui, use_tensor_parallel_ui, # New flags
        out_dir_gradio_ui, log_interval_gradio_ui, eval_interval_gradio_ui, eval_iters_gradio_ui
    ):
        nonlocal current_run_metrics_buffer
        current_run_metrics_buffer = [] 

        try:
            learning_rate_f = float(learning_rate_ui); weight_decay_f = float(weight_decay_ui)
            beta1_f = float(beta1_ui); beta2_f = float(beta2_ui); grad_clip_f = float(grad_clip_ui)
            min_lr_f = float(min_lr_ui); block_size_i = int(block_size_ui)
            vocab_size_i = int(vocab_size_ui); n_layer_i = int(n_layer_ui)
            n_head_i = int(n_head_ui); n_embd_i = int(n_embd_ui); dropout_f = float(dropout_ui)
            # batch_size_effective_i removed as direct input, now micro_batch and grad_accum are direct
            micro_batch_size_i = int(micro_batch_size_ui)
            grad_accumulation_steps_direct_i = int(grad_accumulation_steps_direct_ui) # Use this directly
            max_iters_i = int(max_iters_ui)
            warmup_iters_i = int(warmup_iters_ui); lr_decay_iters_i = int(lr_decay_iters_ui)
            log_interval_i = int(log_interval_gradio_ui); eval_interval_i = int(eval_interval_gradio_ui)
            eval_iters_i = int(eval_iters_gradio_ui)
        except ValueError as ve:
            error_msg = f"Input Error: Check numerical fields. Details: {ve}"; empty_fig, _ = visualize_results_gradio([],[])
            yield empty_fig, error_msg, "Config Error.", "", error_msg; return

        if not micro_batch_size_i > 0: error_msg = "Error: Micro-Batch Size must be > 0."
        elif not grad_accumulation_steps_direct_i > 0: error_msg = "Error: Grad Accumulation Steps must be > 0."
        else: error_msg = None
        if error_msg: empty_fig, _ = visualize_results_gradio([], []); yield empty_fig, error_msg, "Config Error.", "", error_msg; return

        batch_size_effective_calculated = micro_batch_size_i * grad_accumulation_steps_direct_i

        current_config_summary_dict = {
            "dataset_name_ui": str(dataset_name_ui), "block_size_ui": block_size_i, "vocab_size_ui": vocab_size_i,
            "n_layer_ui": n_layer_i, "n_head_ui": n_head_i, "n_embd_ui": n_embd_i, "dropout_ui": dropout_f,
            "bias_ui": bool(bias_ui),
            "batch_size_effective_calculated": batch_size_effective_calculated, # Store calculated
            "micro_batch_size_ui": micro_batch_size_i,
            "grad_accumulation_steps_direct_ui": grad_accumulation_steps_direct_i, # Store direct input
            "learning_rate_ui": learning_rate_f, "max_iters_ui": max_iters_i,
            "use_recompute_ui": bool(use_recompute_ui), "use_data_parallel_ui": bool(use_data_parallel_ui),
            "use_pipeline_parallel_ui": bool(use_pipeline_parallel_ui), 
            "use_tensor_parallel_ui": bool(use_tensor_parallel_ui),    
            "out_dir_gradio_ui": str(out_dir_gradio_ui),
        }
        
        empty_fig, _= visualize_results_gradio([], []) 
        yield empty_fig, "Starting training...", "Initializing...", "", ""

        training_generator = run_nanoGPT_training(
            block_size=block_size_i, vocab_size=vocab_size_i, n_layer=n_layer_i, n_head=n_head_i,
            n_embd=n_embd_i, dropout=dropout_f, bias=bool(bias_ui),
            batch_size_effective=batch_size_effective_calculated, # Pass calculated for logging in train_core
            micro_batch_size_ui=micro_batch_size_i,
            learning_rate=learning_rate_f, max_iters=max_iters_i, weight_decay=weight_decay_f,
            beta1=beta1_f, beta2=beta2_f, grad_clip=grad_clip_f, decay_lr=bool(decay_lr_ui),
            warmup_iters=warmup_iters_i, lr_decay_iters=int(lr_decay_iters_i if lr_decay_iters_i <= max_iters_i else max_iters_i),
            min_lr=min_lr_f,
            grad_accumulation_steps_ui=grad_accumulation_steps_direct_i, # Pass direct UI value
            use_recompute_ui=bool(use_recompute_ui), use_data_parallel_ui=bool(use_data_parallel_ui),
            use_pipeline_parallel_ui=bool(use_pipeline_parallel_ui), 
            use_tensor_parallel_ui=bool(use_tensor_parallel_ui),     
            dataset_name=str(dataset_name_ui), out_dir_ui=str(out_dir_gradio_ui),
            log_interval_ui=log_interval_i, eval_interval_ui=eval_interval_i, eval_iters_ui=eval_iters_i
        )

        log_stream_for_ui = ""; live_plot_fig = empty_fig 
        for progress_update in training_generator:
            current_run_metrics_buffer.append(progress_update.copy()); status_label_text = "Running..."
            if progress_update["type"] == "error":
                status_label_text = f"Error: {progress_update['message']}"; log_stream_for_ui = f"{status_label_text}\n{log_stream_for_ui}"
                empty_fig_err, _ = visualize_results_gradio([],[]); yield empty_fig_err, status_label_text, log_stream_for_ui, status_label_text; return 
            elif progress_update["type"] == "eval":
                status_label_text = f"Iter {progress_update['iter']}: Val Loss {progress_update.get('val_loss', float('nan')):.4f}"
                log_stream_for_ui = f"[EVAL] Iter {progress_update['iter']}: Val={progress_update.get('val_loss', float('nan')):.4f}, TrainEst={progress_update.get('train_loss_est', float('nan')):.4f}, LR={progress_update.get('lr', 0):.2e}, GPU Mem Max: {progress_update.get('gpu_mem_gb_max', 0.0):.2f}GB\n{log_stream_for_ui}"
            elif progress_update["type"] == "train_iter":
                status_label_text = f"Iter {progress_update['iter']}: Train Loss {progress_update.get('loss', float('nan')):.4f}"
                log_stream_for_ui = f"[TRAIN] Iter {progress_update['iter']}: Loss={progress_update.get('loss', float('nan')):.4f}, LR={progress_update.get('lr',0):.2e}, dt={progress_update.get('time_per_iter_ms',0):.0f}ms, GPU Mem Max: {progress_update.get('gpu_mem_gb_max_iter',0.0):.2f}GB\n{log_stream_for_ui}"
            elif progress_update["type"] == "info": # For TP/PP placeholders
                log_stream_for_ui = f"[INFO] {progress_update['message']}\n{log_stream_for_ui}"
            elif progress_update["type"] == "finished":
                status_label_text = progress_update["message"]; log_stream_for_ui = f"[FINISHED] {progress_update['message']}\n{log_stream_for_ui}"
            if len(log_stream_for_ui) > 4000: log_stream_for_ui = log_stream_for_ui[:4000] + "\n... (log truncated)"
            live_plot_fig, _ = visualize_results_gradio([current_run_metrics_buffer], [current_config_summary_dict])
            yield live_plot_fig, status_label_text, log_stream_for_ui, "" 

        if current_run_metrics_buffer and current_run_metrics_buffer[-1]["type"] != "error":
            history["metrics_list"].append(list(current_run_metrics_buffer)); history["config_list"].append(current_config_summary_dict)
            final_comparative_plot_fig, plot_msg = visualize_results_gradio(history["metrics_list"], history["config_list"])
            final_analysis_text = analyze_results_gradio(history["metrics_list"], history["config_list"])
            final_status = "Training Finished!"
            if current_run_metrics_buffer and current_run_metrics_buffer[-1]["type"] == "finished": final_status = current_run_metrics_buffer[-1]["message"]
            yield final_comparative_plot_fig, final_status, log_stream_for_ui, final_analysis_text
        else:
            error_msg_final = "Training ended with error or no data."
            if current_run_metrics_buffer and current_run_metrics_buffer[-1]["type"] == "error": error_msg_final = current_run_metrics_buffer[-1]["message"]
            empty_fig_final, _ = visualize_results_gradio([], []); yield empty_fig_final, error_msg_final, log_stream_for_ui, error_msg_final

    def clear_history_gradio():
        nonlocal current_run_metrics_buffer; history["metrics_list"] = []; history["config_list"] = []; current_run_metrics_buffer = []
        empty_fig_live, msg_live = visualize_results_gradio([], []); empty_fig_overall, msg_overall = visualize_results_gradio([], [])
        return empty_fig_live, "History Cleared. Ready for new run.", "", "", empty_fig_overall

    def refresh_overall_plots_and_analysis_wrapper():
        fig, _ = visualize_results_gradio(history["metrics_list"], history["config_list"])
        analysis = analyze_results_gradio(history["metrics_list"], history["config_list"])
        return fig, analysis

    with gr.Blocks(title="nanoGPT Optimization Workbench", theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:
        gr.Markdown("# nanoGPT Optimization Experiment Workbench")
        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Configuration & Live Training", id="config_tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("### nanoGPT Model & Dataset")
                        dataset_name_ui = gr.Dropdown(label="Dataset", choices=["shakespeare_char", "openwebtext", "shakespeare"], value="shakespeare_char", interactive=True)
                        block_size_ui = gr.Slider(minimum=32, maximum=1024, step=32, value=256, label="Block Size", interactive=True)
                        vocab_size_ui = gr.Number(value=50304, label="Vocab Size", interactive=True) # Actual from meta.pkl used
                        n_layer_ui = gr.Slider(minimum=1, maximum=48, step=1, value=6, label="Layers", interactive=True)
                        n_head_ui = gr.Slider(minimum=1, maximum=25, step=1, value=6, label="Heads", interactive=True)
                        n_embd_ui = gr.Slider(minimum=64, maximum=1600, step=32, value=384, label="Embedding Dim", interactive=True)
                        dropout_ui = gr.Slider(minimum=0.0, maximum=0.5, step=0.01, value=0.2, label="Dropout", interactive=True)
                        bias_ui = gr.Checkbox(value=False, label="Bias in Linear Layers?", interactive=True)
                        
                        gr.Markdown("### Training Hyperparameters")
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
                            lr_decay_iters_ui = gr.Slider(minimum=10, maximum=20000, step=10, value=2000, label="LR Decay Iterations (<= max_iters)", interactive=True)
                            min_lr_ui = gr.Textbox(value="6e-5", label="Minimum Learning Rate", interactive=True)

                        gr.Markdown("### Optimization Techniques")
                        use_recompute_ui = gr.Checkbox(label="Activation Recomputation", value=False, interactive=True)
                        use_data_parallel_ui = gr.Checkbox(label="Data Parallelism (nn.DataParallel)", value=torch.cuda.is_available() and torch.cuda.device_count() > 1, interactive=True)
                        use_pipeline_parallel_ui = gr.Checkbox(label="Pipeline Parallelism (Flag Only)", value=False, interactive=True) 
                        use_tensor_parallel_ui = gr.Checkbox(label="Tensor Parallelism (Flag Only)", value=False, interactive=True) 
                        
                        gr.Markdown("### Logging & Output")
                        out_dir_gradio_ui = gr.Textbox(value="out-gradio-run", label="Output SubDirectory", interactive=True)
                        log_interval_gradio_ui = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Log Interval", interactive=True)
                        eval_interval_gradio_ui = gr.Slider(minimum=10, maximum=1000, step=10, value=100, label="Evaluation Interval", interactive=True)
                        eval_iters_gradio_ui = gr.Slider(minimum=1, maximum=200, step=1, value=20, label="Eval Iterations", interactive=True)
                        with gr.Row():
                            train_button = gr.Button("🚀 Launch nanoGPT Training", variant="primary", scale=2)
                            clear_button = gr.Button("🧹 Clear All Run History", scale=1)
                    with gr.Column(scale=2, min_width=600):
                        gr.Markdown("### Live Training Progress")
                        status_label_ui = gr.Label(value="Ready. Configure and Launch.", label="Current Status")
                        live_plot_output_ui = gr.Plot(label="Current Run: Loss Curve & GPU Memory") 
                        progress_log_ui = gr.Textbox(label="Training Log Stream (Latest First)", lines=15, max_lines=30, interactive=False, autoscroll=False)
            with gr.TabItem("📊 Comparative Results & Analysis (All Runs)", id="results_tab"):
                gr.Markdown("### Aggregated Results Across All Runs in This Session")
                overall_plot_output_ui = gr.Plot(label="Comparison Plots Across All Runs")
                overall_analysis_output_ui = gr.Markdown(label="Automated Analysis of All Runs")
                refresh_button = gr.Button("🔄 Refresh Overall Plots & Analysis")
                gr.Markdown("*Note: Plots and analysis here update after each run completes or when this refresh button is clicked.*")

        def update_effective_bs_display_wrapper(micro_bs, accum_steps): # Wrapper for Gradio
            try: return str(int(micro_bs) * int(accum_steps))
            except ValueError: return "Invalid input"
        micro_batch_size_ui.change(update_effective_bs_display_wrapper, inputs=[micro_batch_size_ui, grad_accumulation_steps_direct_ui], outputs=effective_batch_size_display)
        grad_accumulation_steps_direct_ui.change(update_effective_bs_display_wrapper, inputs=[micro_batch_size_ui, grad_accumulation_steps_direct_ui], outputs=effective_batch_size_display)

        train_button.click(
            fn=run_training_interface_generator,
            inputs=[ 
                dataset_name_ui, block_size_ui, vocab_size_ui, n_layer_ui, n_head_ui, n_embd_ui, dropout_ui, bias_ui,
                micro_batch_size_ui, grad_accumulation_steps_direct_ui, # Updated
                learning_rate_ui, max_iters_ui,
                weight_decay_ui, beta1_ui, beta2_ui, grad_clip_ui,
                decay_lr_ui, warmup_iters_ui, lr_decay_iters_ui, min_lr_ui,
                use_recompute_ui, use_data_parallel_ui,
                use_pipeline_parallel_ui, use_tensor_parallel_ui, # Added
                out_dir_gradio_ui, log_interval_gradio_ui, eval_interval_gradio_ui, eval_iters_gradio_ui
            ],
            outputs=[ live_plot_output_ui, status_label_ui, progress_log_ui, overall_analysis_output_ui ]
        )
        refresh_button.click(fn=refresh_overall_plots_and_analysis_wrapper, inputs=[], outputs=[overall_plot_output_ui, overall_analysis_output_ui])
        clear_button.click(fn=clear_history_gradio, inputs=[], outputs=[live_plot_output_ui, status_label_ui, progress_log_ui, overall_analysis_output_ui, overall_plot_output_ui])
        return demo

if __name__ == "__main__":
    print("Starting Gradio Application for nanoGPT Optimization Experiments...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()): print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else: print("CUDA not available, running on CPU. Data Parallelism will be disabled if selected.")
    import matplotlib
    matplotlib.use('Agg')
    gradio_app_instance = create_gradio_interface()
    try: gradio_app_instance.launch(share=True, debug=True) 
    except Exception as e: print(f"Error launching Gradio app: {e}")