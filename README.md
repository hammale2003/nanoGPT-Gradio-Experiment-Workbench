# nanoGPT Gradio Experiment Workbench

This project provides an interactive web interface using Gradio to experiment with training Andrej Karpathy's nanoGPT model. It allows users to configure model architecture, training hyperparameters, and select various optimization techniques like gradient accumulation, activation recomputation, and data parallelism, then observe the training process and results in real-time.

## Features

* **Interactive Configuration**: Easily set nanoGPT model parameters (layers, heads, embedding dimension, etc.) and training hyperparameters (learning rate, batch sizes, iterations) through the Gradio UI.
* **Optimization Technique Toggling**:
    * Enable/disable **Gradient Accumulation** by setting micro-batch size and gradient accumulation steps.
    * Toggle **Activation Recomputation** (checkpointing) to save memory.
    * Toggle **Data Parallelism** (`nn.DataParallel`) for using multiple available GPUs on a single node.
    * UI placeholders for **Pipeline Parallelism** and **Tensor Parallelism** (these are flags passed to the training core but require manual implementation of the actual parallelism logic).
* **Live Monitoring**: View training progress, loss curves (live update for the current run), and estimated GPU memory usage (if CUDA is used and tracked).
* **Comparative Analysis**: After multiple runs with different configurations, view comparative plots for loss, training time, and (eventually) GPU memory. An automated analysis provides a summary of each run.
* **Modular Core**: The training logic is separated into `train_core.py`, which handles the nanoGPT model and training loop, making it adaptable.

## Project Structure
Okay, here's a draft for a README.md file for your project. This README assumes the current state of your project, including the Gradio interface for experimenting with nanoGPT and its various optimization techniques.

Markdown

# nanoGPT Gradio Experiment Workbench

This project provides an interactive web interface using Gradio to experiment with training Andrej Karpathy's nanoGPT model. It allows users to configure model architecture, training hyperparameters, and select various optimization techniques like gradient accumulation, activation recomputation, and data parallelism, then observe the training process and results in real-time.

## Features

* **Interactive Configuration**: Easily set nanoGPT model parameters (layers, heads, embedding dimension, etc.) and training hyperparameters (learning rate, batch sizes, iterations) through the Gradio UI.
* **Optimization Technique Toggling**:
    * Enable/disable **Gradient Accumulation** by setting micro-batch size and gradient accumulation steps.
    * Toggle **Activation Recomputation** (checkpointing) to save memory.
    * Toggle **Data Parallelism** (`nn.DataParallel`) for using multiple available GPUs on a single node.
    * UI placeholders for **Pipeline Parallelism** and **Tensor Parallelism** (these are flags passed to the training core but require manual implementation of the actual parallelism logic).
* **Live Monitoring**: View training progress, loss curves (live update for the current run), and estimated GPU memory usage (if CUDA is used and tracked).
* **Comparative Analysis**: After multiple runs with different configurations, view comparative plots for loss, training time, and (eventually) GPU memory. An automated analysis provides a summary of each run.
* **Modular Core**: The training logic is separated into `train_core.py`, which handles the nanoGPT model and training loop, making it adaptable.

## Project Structure

```.
├── nanoGPT/                  # Cloned official nanoGPT repository
│   ├── model.py              # (Potentially modified for activation recomputation flag)
│   ├── data/
│   │   ├── shakespeare_char/
│   │   │   ├── prepare.py    # Example data preparation script
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── your_app_directory/       # Or THIS_STUDIO, where your app files are
│   ├── gradio_app.py         # Main Gradio application script
│   ├── train_core.py         # Core training logic adapted from nanoGPT
│   └── out-gradio-run/       # Example output directory for checkpoints (created by the app)
└── README.md                 # This file
```


## Prerequisites

1.  **Python 3.8+**
2.  **PyTorch**: Install from [pytorch.org](https://pytorch.org/get-started/locally/). Ensure you have a version compatible with your CUDA version if using GPUs.
3.  **Cloned nanoGPT Repository**: You must have Andrej Karpathy's `nanoGPT` repository cloned. This project assumes it's located such that `train_core.py` can import from it (e.g., as a subdirectory within your main application folder, or ensure the `nanoGPT` folder is in your `PYTHONPATH`).
    ```bash
    git clone [https://github.com/karpathy/nanoGPT.git](https://github.com/karpathy/nanoGPT.git)
    ```
4.  **Python Packages**:
    ```bash
    pip install torch numpy gradio matplotlib pandas
    # nanoGPT itself might have other dependencies like tiktoken if you use its data prep scripts extensively
    # pip install tiktoken datasets tqdm # if running nanoGPT's openwebtext prepare.py for example
    ```

## Setup

1.  **Clone nanoGPT**: If you haven't already, clone the official nanoGPT repository into your project structure (e.g., inside your main application directory `THIS_STUDIO/nanoGPT/`).
2.  **Modify `nanoGPT/model.py` (for Activation Recomputation)**:
    * Open `nanoGPT/model.py`.
    * Ensure `import torch.utils.checkpoint` is present at the top.
    * Modify the `forward` method of the `GPT` class to accept a `use_recompute=False` argument and apply `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)` to each transformer block if `use_recompute` is true and `model.training` is true. (Refer to the provided code snippets in previous discussions for the exact modification).
3.  **Place Application Files**:
    * Save the provided `gradio_app.py` and `train_core.py` into your main application directory (e.g., `THIS_STUDIO/`).
4.  **Prepare Data**:
    * For each dataset you intend to use (e.g., `shakespeare_char`), you **must** run its `prepare.py` script first.
    * Example for `shakespeare_char`:
        ```bash
        cd nanoGPT/data/shakespeare_char/
        python prepare.py
        cd ../../.. # Navigate back to your app's root or where gradio_app.py is
        ```
    * The application includes a basic automatic preparation attempt for known small datasets if data files are missing, but **manual preparation is strongly recommended**, especially for larger datasets like `openwebtext`. The automatic preparation uses `sys.executable` to call `python prepare.py`.

## Running the Application

1.  Navigate to the directory containing `gradio_app.py` and `train_core.py` (e.g., `THIS_STUDIO/`).
2.  Run the Gradio application:
    ```bash
    python gradio_app.py
    ```
3.  Open the local URL provided in your terminal (usually `http://127.0.0.1:7860` or `http://localhost:7860`) in your web browser.
4.  Use the interface to configure your model and training run, then click "Launch nanoGPT Training."

## How it Works

* **`gradio_app.py`**:
    * Defines the Gradio user interface with sliders, checkboxes, and dropdowns for all configurations.
    * Manages the collection of metrics from training runs for plotting and analysis.
    * Calls `train_core.py` to execute training runs based on UI settings.
    * Updates plots and logs in real-time (or near real-time) as data is yielded from the training core.
* **`train_core.py`**:
    * Adapts the training loop from nanoGPT's `train.py`.
    * Initializes the `GPT` model (from the cloned `nanoGPT/model.py`) with the configuration specified through Gradio.
    * Handles data loading, optimizer setup, the training loop, gradient accumulation, and loss calculation.
    * Implements toggles for activation recomputation and `nn.DataParallel`.
    * Includes (non-functional) placeholders for Tensor Parallelism and Pipeline Parallelism flags.
    * Yields progress (loss, iteration, timing, GPU memory) back to `gradio_app.py` for display.
    * Includes a basic automatic data preparation step using `subprocess` to call the dataset's `prepare.py` if binary data files are not found (recommended for small datasets only).

## Future Development / Advanced Parallelism Notes

* **Tensor Parallelism & Pipeline Parallelism**: The UI includes flags for these, and `train_core.py` accepts these flags. However, the actual implementation of these advanced parallelism techniques is **not included** and is a significant undertaking. Users wishing to implement these would need to:
    1.  Choose a library (e.g., DeepSpeed, FairScale, PyTorch native TP/PP utilities).
    2.  Modify `nanoGPT/model.py` extensively to shard layers (for TP) or define pipeline stages (for PP) using the chosen library's APIs.
    3.  Heavily modify `train_core.py` to initialize the distributed environment, wrap the model with the library's parallel equivalents, and adapt the training loop to the library's specific methods for forward/backward passes and optimizer steps.
    4.  This would likely involve moving away from `nn.DataParallel` if these more advanced techniques are used.

## Troubleshooting

* **`ModuleNotFoundError: No module named 'nanoGPT.model'` (or similar for `train_core`)**:
    * Ensure your `nanoGPT` directory is correctly placed relative to `train_core.py` and `gradio_app.py` as per the "Project Structure" and that the `sys.path` modifications at the top of the Python scripts are working for your layout. The scripts assume `gradio_app.py` and `train_core.py` are in a directory, and the `nanoGPT` cloned repository is a direct subdirectory of that directory.
* **`Error: Data files not found in .../nanoGPT/data/<dataset_name>`**:
    * You must run the `python prepare.py` script within the specific `nanoGPT/data/<dataset_name>/` directory *before* trying to use that dataset in the Gradio app.
* **`Error: 'python' command not found. Cannot run preparation script automatically`**:
    * This occurs if the automatic data preparation step in `train_core.py` cannot find your Python executable. The script now uses `sys.executable` which should point to the currently running Python, making this less likely. If it persists, ensure `sys.executable` is valid or prepare data manually.
* **Gradio Errors (`AttributeError: Cannot call click outside of a gradio.Blocks context.`)**:
    * Ensure all Gradio UI component definitions and their event handlers (like `.click()`) are strictly within the `with gr.Blocks() as demo:` context in `gradio_app.py`.


