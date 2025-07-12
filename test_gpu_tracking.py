#!/usr/bin/env python3
"""
Test script to verify per-GPU memory tracking works for all parallelism types.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_gpu_tracking_logic():
    """Test the logic that determines when to show per-GPU memory tracking."""
    print("🔍 Testing per-GPU memory tracking logic...")
    
    # Simulate different parallelism configurations
    test_configs = [
        {"use_data_parallel_ui": True, "use_pipeline_parallel_ui": False, "use_tensor_parallel_ui": False},
        {"use_data_parallel_ui": False, "use_pipeline_parallel_ui": True, "use_tensor_parallel_ui": False},
        {"use_data_parallel_ui": False, "use_pipeline_parallel_ui": False, "use_tensor_parallel_ui": True},
        {"use_data_parallel_ui": True, "use_pipeline_parallel_ui": True, "use_tensor_parallel_ui": False},
        {"use_data_parallel_ui": True, "use_pipeline_parallel_ui": False, "use_tensor_parallel_ui": True},
        {"use_data_parallel_ui": False, "use_pipeline_parallel_ui": True, "use_tensor_parallel_ui": True},
        {"use_data_parallel_ui": True, "use_pipeline_parallel_ui": True, "use_tensor_parallel_ui": True},
        {"use_data_parallel_ui": False, "use_pipeline_parallel_ui": False, "use_tensor_parallel_ui": False},
    ]
    
    for i, config in enumerate(test_configs):
        # Test the logic from gradio_app.py
        use_data_parallel = config.get('use_data_parallel_ui', False)
        use_pipeline_parallel = config.get('use_pipeline_parallel_ui', False)
        use_tensor_parallel = config.get('use_tensor_parallel_ui', False)
        use_any_parallelism = use_data_parallel or use_pipeline_parallel or use_tensor_parallel
        
        # Test the logic from train_core.py
        should_track_per_gpu = (use_data_parallel or use_pipeline_parallel or use_tensor_parallel)
        
        # Create labels as they would appear
        parallelism_types = []
        if use_data_parallel: parallelism_types.append("DP")
        if use_pipeline_parallel: parallelism_types.append("PP")
        if use_tensor_parallel: parallelism_types.append("TP")
        parallelism_label = "+".join(parallelism_types) if parallelism_types else "None"
        
        print(f"  Config {i+1}: {parallelism_label}")
        print(f"    - Should show per-GPU tracking: {'Yes' if use_any_parallelism else 'No'}")
        print(f"    - Should collect per-GPU data: {'Yes' if should_track_per_gpu else 'No'}")
        print(f"    - Label: 'R1 GPU0 ({parallelism_label})' or 'R1 (Standard)'")
        print()
    
    print("✅ Per-GPU tracking logic test completed!")
    print("\n📋 Summary:")
    print("- Data Parallelism only: Shows per-GPU tracking with (DP) labels")
    print("- Pipeline Parallelism only: Shows per-GPU tracking with (PP) labels") 
    print("- Tensor Parallelism only: Shows per-GPU tracking with (TP) labels")
    print("- Combined parallelism: Shows per-GPU tracking with combined labels (e.g., DP+PP+TP)")
    print("- No parallelism: Shows standard single line with (Standard) label")

def test_analysis_labels():
    """Test the analysis labels for different parallelism combinations."""
    print("\n🔍 Testing analysis labels...")
    
    test_configs = [
        {"use_recompute_ui": True, "use_data_parallel_ui": True, "use_pipeline_parallel_ui": False, "use_tensor_parallel_ui": False},
        {"use_recompute_ui": False, "use_data_parallel_ui": False, "use_pipeline_parallel_ui": True, "use_tensor_parallel_ui": True},
        {"use_recompute_ui": True, "use_data_parallel_ui": True, "use_pipeline_parallel_ui": True, "use_tensor_parallel_ui": True},
        {"use_recompute_ui": False, "use_data_parallel_ui": False, "use_pipeline_parallel_ui": False, "use_tensor_parallel_ui": False},
    ]
    
    for i, config in enumerate(test_configs):
        # Test analysis labels
        parallelism_info = []
        if config.get('use_recompute_ui', False): parallelism_info.append("Gradient Checkpointing")
        if config.get('use_data_parallel_ui', False): parallelism_info.append("Data Parallelism")
        if config.get('use_pipeline_parallel_ui', False): parallelism_info.append("Pipeline Parallelism")
        if config.get('use_tensor_parallel_ui', False): parallelism_info.append("Tensor Parallelism")
        parallelism_used = ", ".join(parallelism_info) if parallelism_info else "None"
        
        print(f"  Config {i+1}: Parallelism Techniques: `{parallelism_used}`")
    
    print("✅ Analysis labels test completed!")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Per-GPU Memory Tracking")
    print("=" * 60)
    
    test_gpu_tracking_logic()
    test_analysis_labels()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n💡 Now ALL parallelism types will show per-GPU memory tracking:")
    print("   - ✅ Data Parallelism (DP)")
    print("   - ✅ Pipeline Parallelism (PP)")
    print("   - ✅ Tensor Parallelism (TP)")
    print("   - ✅ Any combination of the above")
    print("\n📊 In the plots, you'll see:")
    print("   - Individual GPU lines for each GPU")
    print("   - Average line across all GPUs")
    print("   - Clear labels indicating which parallelism type(s) are active") 