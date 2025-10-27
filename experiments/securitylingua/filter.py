# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict
import numpy as np
import torch

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Filter compressed prompts based on metrics.")
    parser.add_argument(
        "--load_path",
        help="path to load data",
        default="../../../results/meetingbank/gpt-4-32k_comp/annotation_cs512_meetingbank_train_formated.pt",
    )
    parser.add_argument(
        "--save_path",
        help="path to save filtered data",
        default="../../../results/meetingbank/gpt-4-32k_comp/annotation_kept_cs512_meetingbank_train_formated.pt",
    )
    parser.add_argument(
        "--percentile",
        help="percentile threshold for filtering",
        default=90,
        type=int
    )
    return parser.parse_args()

def filter_by_metric(
    data: DefaultDict[str, List], 
    metric_name: str, 
    percentile: float
) -> Tuple[DefaultDict[str, List], DefaultDict[str, List]]:
    """
    Filter data based on a specific metric and percentile threshold
    
    Args:
        data: Dictionary containing all data points and their metrics
        metric_name: Name of the metric to filter by
        percentile: Percentile threshold for filtering
        
    Returns:
        Tuple of (kept_data, filtered_data)
    """
    metric_list = data[metric_name]
    threshold = np.percentile(metric_list, percentile)
    
    kept = defaultdict(list)
    filtered = defaultdict(list)
    
    # List of all metrics to transfer
    metrics = [
        "labels", "origin", "comp", "retrieval", "comp_rate",
        "variation_rate", "hitting_rate", "matching_rate", "alignment_gap"
    ]
    
    for values in zip(*(data[metric] for metric in metrics)):
        # Create a dictionary of current values
        current = dict(zip(metrics, values))
        
        # Determine which container to use based on the metric threshold
        target = filtered if current[metric_name] >= threshold else kept
        
        # Add values to appropriate container
        for metric, value in current.items():
            target[metric].append(value)
            
    return kept, filtered

def main():
    """Main function to run the filtering process"""
    args = parse_arguments()
    
    # Load data
    res_pt = torch.load(args.load_path, weights_only=False)
    print(f"Initial sample count: {len(res_pt['variation_rate'])}")
    
    # First filtering stage: variation rate
    kept, filtered = filter_by_metric(
        data=res_pt,
        metric_name="variation_rate",
        percentile=args.percentile
    )
    
    # Second filtering stage: alignment gap
    final_kept, additional_filtered = filter_by_metric(
        data=kept,
        metric_name="alignment_gap",
        percentile=args.percentile
    )
    
    # Save filtered results
    torch.save(final_kept, args.save_path)
    
    # Print statistics
    print(f"Samples after first filter: {len(kept['variation_rate'])}")
    print(f"Final kept samples: {len(final_kept['variation_rate'])}")

if __name__ == "__main__":
    main()
