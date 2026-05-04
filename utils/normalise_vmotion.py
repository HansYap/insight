def normalize_vmotion(stats: dict, norm_config: dict) -> np.ndarray:
    keys = ["mean_magnitude", "std_magnitude", "directionality",
            "coverage_ratio", "dominant_sin", "dominant_cos"]
    return np.array([
        (stats[k] - norm_config[k]["mean"]) / norm_config[k]["std"]
        for k in keys
    ])


"""
mean_magnitude has a long right tail (max=4.39, but 75th percentile is only 0.27). 
Z-score normalization handles symmetric distributions well but can compress the interesting 
high-motion events into a small range at the top.  if the classifier 
later struggles to distinguish high-motion activities switch to 
log-normalizing that specific stat.

"""