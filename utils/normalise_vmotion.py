def normalize_vmotion(stats: dict, norm_config: dict) -> dict:
    normalized = {}
    for key, value in stats.items():
        mean = norm_config[key]["mean"]
        std  = norm_config[key]["std"]
        normalized[key] = (value - mean) / std if std > 0 else 0.0
    return normalized


"""
mean_magnitude has a long right tail (max=4.39, but 75th percentile is only 0.27). 
Z-score normalization handles symmetric distributions well but can compress the interesting 
high-motion events into a small range at the top.  if the classifier 
later struggles to distinguish high-motion activities switch to 
log-normalizing that specific stat.

"""