from typing import List
import pandas as pd
import numpy as np
import SpatialDE
import NaiveDE

# instance = visium_data_filtered["A1" | "B1" | "C1" | "D1"]
def run_spatialde(counts: np.ndarray, genes: List[str], image_x: np.ndarray, image_y: np.ndarray):
    # make sure binary logits are OK
    counts[counts < 0] = 0

    sample = pd.DataFrame({
        'total_counts': counts.sum(axis=1),
        'image_x': image_x,
        'image_y': image_y,
    })

    counts_df = pd.DataFrame(counts, columns=genes)

    normalized_counts = NaiveDE.stabilize(counts_df.T).T
    residual_expression = NaiveDE.regress_out(sample, normalized_counts.T, 'np.log(total_counts)').T

    sample_resid_exp = residual_expression.sample(n=len(genes), axis=1, random_state=1)
    results = SpatialDE.run(sample[['image_x', 'image_y']], sample_resid_exp)
    
    return results

if __name__ == "__main__":
    import pickle
    import sys

    # with open('../DH/visium/preprocessed_data/visium_data_smooth.pkl', 'rb') as f:
    #     smooth_visium = pickle.load(f)
    
    # simple_spatialde(smooth_visium['B1'][0], smooth_visium['B1'][1]).to_csv("SpatialDE_B1_Smooth.csv")
    # simple_spatialde(smooth_visium['C1'][0], smooth_visium['C1'][1]).to_csv("SpatialDE_C1_Smooth.csv")
    # simple_spatialde(smooth_visium['D1'][0], smooth_visium['D1'][1]).to_csv("SpatialDE_D1_Smooth.csv")

    with open("../DH/visium/preprocessed_data/visium_data_filtered.pkl", 'rb') as f:
        visium_data_filtered = pickle.load(f)

    key = sys.argv[1]

    print("Running on key", key)
    # run_spatialde_on_ground_truth(visium_data_filtered[key]).to_csv("./out/SpatialDE_GroundTruth/Spatial" + key + "Results.csv")
