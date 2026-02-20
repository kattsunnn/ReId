import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class OSNetReID:
    extractor = None

    @classmethod
    def find_optimal_eps(cls, feats_norm, k=2):
        neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
        neigh.fit(feats_norm)
        distances, _ = neigh.kneighbors(feats_norm)
        
        k_distances = np.sort(distances[:, k-1])

        acc = np.diff(k_distances, n=2)
        max_idx = np.argmax(np.abs(acc))
        eps = k_distances[max_idx+1]
        # print(eps)
        # plt.figure(figsize=(10, 6))
        # plt.plot(k_distances, marker='o', markersize=2, linestyle='-')
        # plt.axhline(y=eps, color='r', linestyle='--', label='Candidate EPS (e.g. 0.7)') # 目安線
        # plt.title(f"k-distance Graph (k={k}) - DBSCAN Parameter Selection")
        # plt.xlabel("Points sorted by distance")
        # plt.ylabel(f"{k}-th Nearest Neighbor Distance (L2)")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        return eps

    @classmethod
    def _init_extractor(cls):
        if cls.extractor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.extractor = FeatureExtractor(
                model_name='osnet_x1_0',
                device=device
            )

    @classmethod
    def cluster_imgs(cls, img_paths, min_samples=2):
        cls._init_extractor()
        feats = np.asarray(cls.extractor(img_paths))
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        feats_normalized = feats / norms
        eps = cls.find_optimal_eps(feats_normalized)
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = cluster.fit_predict(feats_normalized)
        groups = defaultdict(list)
        for path, label in zip(img_paths, labels):
            groups[label].append(path)
        return groups
    
if __name__ == "__main__":

    import glob
    import os
    import sys
    from img_utils.img_utils import load_img_paths_from_dir

    input_dir = sys.argv[1]
    img_paths = load_img_paths_from_dir(input_dir)

    groups = OSNetReID.cluster_imgs(img_paths)

    for label, paths in groups.items():
        print(f"Cluster :{label}")
        for path in paths:
            print(f"  {path}")
