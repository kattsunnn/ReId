import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class OSNet:
    extractor = None

    @classmethod
    def find_optimal_eps(cls, feats_norm, k=2):
        neigh = NearestNeighbors(n_neighbors=k+1, metric='euclidean') # n_neighbors:自分自身を含む何番目まで遠い距離を探すか（1はじまり） 自分自身を含むためk+1
        neigh.fit(feats_norm)
        distances, _ = neigh.kneighbors(feats_norm)
        k_distances = np.sort(distances[:, k]) # 何番目の距離を取り出すか．0はじまりのためk

        acc = np.diff(k_distances, n=2)
        # max_idx = np.argmax(np.abs(acc))
        max_idx = np.argmax(acc)
        eps = k_distances[max_idx+1]
        # 谷を直接epsに設定すると，谷部分に相当する点がクラスタに含まれなくなってしまうため，小さな値を足し合わせる
        eps = eps + 1e-5
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances, marker='o', markersize=2, linestyle='-')
        plt.axhline(y=eps, color='r', linestyle='--', label='Candidate EPS (e.g. 0.7)') # 目安線
        plt.title(f"k-distance Graph (k={k}) - DBSCAN Parameter Selection")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{k}-th Nearest Neighbor Distance (L2)")
        plt.grid(True)
        plt.legend()
        plt.show()
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
    def cluster_imgs_with_auto_eps(cls, img_paths, min_samples=2):
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
        return groups, eps
    
