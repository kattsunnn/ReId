import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx


class OSNet:
    extractor = None


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
        return eps

    @classmethod
    def clustering_by_mst(cls, img_paths, max_size, min_size=3):
        cls._init_extractor()
        feats = np.asarray(cls.extractor(img_paths))
        labels, components = cls.mst_constrained_clustering(feats, max_size, min_size)
        groups = defaultdict(list)
        for path, label in zip(img_paths, labels):
            groups[label].append(path)
        return groups

    @staticmethod
    def mst_constrained_clustering(feats, max_size, min_size=3):
        # 1. 特徴量の正規化 (L2 Normalization)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        feats_normalized = feats / norms
        # 2. 距離行列の計算と最小全域木(MST)の構築
        dist_matrix = squareform(pdist(feats_normalized))
        mst = minimum_spanning_tree(dist_matrix).toarray()
        # NetworkXグラフへの変換
        n_samples = feats.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))
        # エッジ情報の抽出
        edges = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if mst[i, j] > 0:
                    G.add_edge(i, j, weight=mst[i, j])
                    edges.append((i, j, mst[i, j]))
        # 3. エッジを距離（重み）の降順にソート（長いエッジ＝密度の谷 から切断するため）
        edges.sort(key=lambda x: x[2], reverse=True)
        # 4. 切断プロセス
        for u, v, weight in edges:
            components = list(nx.connected_components(G)) # 現在の全クラスター（連結成分）を取得
            # 終了条件: 全てのグループが max_size 以下になったら処理を終える
            if all(len(comp) <= max_size for comp in components):
                break
            comp_containing_edge = next(c for c in components if u in c and v in c) # 対象のエッジ (u, v) が含まれるグループを取得
            # グループの要素数が閾値 (max_size) を超えている場合のみエッジを切断
            if len(comp_containing_edge) > max_size:
                G.remove_edge(u, v)
        # 5. ラベル付け
        final_components = list(nx.connected_components(G))
        labels = np.full(n_samples, -1, dtype=int) # 初期値 -1 (ノイズ)
    
        cluster_id = 0
        for comp in final_components:
            # 規定値 (min_size) 以上のグループのみ正式なクラスターとする
            if len(comp) >= min_size:
                for node in comp:
                    labels[node] = cluster_id
                cluster_id += 1
            
        return labels, final_components
    