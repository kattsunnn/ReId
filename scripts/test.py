from networkx.algorithms import shortest_paths
import img_utils as iu
from person_re_identification.osnet import OSNet
import sys

input_dir = sys.argv[1]

img_paths = iu.glob_img_paths(input_dir)

# groups, eps = OSNet.cluster_imgs_with_auto_eps(img_paths) 
# groups = OSNet.clustering_by_mst(img_paths, max_size=2, min_size=2)
# groups = OSNet.dbscan_by_k_reciprocal_jaccard(img_paths, eps=0.4, min_samples=3, k=4)
# for label, paths in groups.items():
#     print(f"Cluster :{label}")
#     for path in paths:
#         print(f"  {path}")

pairs = OSNet.find_top_n_similar_pairs(img_paths, top_n=-1, similarity_threshold=0.75)
print("Top similar pairs:")
for idx, (img_a, img_b, score) in enumerate(pairs):
    print(f"  Pair {idx+1}: {img_a} and {img_b} (similarity: {score:.4f})")

