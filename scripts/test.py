from networkx.algorithms import shortest_paths
import img_utils as iu
from person_re_identification.osnet import OSNet
import sys

input_dir = sys.argv[1]

img_paths = iu.glob_img_paths(input_dir)

# groups, eps = OSNet.cluster_imgs_with_auto_eps(img_paths) 
# groups = OSNet.clustering_by_mst(img_paths, max_size=2, min_size=2)
groups = OSNet.dbscan_by_k_reciprocal_jaccard(img_paths, eps=0.4, min_samples=2)


for label, paths in groups.items():
    print(f"Cluster :{label}")
    for path in paths:
        print(f"  {path}")