import img_utils as iu
from person_re_identification.osnet import OSNet
import sys

input_dir = sys.argv[1]

img_paths = iu.glob_img_paths(input_dir)

groups, eps = OSNet.cluster_imgs_with_auto_eps(img_paths) 

print(f"eps:{eps}")

for label, paths in groups.items():
    print(f"Cluster :{label}")
    for path in paths:
        print(f"  {path}")