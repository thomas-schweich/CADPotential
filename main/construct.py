from load_voxels import Voxels
import json
import os
import sys
import numpy as np
import itertools

directory = os.path.join(sys.path[0], 'srg_components')
meta_name = 'meta.json'
files = ['srg_components_r/*.voxels']
voxels_per_unit = 25.4

with open(os.path.join(directory, meta_name), 'r') as f:
    meta_from_json = json.load(f)
meta_objects = {}  # Dict which will map voxel objects to their metadata which is stored in the json file
voxels = Voxels()
voxels.read_all(files, object_callback_func=lambda obj_, fname: meta_objects.update(
    {obj_: meta_from_json[os.path.basename(fname)[:os.path.basename(fname).index('.')]]}))

# object_callback_func reads the metadata from the json file which is stored under the object's file name (without the
# extension) and stores it in meta_objects under the actual object that it concerns.

out = np.zeros((len(voxels.get_all_voxels()), 3), dtype=np.float32)
start = 0
for obj in voxels.object_to_voxels.iterkeys():
    vox = voxels.object_to_voxels[obj]
    loc = meta_objects[obj]['loc']
    zero_coord = obj.zero_coordinate
    units_per_voxel = 1. / voxels_per_unit
    object_length = len(vox)
    # all_i = np.fromiter((v.i for v in vox), dtype=np.int32, count=object_length)
    # all_j = np.fromiter((v.j for v in vox), dtype=np.int32, count=object_length)
    # all_k = np.fromiter((v.k for v in vox), dtype=np.int32, count=object_length)
    # avg_x = (np.max(all_i) + np.min(all_i)) / 2.
    # avg_y = (np.max(all_j) + np.min(all_j)) / 2.
    # avg_z = (np.max(all_k) + np.min(all_k)) / 2.
    out[start:start + object_length] = np.fromiter(
        itertools.chain.from_iterable(((float(v.i) - vox[0].i) * units_per_voxel + loc[0],
                                       (float(v.j) - vox[0].j) * units_per_voxel + loc[1],
                                       (float(v.k) - vox[0].k) * units_per_voxel + loc[2]) for v in vox),
        dtype=np.float32, count=object_length * 3).reshape(object_length, 3)
    start += object_length

Voxels._plot_graph(out)
