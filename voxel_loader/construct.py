from load_voxels import Voxels
import json
import os
import sys

directory = os.path.join(sys.path[0], 'srg_components')
meta_name = 'meta.json'
files = ['main/srg_components_r/*.voxels']
voxels_per_unit = 25.4

with open(os.path.join(directory, meta_name), 'r') as f:
    meta_from_json = json.load(f)
meta_objects = {}  # Dict which will map voxel objects to their metadata which is stored in the json file
voxels = Voxels()
voxels.read_all(files, object_callback_func=lambda obj_, fname: meta_objects.update(
    {obj_: meta_from_json[os.path.basename(fname)[:os.path.basename(fname).index('.')]]}))

# object_callback_func reads the metadata from the json file which is stored under the object's file name (without the
# extension) and stores it in meta_objects under the actual object that it concerns.

for obj in voxels.object_to_voxels.iterkeys():
    vox = voxels.object_to_voxels[obj]
    loc = meta_objects[obj]['loc']
    for v in vox:
        v.i += int(loc[0] * voxels_per_unit)
        v.j += int(loc[1] * voxels_per_unit)
        v.k += int(loc[2] * voxels_per_unit)

voxels.plot_graph()
