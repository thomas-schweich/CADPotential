from load_voxels import Voxels

files = ['srg_components_r/*.voxels']

voxels = Voxels()

object_to_name = {}

voxels.read_all(files, object_callback_func=lambda o, f: object_to_name.update({o: f}))

suspects = set()
for k, v in voxels.object_to_voxels.iteritems():
    if len(v) < 10:
        suspects.update({k})
        continue
    max_i = int(max(voxel.i for voxel in v))
    max_j = int(max(voxel.j for voxel in v))
    max_k = int(max(voxel.k for voxel in v))
    min_i = int(min(voxel.i for voxel in v))
    min_j = int(min(voxel.j for voxel in v))
    min_k = int(min(voxel.k for voxel in v))
    biggest_vol = int(max_i - min_i) * int(max_j - min_j) * int(max_k - min_k)
    print "%s's largest possible volume is %d. Its actual volume is %d." % (
        object_to_name[k], biggest_vol, k.num_voxels)
    if biggest_vol == 0:
        suspects.update({k})
    else:
        percent_occupied = float(k.num_voxels) / biggest_vol
        if percent_occupied > .90:
            suspects.update({k})

sus_fnames = {sus_fname for suspect, sus_fname in object_to_name.iteritems() if suspect in suspects}

print 'These %d files are suspect:' % len(sus_fnames)
for name in sorted(sus_fnames):
    print '- %s' % name
