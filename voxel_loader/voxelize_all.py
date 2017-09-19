""" Script for voxelizing a sequence of files """

import os


def voxelize_all_with_calculated_resolution(dims, voxels_per_unit, dir_, file_prefix, file_ext, **kwargs):
    def filename(idx):
        return '%s/%s%d%s' % (dir_, file_prefix, idx, file_ext)

    def base_name(idx):
        return '%s%d%s' % (file_prefix, idx, file_ext[:file_ext.index('.')])

    def voxelize(idx):
        voxelize_file(filename(idx), long_axis_resolution=int(max(dims[base_name(idx)]['dim']) * voxels_per_unit),
                      **kwargs)

    i = 0
    if os.path.isfile(filename(i)):
        voxelize(i)
    i = 1
    while os.path.isfile(filename(i)):
        voxelize(i)
        i += 1


def voxelize_all(dir_, file_prefix, file_ext, **kwargs):
    """ Uses Stanford Haptics Lab's 'Voxelizer' to voxelize all files in the given directory with the given prefix and 
    extension
     
     A sequential numeric identifier should follow the prefix of each file.
     The working directory must contain voxelizer.exe as well as voxelizer.ini
     The extension should include a dot.
     """

    def filename(idx):
        return '%s/%s%d%s' % (dir_, file_prefix, idx, file_ext)

    i = 0
    if os.path.isfile(filename(i)):
        voxelize_file(filename(i), **kwargs)
    i = 1
    while os.path.isfile(filename(i)):
        voxelize_file(filename(i), **kwargs)
        i += 1


def voxelize_file(filename, seed_triangle=10, long_axis_resolution=80, compute_distance_field=0):
    """ Runs the voxelizer on the given file with the given settings """
    print 'Voxelizing %s' % filename
    with open('voxelizer.ini', 'w') as f:
        f.writelines(['LONG_AXIS_RESOLUTION %d\n' % long_axis_resolution,
                      'COMPUTE_DISTANCE_FIELD %d\n' % compute_distance_field,
                      'OBJECT_TO_VOXELIZE %s\n' % filename,
                      'SEED_TRIANGLE %d\n' % seed_triangle,
                      'VOXELIZE_IMMEDIATELY 1'])
    os.system('voxelizer.exe')


if __name__ == '__main__':
    import json
    directory = 'srg_components'
    prefix = '_ncl1_'
    suffix = '.obj'
    voxels_per_unit_ = 25.4

    with open(os.path.join(directory, 'meta.json')) as f:
        dims_ = json.load(f)
    voxelize_all_with_calculated_resolution(dims_, voxels_per_unit_, 'srg_components', '_ncl1_', '.obj')
