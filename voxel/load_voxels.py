""" Module, written by Thomas Schweich at the GEE Lab at Washington University in St. Louis, containing a Python 
implementation of a .voxels file reader for files generated from Dan Morris at Stanford Haptics Lab's Voxelizer program, 
as well as various related utility methods.

Dan Morris' Voxelizer can be found here: http://dmorris.net/projects/voxelizer/

Running the script from the command line provides options for loading a single .voxels file or multiple files, indicated
by a file glob (using * as a wildcard), into a single `Voxels` Python object (declared below). 

The script retains all information in the file(s) except the file header(s).

The script can be run interactively, in which case the Voxels object resulting from the loaded file(s) can be accessed 
through a variable called 'voxels', such as in the following example:

$ python -i load_voxels.py myvoxels_*.voxels
>>> voxels.to_set()
{(0, 0, 0), (0, 1, 0) ... }
>>> first_voxel = voxels.get_all_voxels()[0]
>>> first_voxel
<__main__.VoxelfileVoxel object at 0x0000000006A3FEC8>
>>> first_voxel.i
0
>>> first_voxel.has_normal
1
>>> voxels.plot_graph()
# Graph appears
# etc... 

where 'myvoxels_1.voxels', 'myvoxels_2.voxels'... are the names of your .voxels files.
The set resulting from the call to voxels.to_set() would contain the positions of each voxel from the file. 
The VoxelfileVoxel object resulting from the call to voxels.voxels[0] would be the first voxel in the file. 
ipython could also be used in place of python. The dollar-sign ($) is simply meant to indicate that the proceeding 
line is entered into a console (whose working directory is the one containing the script).

Use `$ python load_voxels.py -h` for help. The module may also, of course, simply be imported by other scripts.

The Voxels object also contains methods for dealing with the concatenation of multiple .voxels files into a single
object. This is due to the fact that the script was originally written for use with the voxelization of a complex CAD
file which was divided into many small components. As a result, the script would theoretically work with multiple
objects, as well.

The script has been tested to run relatively fluently with multi-million-voxel files, including plotting. The test was 
run on a laptop with a 2 core, 4 thread Intel i5 processor with 8GB RAM. Performance likely scales quite well with 
improved hardware.
"""

import numpy as np
import sys
import math
import argparse
import itertools
import glob
from collections import namedtuple
from ctypes import *

__author__ = 'Thomas Schweich, GEE Lab at Washington University in St. Louis'


class VoxelfileFileHeader(Structure):
    """ The file header in a .voxels file """
    _pack_ = 1
    _fields_ = [('header_size', c_int),
                ('num_objects', c_int),
                ('object_header_size', c_int),
                ('voxel_struct_size', c_int)]


# noinspection PyTypeChecker
class VoxelfileObjectHeader(Structure):
    """ An object header in a .voxels file
     
     Note that the Voxels object stores the object headers as ImmutableVoxelfileObjectHeader objects, a namedtuple
      defined in class_as_namedtuple below, such that they can be used as dictionary keys. 
     """
    MAX_PATH = 260  # Windows max path variable
    _pack_ = 1
    _fields_ = [('num_voxels', c_int),
                ('voxel_resolution', c_int * 3),
                ('voxel_size', c_float * 3),
                ('model_scale_factor', c_float),
                ('model_offset', c_float * 3),
                ('zero_coordinate', c_float * 3),
                ('has_texture', c_ubyte),
                ('texture_filename', c_char * MAX_PATH)]

    @staticmethod
    def class_as_namedtuple():
        """ Returns a namedtuple subclass including all fields defined in _fields_ """
        return namedtuple('ImmutableVoxelfileObjectHeader',
                          'num_voxels voxel_resolution voxel_size model_scale_factor model_offset zero_coordinate '
                          'has_texture texture_filename')

    def fields_as_immutable_tuple(self):
        """ Returns all fields of this instance as a tuple, in which every field is made immutable """
        return (self.num_voxels, tuple(self.voxel_resolution), tuple(self.voxel_size), self.model_scale_factor,
                tuple(self.model_offset), tuple(self.zero_coordinate), self.has_texture, self.texture_filename)

    def to_namedtuple(self):
        """ Returns an ImmutableVoxelfileObjectHeader generated from this instance """
        return VoxelfileObjectHeader.class_as_namedtuple()(*self.fields_as_immutable_tuple())


# noinspection PyTypeChecker
class VoxelfileVoxel(Structure):
    """ A voxel read from a .voxels file """
    MAX_NUM_MODIFIERS = 5
    _pack_ = 1
    _fields_ = [('i', c_short),
                ('j', c_short),
                ('k', c_short),
                ('has_texture', c_ubyte),
                ('u', c_float),
                ('v', c_float),
                ('has_normal', c_ubyte),
                ('normal', c_float * 3),
                ('has_distance', c_ubyte),
                ('distance_to_surface', c_float),
                ('distance_gradient', c_float * 3),
                ('num_modifiers', c_ubyte),
                ('distance_to_modifier', c_float * MAX_NUM_MODIFIERS),
                ('modifier_gradient', c_float * MAX_NUM_MODIFIERS * 3),
                ('is_on_border', c_ubyte)]


class Voxels(object):
    """ Object containing a groups of VoxelfileVoxels mapped to their object headers 
    
    Includes methods for reading voxels from disk, plotting voxels as 3d scatter plots, scaling, and conversion to 
    numpy arrays of positions.
    """
    ASCIIVOXEL_NOVOXEL = '0'
    ASCIIVOXEL_INTERNAL_VOXEL = '1'
    ASCIIVOXEL_TEXTURED_VOXEL = '2'

    def __init__(self):
        # Dictionary mapping all ImmutableVoxelfileObjectHeaders to lists of their respective VoxelfileVoxels
        self.object_to_voxels = {}

    @staticmethod
    def _update_progress(progress, length):
        """ Updates a command-line progress bar """
        percent = math.ceil(float(progress) / length * 100)
        num_pounds = int(.2 * percent)
        sys.stdout.write('\r[{0}{1}] {2}%'.format('#' * num_pounds,
                                                  ' ' * (20 - num_pounds),
                                                  int(percent)))

    @staticmethod
    def _plot_graph(array):
        """ Shows a graph with QTGraph (strongly preferred to Matplotlib for speed) 
        
        Use scroll wheel to zoom in and out, click and drag to orbit.
        """
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
        except ImportError:
            raise NotImplementedError('Plotting graphs requires pyqtgraph, which in turn requires pyopengl.\n'
                                      'To install these modules, use:\n\n'
                                      '$ pip install PyOpenGL PyOpenGL_accelerate\n\n'
                                      'followed by:\n\n'
                                      '$ pip install pyqtgraph\n\n'
                                      'Alternatively, you can use Matplotlib and plot_graph_mpl() for (much slower) '
                                      'plotting.')
        number_points = len(array)
        print 'Plotting %d voxels...' % number_points
        app = pg.mkQApp()
        view = gl.GLViewWidget()
        view.show()

        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view.addItem(xgrid)
        view.addItem(ygrid)
        view.addItem(zgrid)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        xgrid.scale(10, 10, 10)
        ygrid.scale(10, 10, 10)
        zgrid.scale(10, 10, 10)

        red_grad = np.abs(np.sin(np.linspace(0., np.pi, dtype=np.float32, num=number_points)))
        green_grad = np.abs(np.sin(np.linspace(np.pi / 3, np.pi + np.pi / 3, dtype=np.float32, num=number_points)))
        blue_grad = np.abs(
            np.sin(np.linspace(2 * np.pi / 3, np.pi + 2 * np.pi / 3, dtype=np.float32, num=number_points)))
        color_gradient = np.ones((number_points, 4), dtype=np.float32)
        color_gradient[:, 0] = red_grad
        color_gradient[:, 1] = green_grad
        color_gradient[:, 2] = blue_grad

        scatter = gl.GLScatterPlotItem(pos=array, color=color_gradient)
        scatter.setGLOptions('opaque')
        view.addItem(scatter)
        app.exec_()

    def add_voxel(self, object_header, voxel):
        """ Adds the voxel to object_to_voxels under the given header, creating an entry if it does not exist 
        
        The header must be converted to an immutable object (via to_namedtuple).
        """
        try:
            self.object_to_voxels[object_header] += [voxel]
        except KeyError:
            self.object_to_voxels.update({object_header: [voxel]})

    def get_all_voxels(self):
        """ Returns an iterable of all voxels in no particular order """
        return list(itertools.chain.from_iterable(self.object_to_voxels.itervalues()))

    def read_all(self, file_names, object_callback_func=None):
        """ Reads all files in the given iterable of file names/globs

        If object_callback_func is specified, it is passed to each call to from_file.
        """
        for g in file_names:
            for f in glob.glob(g):
                self.from_file(f, object_callback_func)

    def from_file(self, filename, object_callback_func=None):
        """ Read voxels from a .voxels file 
        
        .voxels format is as described in http://dmorris.net/projects/voxelizer/voxel_file_format.h
         
         If object_callback_func is specified, it must be a function of two arguments: the first will be the
         ImmutableVoxelfileObjectHeader object which was read, and the second will be the filename that was used to
         access the object. If, in theory, there were multiple objects in the .voxels file, the function would be
         called once per object. The use case is to associate some extra data with the objects on a per-file basis.
         Note that the object passed is _immutable_.
        """
        with open(filename, 'rb') as f:
            voxelfile_file_header = VoxelfileFileHeader()
            f.readinto(voxelfile_file_header)
            print 'Read file header for %s' % filename
            for _ in range(0, voxelfile_file_header.num_objects):
                voxelfile_object_header = VoxelfileObjectHeader()
                f.readinto(voxelfile_object_header)
                voxelfile_object_header = voxelfile_object_header.to_namedtuple()
                resolution_product = int(np.prod(voxelfile_object_header.voxel_resolution))
                for i in range(0, resolution_product):
                    header = f.read(1)
                    if header != Voxels.ASCIIVOXEL_NOVOXEL:
                        voxelfile_voxel = VoxelfileVoxel()
                        f.readinto(voxelfile_voxel)
                        self.add_voxel(voxelfile_object_header, voxelfile_voxel)
                print 'Found %d voxels in object' % len(self.object_to_voxels[voxelfile_object_header])
                if object_callback_func:
                    object_callback_func(voxelfile_object_header, filename)

    def scaled_voxel_tup_generator(self):
        """ Returns a generator which creates 3-tuples of the _scaled_ i, j, k positions of each voxel """
        return (((v.i + header.model_offset[0]) * header.model_scale_factor,
                 (v.j + header.model_offset[1]) * header.model_scale_factor,
                 (v.k + header.model_offset[2]) * header.model_scale_factor)
                for header, voxel_group in self.object_to_voxels.iteritems() for v in
                voxel_group)

    def voxel_tup_generator(self):
        """ Returns a generator which creates 3-tuples of the _unscaled_ i, j, k positions of each voxel """
        return ((v.i, v.j, v.k) for voxel_group in self.object_to_voxels.itervalues() for v in voxel_group)

    def to_3col_array(self, scaled=False):
        """ Returns a 3 column ndarray containing the i, j, k positions of each voxel
        
        If kwarg 'scaled' is set to True, the resulting positions are scaled according to each VoxelfileVoxel's 
        model_scale_factor and model_offset variables such that the voxels should align with the mesh from which
        they were originally created. The returned values are then of type np.float32.
        
        Array creation is kept entirely within vectorized Numpy operations for optimal performance.
        """
        total_num = len(self.get_all_voxels())
        if scaled:
            return np.fromiter(itertools.chain.from_iterable(self.scaled_voxel_tup_generator()), dtype=np.float32,
                               count=total_num * 3).reshape((total_num, 3))
        else:
            return np.fromiter(itertools.chain.from_iterable(self.voxel_tup_generator()),
                               dtype=np.int32, count=total_num * 3).reshape((total_num, 3))

    def to_set(self, scaled=False):
        """ Returns a set of 3-tuples containing the i, j, k positions of each voxel """
        if scaled:
            return {s for s in self.scaled_voxel_tup_generator()}
        else:
            return {u for u in self.voxel_tup_generator()}

    def geometric_center(self, obj_):
        """ Returns the geometric center (as a 3-tuple of floats) of the voxel object """
        vox = self.object_to_voxels[obj_]
        border_voxels = np.array([[v.i, v.j, v.k] for v in vox if v.is_on_border]).astype(np.float32)
        return ((np.max(border_voxels[:, 0]) + np.min(border_voxels[:, 0])) / 2,
                (np.max(border_voxels[:, 1]) + np.min(border_voxels[:, 1])) / 2,
                (np.max(border_voxels[:, 2]) + np.min(border_voxels[:, 2])) / 2)

    def plot_graph_mpl(self, show=True):
        """ Shows a graph of the voxels with Matplotlib (_significantly_ slower than the QTGraph implementation) 
        
        Should be used for small files only.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise NotImplementedError('Matplotlib is required for use of this function.')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vox_set = self.to_set()
        set_len = len(vox_set)
        print 'Generating graph of %d unique points' % set_len
        for i, v in enumerate(vox_set):
            ax.scatter(v[0], v[1], v[2], marker=',')
        print ''
        if show:
            plt.show(ax)

    def plot_graph(self, scaled=False):
        self._plot_graph(self.to_3col_array(scaled=scaled))


def load_from_command_line():
    """ Loads the file(s) given by the command line arguments, plotting if requested """
    parser = argparse.ArgumentParser(description='Loads voxels from a .voxels file. '
                                                 'Specify either a single file or a glob of files.')
    parser.add_argument('file', help='Name, group, or glob of file(s) to read', nargs='+')
    parser.add_argument('-g', '--graph', help='Plot a graph after loading', action='store_true')
    parser.add_argument('-s', '--scaled', help='Scale the graph if graphed', action='store_true')
    args = parser.parse_args()
    vox = Voxels()
    vox.read_all(args.file)
    if args.graph:
        vox.plot_graph(scaled=args.scaled)
    return vox


if __name__ == "__main__":
    voxels = load_from_command_line()
