"""calculate_potentials.py

Calculate the gravitational potential at each point in any given volume matrix due to any given mass matrix using OpenCL

The module can be run interactively as in the following example:

$ python -i calculate_potentials.py
>>> VOLUME_MASS_DENSITY = 0.5                           # The mass density of the potential matrix
>>> MASS_MASS_DENSITY = 0.5                             # The mass density of the mass matrix
>>> volume = make_cube(100)                             # Create the potential matrix
>>> mass = make_cube(10000)                             # Create the mass matrix
>>> cl_objects = generate_opencl_objects(volume, mass)  # Create the OpenCL interface
>>> result = calculate_potentials_opencl(cl_objects, VOLUME_MASS_DENSITY, MASS_MASS_DENSITY)
>>> plot_results(volume, mass, result)                  # Look at the results
>>> idx = 0                                             # An example index we wish to access
>>> print 'The gravitational potential at (%f, %f, %f) is %f' % (volume[idx][0], volume[idx][1], volume[idx][2], 
...                                                              result[idx])
>>> import numpy as np                                  # Save the two matrices for future use:
>>> np.savetxt('potentials.csv', result, delimiter=',')
>>> np.savetxt('volume.csv', result, delimiter=',')
"""

from __future__ import division
import pyopencl as cl
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import random

__author__ = 'Thomas Schweich, GEE Lab at Washington University in St. Louis'


def generate_opencl_objects(volume, mass):
    """ Function which returns a dict of all items needed to perform the OpenCL computation """
    ## Obtain a list of OpenCL-supporting platforms
    platform = cl.get_platforms()

    ## Select the last one on the list (this may be changed) to use as the device and create a context using that device
    my_gpu_devices = platform[-1].get_devices(cl.device_type.GPU)
    ctx = cl.Context([my_gpu_devices[0]])

    ## Create an OpenCL command queue using the context
    queue = cl.CommandQueue(ctx)

    ## Create buffers to send to device
    # OpenCL float3 types actually use 2^2 = 4 32 bit blocks of memory. Thus, to pass OpenCL an array of float3s, we
    # will simply populate the fourth column with zeros.

    # Create new arrays with 4 columns instead of 3, populated with zeros
    volume_vecs = np.zeros((len(volume), 4), dtype=np.float32)
    mass_vecs = np.zeros((len(mass), 4), dtype=np.float32)

    # Copy the old arrays into the new ones, leaving the fourth column just as zeros
    volume_vecs[:, :-1] = volume
    mass_vecs[:, :-1] = mass

    mf = cl.mem_flags

    # Copy the vector arrays into input buffers on the GPU
    volume_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=volume_vecs)
    mass_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mass_vecs)

    # Create output buffer
    potentials = np.empty(len(volume), dtype=np.float32)
    potentials_out = cl.Buffer(ctx, mf.WRITE_ONLY, potentials.nbytes)

    ## Read and compile the opencl kernel
    with open('potential_kernel.cl') as kernel:
        prg = cl.Program(ctx, kernel.read()).build()

    return {'queue': queue, 'program': prg, 'volume_in': volume_in, 'mass_in': mass_in, 'mass_len': len(mass),
            'potentials_out': potentials_out, 'out_ary': potentials}


def calculate_potentials_opencl(cl_objects, volume_material_mass, mass_material_mass):
    """ Calculates potentials using OpenCL 

    cl_objects:
    - Uses cl_objects['volume_in'] as the input volume, and cl_objects['mass_in'] as the mass object.
    - The PyOpenCL queue should be given as cl_objects['queue']
    - The output buffer should be given as cl_objects['potentials_out']
    - The empty output array should be given as cl_objects['out_ary']
    - The built kernel should be given as cl_objects['program']
        - The program should be compiled from a kernel named 'compute_potential' which takes the following arguments:
            - __global const float3 *volume_matrix_in:    The matrix of positions representing the volume
            - __global const float3 *mass_matrix_in:      The matrix of positions representing the mass
            - const int size_mass_matrix:                 The number of positions in the mass matrix
            - const float volume_material_mass:           The mass density of the material the volume matrix is made of
            - const float mass_material_mass:             The mass density of the material the mass matrix is made of
            - __global float *potential_matrix_out:       The output matrix
    """
    cl_objects['program'].compute_potential(cl_objects['queue'], cl_objects['out_ary'].shape, None,
                                            cl_objects['volume_in'], cl_objects['mass_in'],
                                            np.int32(cl_objects['mass_len']),
                                            np.float32(volume_material_mass),
                                            np.float32(mass_material_mass),
                                            cl_objects['potentials_out'])
    cl.enqueue_copy(cl_objects['queue'], cl_objects['out_ary'], cl_objects['potentials_out'])
    cl_objects['queue'].finish()
    return cl_objects['out_ary']


def make_cube(side_length):
    """ Generates a 2d array of 3-tuples containing the positions of each voxel in a cube of the given side length

    The generated cube has an origin at (0, 0, 0). All coordinates are 32 bit floats. In retrospect this could've used
    numpy.mgrid.
    """
    ## Create an empty array containing the cube of the side length number of points:
    cube = np.zeros((side_length ** 3, 3), dtype=np.float32)

    ## Create a sequence from 0 to the side length:
    sequence = np.arange(side_length, dtype=np.float32)

    ## Set the x-column of the output to the sequence repeated to fill all points:
    # Because `sequence` is a sequence containing `side_length` number of points, tiling `sequence` `side_length`^2
    # number of times yields `side_length` * `side_length`^2 = `side_length`^3 points
    cube[:, 0] = np.tile(sequence, side_length ** 2)

    ## Set the y-column of the output to the array of all elements in the sequence repeated `side_length`^2 times
    # The math works out here the same way as above
    cube[:, 1] = np.repeat(sequence, side_length ** 2)

    ## Create a sequence containing each number from 0 to `side_length` repeated `side_length` times
    # The length of the sequence will thus turn out to be `side_length`^2
    repeated_sequence = np.repeat(sequence, side_length)

    ## Set the z-column to the repeated sequence tiled `side_length` number of times
    # This fills out all possible coordinate values in the volume of the cube
    # Since `repeated_sequence` is `side_length`^2 points long, tiling `repeated_sequence` `side_length` times yields
    # `side_length`^2 * `side_length` = `side_length`^3 points
    cube[:, 2] = np.tile(repeated_sequence, side_length)

    ## Return the resulting cube of coordinates
    return cube


def make_cylinder(radius, length):
    """ Generates a 2d array of 3-tuples containing the positions of each voxel in a cylinder """
    ## Create a square grid large enough to hold the circle face of the cylinder
    # Define the coordinate space of the square
    coords = np.arange(radius * 2, dtype=np.float32)
    # Get all coordinates in the square -- all x values in one array, all y values in another
    xx, yy = np.meshgrid(coords, coords)
    xx = np.ravel(xx)
    yy = np.ravel(yy)

    ## Move the grid's center to the origin
    xx -= radius
    yy -= radius

    ## Determine the indices which fall into the radius of the circle
    in_circle = np.sqrt(xx ** 2 + yy ** 2) < radius

    ## Filter out the points on the grid which don't fall into the radius of the circle
    circle_x = xx[in_circle]
    circle_y = yy[in_circle]

    ## Tile the coordinates `length` number of times
    cylinder_x = np.tile(circle_x, length)
    cylinder_y = np.tile(circle_y, length)

    ## Create the z-coordinates for each tiled circle
    # Determine the number of coordinates in an individual circle
    num_circle_coords = len(circle_x)
    # Create the z coordinates by repeating the numbers 0 through `length` `num_circle_coords` number of times
    cylinder_z = np.repeat(np.arange(length, dtype=np.float32), num_circle_coords)

    # Combine the columns to form the resulting cylinder
    return np.column_stack((cylinder_x, cylinder_y, cylinder_z))


def create_pg_view():
    """ Creates a PyQTGraph view with visible coordinate axes"""
    view = gl.GLViewWidget()
    view.show()
    # xgrid = gl.GLGridItem(color=(1, 1, 1, 1))
    # ygrid = gl.GLGridItem(color=(1, 1, 1, 1))
    # zgrid = gl.GLGridItem(color=(1, 1, 1, 1))
    # view.addItem(xgrid)
    # view.addItem(ygrid)
    # view.addItem(zgrid)
    # xgrid.rotate(90, 0, 1, 0)
    # ygrid.rotate(90, 1, 0, 0)
    # xgrid.scale(10, 10, 10)
    # ygrid.scale(10, 10, 10)
    # zgrid.scale(10, 10, 10)
    return view


def plot_results(volume, mass, potentials):
    """ Plots the results given the volume, mass, and the array of potentials 
    
    The potential at each element of the volume is proportional to the element's red channel
    """
    app = pg.mkQApp()
    view = create_pg_view()
    # Create a color map for the volume
    volume_colors = np.ones((len(volume), 4), dtype=np.float32)
    potentials_normalized = potentials - np.min(potentials)
    # Make the red values reflect potentials values
    volume_colors[:, 0] = potentials_normalized / np.max(potentials_normalized)
    volume_colors[:, 1] = 0  # Set green values to 0 for a purple gradient
    volume_colors[:, 2] = .5  # Set blue values to 50%
    volume_scatter = gl.GLScatterPlotItem(pos=volume, color=volume_colors)
    mass_scatter = gl.GLScatterPlotItem(pos=mass, color=(0, 1, 0, 1))
    volume_scatter.setGLOptions('opaque')
    mass_scatter.setGLOptions('opaque')
    view.addItem(volume_scatter)
    view.addItem(mass_scatter)
    app.exec_()


def calculate_example(num_mass_points, num_volume_points, min_distance_from_volume, num_cubes=1):
    """ Performs an example computation with a unique apparatus of approximately the given number of points 
    
    Creates a universe of mass cubes around a volume cube, and calculates the gravitational potential at each voxel
    within the volume cube due to every voxel of every mass cube.
    
    Plots the mass cubes in green. Plots the volume cube on a gradient from blue to red, where the red channel is
    directly proportional to the relative gravitational potential of the point.
    """
    mass_side_length = int(np.ceil((num_mass_points / num_cubes) ** (1 / 3)))
    volume_side_length = int(np.ceil(num_volume_points ** (1 / 3)))
    volume = make_cube(volume_side_length)
    cubes = []
    for i in range(num_cubes):
        cube = make_cube(mass_side_length)
        cube[:, 0] += min_distance_from_volume + max((volume_side_length, mass_side_length)) * random.choice(
            (-2, 2)) + random.randint(-i * mass_side_length - mass_side_length, mass_side_length * num_cubes)
        cube[:, 1] += random.randint(-i * mass_side_length - mass_side_length, mass_side_length * num_cubes)
        cube[:, 2] += random.randint(-i * mass_side_length - mass_side_length, mass_side_length * num_cubes)
        cubes += [cube]
    mass = np.concatenate(cubes)
    cl_objects = generate_opencl_objects(volume, mass)
    potentials = calculate_potentials_opencl(cl_objects, random.random(), random.random())
    print 'Calculated %d potentials due to %d points, totaling %d potential calculations. Plotting result...' % (
        len(volume), len(mass), (len(volume) * len(mass)))
    plot_results(volume, mass, potentials)


if __name__ == '__main__':
    import __main__ as main

    if hasattr(main, '__file__'):  # Run the example if the module is not being run interactively
        calculate_example(20000, 100000, 10, num_cubes=20000)
