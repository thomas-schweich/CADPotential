""" Tests that the output of the OpenCL implementation of the gravitational potential calculator obtains the same result
as a non-parallel, easy to read NumPy implementation. Also compares their speeds.

All length units are assumed to be mm. The volume's material is assumed to be aluminum.
"""

from __future__ import division
import pyopencl as cl
import timeit
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

G = 0.000062608


def make_cube(side_length):
    """ Generates a 2d array of 3-tuples containing the positions of each voxel in a cube of the given side length
    
    The generated cube has an origin at (0, 0, 0). All coordinates are 32 bit floats.
    """
    ## Create an empty array containing the cube of the side length number of points:
    cube = np.zeros((side_length ** 3, 3), dtype=np.float32)

    ## Create a sequence from 0 to the side length:
    sequence = np.linspace(0, side_length, num=side_length)

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
    """ Plots the results given the volume, mass, and the array of potentials """
    app = pg.mkQApp()
    view = create_pg_view()
    # Create a color map for the volume
    volume_colors = np.ones((len(volume), 4), dtype=np.float32)
    potentials_normalized = potentials - np.min(potentials)
    volume_colors[:, 0] = potentials_normalized / np.max(potentials_normalized)   # Make the red values reflect potentials values
    volume_colors[:, 1] = 0     # Set green values to 0 for a purple gradient
    volume_colors[:, 2] = .5    # Set blue values to 50%
    volume_scatter = gl.GLScatterPlotItem(pos=volume, color=volume_colors)
    mass_scatter = gl.GLScatterPlotItem(pos=mass, color=(0, 1, 0, 1))
    volume_scatter.setGLOptions('opaque')
    mass_scatter.setGLOptions('opaque')
    view.addItem(volume_scatter)
    view.addItem(mass_scatter)
    app.exec_()


def calculate_potentials_python(volume, mass, volume_material_mass, mass_material_mass):
    """ Easy to read python function which calculates potentials using two Python loops 
    
    Still uses NumPy for the rote math.
    """
    potentials = np.zeros(len(volume), dtype=np.float32)
    for volume_i, volume_coord in enumerate(volume):
        for mass_coord in mass:
            potentials[volume_i] += (G * volume_material_mass * mass_material_mass) / np.sqrt(
                np.square(volume_coord - mass_coord).sum())
    return potentials


def calculate_potentials_numpy_vectorized(volume, mass, volume_material_mass, mass_material_mass):
    """ Harder to read python function which calculates potentials only using numpy vectorized functions """
    return np.fromiter(
        (np.sum((G * volume_material_mass * mass_material_mass) / np.sqrt(np.sum(np.square(volume_coord - mass), 1)))
         for volume_coord in volume),
        dtype=np.float32, count=len(volume))


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


def run_test(volume_side_length, mass_side_length, distance_apart, volume_material_mass, mass_material_mass):
    ## Create cubes representing the volume and the mass being tested
    volume = make_cube(volume_side_length)
    mass = make_cube(mass_side_length)

    ## Keep the volume at the origin, while moving the mass such that the volume's nearest side is `distance_apart`
    # mm away and it is centered relative to the mass.
    mass[:, 0] += volume_side_length + distance_apart
    mass[:, 1] -= int((mass_side_length - volume_side_length) // 2)
    mass[:, 2] -= int((mass_side_length - volume_side_length) // 2)

    ## Get results using Python and NumPy
    python_result = calculate_potentials_python(volume, mass, volume_material_mass, mass_material_mass)
    numpy_result = calculate_potentials_numpy_vectorized(volume, mass, volume_material_mass, mass_material_mass)

    ## Get results using OpenCL
    # Create necessary objects to interface with the GPU
    opencl_objects = generate_opencl_objects(volume, mass)
    opencl_result = calculate_potentials_opencl(opencl_objects, volume_material_mass, mass_material_mass)

    ## Make an assertion that calculate_potentials_python and calculate_potentials_numpy_vectorized yield the same
    # result. np.allclose measures equality within a small tolerance to allow for floating point error due to
    # architectural differences.
    # If the values resulting from the calls are not all effectively equal, the program raises an AssertionError and
    # quits.
    assert np.allclose(python_result, numpy_result)

    ## Make an assertion that calculate_potentials_opencl yields the same result as the two python functions.
    #  It is now proven that all results are equal by transitive property.
    assert np.allclose(numpy_result, opencl_result)

    ## Indicate that the run was successful. If the program makes it to this point, that means all arrays were equal.
    print '%d potentials calculated. The run was successful. Performing timing...' % len(python_result)

    ## Recalculate, this time measuring the amount of time each method takes. Each method is used 10 times, and the
    # average time taken per iteration is displayed. Note that this is not a 'proper' measurement -- a lambda function
    # passes the state of this function to the timeit module which takes a lot of time in of itself. However, the
    # relative results are useful.
    python_time = timeit.timeit(
        lambda: calculate_potentials_python(volume, mass, volume_material_mass, mass_material_mass), number=2) / 2
    numpy_time = timeit.timeit(
        lambda: calculate_potentials_numpy_vectorized(volume, mass, volume_material_mass, mass_material_mass),
        number=10) / 10
    cl_time = timeit.timeit(
        lambda: calculate_potentials_opencl(opencl_objects, volume_material_mass, mass_material_mass), number=2) / 2

    print 'Python took %fs, NumPy took %fs, and OpenCL took %fs. ' \
          'These results are only useful relative to each other.' % (python_time, numpy_time, cl_time)

    plot_results(volume, mass, opencl_result)

if __name__ == '__main__':
    run_test(10, 15, 1, .5, .5)
