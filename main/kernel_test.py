""" Tests that the output of the OpenCL implementation of the gravitational potential calculator obtains the same result
as a non-parallel, easy to read NumPy implementation. Also compares their speeds.

All length units are assumed to be mm. The volume's material is assumed to be aluminum.
"""

from __future__ import division
import pyopencl as cl
import timeit

import numpy as np
import pyqtgraph
import itertools

G = 0.000062608


def make_cube(side_length):
    """ Generates a 2d array of 3-tuples containing the positions of each voxel in a cube of the given side length
    
    The generated cube has an origin at (0, 0, 0). All coordinates are integers.
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
    """ Harder to read python function which calculates potentials only using numpy vectorized functions 

    May actually be less memory efficient than the python implementation due to intermediate value creation
    """
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
    mf = cl.mem_flags

    # Input buffers
    volume_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=volume)
    mass_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mass)

    # Output buffer
    potentials = np.empty(len(volume), dtype=np.float32)
    potentials_out = cl.Buffer(ctx, mf.WRITE_ONLY, potentials.nbytes)

    ## Read and compile the opencl kernel
    with open('potential_kernel.cl') as kernel:
        prg = cl.Program(ctx, kernel.read()).build()
    return {'queue': queue, 'program': prg, 'volume_in': volume_in, 'mass_in': mass_in,
            'potentials_out': potentials_out}


def run_test(volume_side_length, mass_side_length, distance_apart, volume_material_mass, mass_material_mass):
    ## Create cubes representing the volume and the mass being tested
    volume = make_cube(volume_side_length)
    mass = make_cube(mass_side_length)

    ## Keep the volume at the origin, while moving the mass such that the volume's nearest side is `distance_apart`
    # mm away.
    mass[:, 0] += volume_side_length + distance_apart
    mass[:, 1] -= int(mass_side_length // 2)
    mass[:, 2] -= int(mass_side_length // 2)

    ## Get results using Python and NumPy
    python_result = calculate_potentials_python(volume, mass, volume_material_mass, mass_material_mass)
    numpy_result = calculate_potentials_numpy_vectorized(volume, mass, volume_material_mass, mass_material_mass)

    ## Make an assertion that calculate_potentials_python and calculate_potentials_numpy_vectorized yield the same
    # result. np.allclose measures equality within a small tolerance to allow for floating point error due to
    # architectural differences.
    # If the values resulting from the calls are not all approximately equal, the program raises an AssertionError and
    # quits.
    assert np.allclose(python_result, numpy_result)
    print '%d potentials calculated. The run was successful. Performing timing...' % len(python_result)

    ## Recalculate, this time measuring the amount of time each method takes. Each method is used 10 times, and the
    # average time taken per iteration is displayed. Note that this is not a 'proper' measurement -- a lambda function
    # passes the state of this function to the timeit module which takes a lot of time in of itself. However, the
    # relative results are useful.
    python_time = timeit.timeit(
        lambda: calculate_potentials_python(volume, mass, volume_material_mass, mass_material_mass), number=10) / 10
    numpy_time = timeit.timeit(
        lambda: calculate_potentials_numpy_vectorized(volume, mass, volume_material_mass, mass_material_mass),
        number=10) / 10

    print 'Python took %fs, NumPy took %fs. These results are only useful relative to each other.' % (
        python_time, numpy_time)


if __name__ == '__main__':
    run_test(5, 10, 1, .5, .5)
