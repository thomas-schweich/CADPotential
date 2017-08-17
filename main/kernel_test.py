""" kernel_test.py

Tests that the output of the OpenCL implementation of the gravitational potential calculator obtains the same result
as a non-parallel, easy to read NumPy implementation. Also compares their speeds.

All length units are assumed to be mm. The volume's material is assumed to be aluminum.
"""

from __future__ import division
import timeit
import numpy as np

from main.calculate_potentials import generate_opencl_objects, calculate_potentials_opencl, plot_results, make_cube, \
    make_cylinder

__author__ = 'Thomas Schweich, GEE Lab at Washington University in St. Louis'

G = 0.000062608


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
    """ Harder to read python function which calculates potentials only using NumPy vectorized functions """
    return np.fromiter(
        (np.sum((G * volume_material_mass * mass_material_mass) / np.sqrt(np.sum(np.square(volume_coord - mass), 1)))
         for volume_coord in volume),
        dtype=np.float32, count=len(volume))


def run_test_all(volume, mass, volume_material_mass, mass_material_mass):
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


def run_cube_test(volume_side_length, mass_side_length, distance_apart, volume_material_mass, mass_material_mass):
    ## Create cubes representing the volume and the mass being tested
    volume = make_cube(volume_side_length)
    mass = make_cube(mass_side_length)
    # mass = make_cylinder(mass_side_length // 2, mass_side_length)

    ## Keep the volume at the origin, while moving the mass such that the volume's nearest side is `distance_apart`
    # mm away and it is centered relative to the mass.
    mass[:, 0] += max((volume_side_length, mass_side_length)) + distance_apart
    mass[:, 1] -= int((mass_side_length - volume_side_length) // 2)
    mass[:, 2] -= int((mass_side_length - volume_side_length) // 2)

    run_test_all(volume, mass, volume_material_mass, mass_material_mass)


if __name__ == '__main__':
    run_cube_test(5, 15, 10, .5, .5)
