# Load OBJ File Voxelizations and Quickly Analyze Gravitational Potentials

Thomas Schweich, GEE Lab at Washington University in St. Louis. MIT License.

The repository serves two purposes, as:

- A clean and efficient Python interface to files created by Dan Morris' [Voxelizer](http://dmorris.net/projects/voxelizer/).
- A massively parallel gravitational potential calculator for arbitrary 3d point matrices (which can be easily modified to serve
  as an electric potential calculator) using OpenCL and PyOpenCL.

Source code relating to the former is stored in the *voxel* directory, while source relating to the latter is stored in the
*main* directory.

## As a Python Interface to Dan Morris' [Voxelizer](http://dmorris.net/projects/voxelizer/)

### load_voxels.py

Running the script from the command line provides options for loading a single .voxels file or multiple files, indicated
by a file glob (using * as a wildcard), into a single `Voxels` Python object.
The script retains all information in the file(s) except the file header(s).
The script can be run interactively, in which case the Voxels object resulting from the loaded file(s) can be accessed
through a variable called 'voxels', such as in the following example:

```
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
```

where 'myvoxels_1.voxels', 'myvoxels_2.voxels'... are the names of your .voxels files.

The set resulting from the call to `voxels.to_set()` would contain the positions of each voxel from the file.
The VoxelfileVoxel object resulting from `voxels.voxels[0]` would be the first voxel in the file.
`ipython` could also be used in place of `python`. The dollar-sign (`$`) is simply meant to indicate that the proceeding
line is entered into a console (whose working directory is the one containing the script).

Use `$ python load_voxels.py -h` for help. The module may also, of course, simply be imported by other scripts.
The Voxels object also contains methods for dealing with the concatenation of multiple .voxels files into a single
object. This is due to the fact that the script was originally written for use with the voxelization of a complex CAD
file which was divided into many small components. As a result, the script would theoretically work with multiple
objects as well.

The script has been tested to run relatively fluently with multi-million-voxel files, including plotting. The test was
run on a laptop with a 2 core, 4 thread Intel i5 processor with 8GB RAM. Performance likely scales quite well with
improved hardware.

The module can also be imported and used in other programs.

### Other scripts

Scripts can be found in the *voxels* directory which serve the following purposes:

- Breaking apart a blender object into its component parts
- Combining a group of voxel files within a directory
- Repeatedly invoking the Voxelizer program on a directory of objects, and
- Detecting suspicious voxelizations

These are more one-off style scripts that will need to be modified to suit your needs.


## As a Massively Parallel Potential Calculator

### calculate_potentials.py

The important function contained in this file is `calculate_potentials_opencl`, along with its helper function `generate_opencl_objects`.
  The `generate_opencl_objects` takes the following parameters:
  - A 2d array of 3-vectors representing each point at which you would like the gravitational potential
  - A 2d array of 3-vectors representing each point of mass you would like to be accounted for in the calculations
  
 The `calculate_potentials_opencl` function takes the following parameters:
  - The result of your call to `generate_opencl_objects`
  - The mass density of the volume matrix material
  - The mass density of the mass matrix material

A function `plot_results` uses PyQTGraph to plot two resulting matrices, with the resulting potentials used as a color scale from
blue (low) to red (high). The module also contains functions for generating example spaces. By default, running the file will generate
an expansive universe of point masses with a cube matrix in the middle for which potential values are calculated. With reasonably new
hardware, despite being an O(n^2) operation, many points can be calculated *very* quickly.

### kernel_test.py

This file simply proves the accuracy of the calculation when compared to a similar calculation made using NumPy as well as rote Python.
Equality assertions are made, so if the script runs successfully your GPU (or other OpenCL device) has arrived at the same conclusion
as your CPU.

### potential_kernel.cl

The file contains the actual kernel used to calculate the gravitational potentials with OpenCL. The value of G could be changed to
Coulomb's Constant and electric charge could be used in place of mass to calculate electric potentials.


## Use in Conjunction

An arbitrary experimental apparatus and volume may be input into the voxelizing module, with the output of a call to `to_ndarray()`
fed into the potential calculation module in order to obtain the potential analysis of a scientific experiment. This was the original
intention of the project.

## License

Copyright 2017 Thomas Schweich and the GEE Lab at Washington University in St. Louis

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

