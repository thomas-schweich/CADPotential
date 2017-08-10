// potential_kernel.cl
// Written by Thomas Schweich, GEE Lab at Washington University in St. Louis, 2017

/* Kernel to compute the gravitational potentials due to an arbitrary mass matrix at each location in a volume matrix

After execution, the potential at position `volume_matrix_in[i]` is equal to `potential_matrix_out[i]`
for any given index `i`
*/
__kernel void compute_potential(__global const float3 *volume_matrix_in,    // The matrix of positions representing the volume
                                __global const float3 *mass_matrix_in,      // The matrix of positions representing the mass
                                const int size_mass_matrix,                 // The number of positions in the mass matrix
                                const float volume_material_mass,           // The mass density of the material the volume matrix is made of
                                const float mass_material_mass ,            // The mass density of the material the mass matrix is made of
                                __global float *potential_matrix_out) {     // The output array of gravitational potentials, whose indices correspond to those of volume_matrix_in
    const float G = 0.000062608;                                                                    // The gravitational constant
    int volume_voxel_i = get_global_id(0);                                                          // The index of the volume which this compute unit is working with
    float3 our_volume_voxel = volume_matrix_in[volume_voxel_i];                                     // The voxel this compute unit is working with
    int mass_voxel_i;                                                                               // The running index of the mass matrix
    float potential = 0.0;                                                                          // Set the starting potential to zero
    for(mass_voxel_i = 0; mass_voxel_i < size_mass_matrix; mass_voxel_i++) {                        // Loop through each element of the mass matrix
        float3 distances_squared = pown(our_volume_voxel - mass_matrix_in[mass_voxel_i], 2);        // Compute the square of the (x, y, z) vector component distances between this voxel
                                                                                                    // of the mass and the compute unit's voxel of the volume
        potential += (G * volume_material_mass * mass_material_mass) /                              // Add the potential due to this voxel of the mass matrix to our potential value
            sqrt(distances_squared.s0 + distances_squared.s1 + distances_squared.s2);
    }
    potential_matrix_out[volume_voxel_i] = potential;
}
