#ifdef OCTOTIGER_HAVE_CUDA

__global__ void kernel_ppm1(double *D1, double *U_, int face_offset);
__global__ void kernel_ppm2(double *Q, double *D1, double *U_, int face_offset);

#endif