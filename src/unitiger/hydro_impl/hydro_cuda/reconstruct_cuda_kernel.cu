#ifdef OCTOTIGER_HAVE_CUDA

//__global__ void kernel_reconstruct(double *Q, double *D1, double *U_, double *X, double omega) {
__global__ void kernel_reconstruct(double omega) {
    bool first_thread = (blockIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    int id = (blockIdx.x * 14 *14) + (threadIdx.y * 14) + (threadIdx.z);
    if (first_thread)
        printf("Hello reconstruct");

}

#endif 