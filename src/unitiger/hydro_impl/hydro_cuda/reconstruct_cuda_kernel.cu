#ifdef OCTOTIGER_HAVE_CUDA

__device__ const int NDIM = 3;
__device__ const int NDIR = 27;
__device__ const int INX = 8;
__device__ const int H_BW = 3;
__device__ const int H_NX = (2 * H_BW + INX);

__device__ const int H_NX_X = H_NX;
__device__ const int H_NX_Y = NDIM > 1 ? H_NX : 1;
__device__ const int H_NX_Z = NDIM > 2 ? H_NX : 1;

__device__ const int H_NX_XM2 = H_NX - 2;
__device__ const int H_NX_YM2 = NDIM > 1 ? H_NX - 2 : 1;
__device__ const int H_NX_ZM2 = NDIM > 2 ? H_NX - 2 : 1;

__device__ const int H_NX_XM4 = H_NX - 4;
__device__ const int H_NX_YM4 = NDIM > 1 ? H_NX - 4 : 1;
__device__ const int H_NX_ZM4 = NDIM > 2 ? H_NX - 4 : 1;

__device__ const int H_DNX = NDIM == 3 ? H_NX * H_NX : (NDIM == 2 ? H_NX : 1);
__device__ const int H_DNY = NDIM == 3 ? H_NX : 1;
__device__ const int H_DNZ = 1;
__device__ const int H_DN0 = 0;
__device__ static int dir[27] = {
	/**/-H_DNX - H_DNY - H_DNZ, +H_DN0 - H_DNY - H_DNZ, +H_DNX - H_DNY - H_DNZ,/**/
	/**/-H_DNX + H_DN0 - H_DNZ, +H_DN0 + H_DN0 - H_DNZ, +H_DNX + H_DN0 - H_DNZ,/**/
	/**/-H_DNX + H_DNY - H_DNZ, +H_DN0 + H_DNY - H_DNZ, +H_DNX + H_DNY - H_DNZ,/**/
	/**/-H_DNX - H_DNY + H_DN0, +H_DN0 - H_DNY + H_DN0, +H_DNX - H_DNY + H_DN0,/**/
	/**/-H_DNX + H_DN0 + H_DN0, +H_DN0 + H_DN0 + H_DN0, +H_DNX + H_DN0 + H_DN0,/**/
	/**/-H_DNX + H_DNY + H_DN0, +H_DN0 + H_DNY + H_DN0, +H_DNX + H_DNY + H_DN0,/**/
	/**/-H_DNX - H_DNY + H_DNZ, +H_DN0 - H_DNY + H_DNZ, +H_DNX - H_DNY + H_DNZ,/**/
	/**/-H_DNX + H_DN0 + H_DNZ, +H_DN0 + H_DN0 + H_DNZ, +H_DNX + H_DN0 + H_DNZ,/**/
	/**/-H_DNX + H_DNY + H_DNZ, +H_DN0 + H_DNY + H_DNZ, +H_DNX + H_DNY + H_DNZ/**/
};

__device__ inline int to_index(int j, int k, int l) {
    return (j * H_NX + k) * H_NX + l;
}

__device__ inline double minmod(double a, double b) {
	return (copysign(0.5, a) + copysign(0.5, b)) * min(abs(a), abs(b));
}

__device__ inline double minmod_theta(double a, double b, double c) {
	return minmod(c * minmod(a, b), 0.5 * (a + b));
}

constexpr size_t H_N3 = 14 * 14 * 14;

__global__ void kernel_reconstruct(double *q, double *D1_SoA, double *u, double *x_SoA, double omega, int face_offset) {
    bool first_thread = (blockIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    int id = (blockIdx.x * 14 *14) + (threadIdx.y * 14) + (threadIdx.z);
    
    const int z = threadIdx.z;
    const int y = threadIdx.y;
    const int x = blockIdx.z;
    if (x >= 14 || y >= 14 || z >= 14)
        return;
    const int f = blockIdx.y + face_offset;
    const int startindex = f * H_N3;
    const int i = startindex + to_index(x + 1, y + 1, z + 1);

    for (int d = 0; d < NDIR; d++) {
        const auto di = dir[d];
        D1_SoA[d * H_N3 + i] = minmod_theta(u[i + di] - u[i], u[i] - u[i - di], 2.0);
    }

    for (int d = 0; d < NDIR; d++) {
        const auto di = dir[d];
        q[d * H_N3 + i] = 0.5 * (u[i] + u[i + di]);
        q[d * H_N3 + i] += (1.0 / 6.0) * (D1_SoA[d * H_N3 + i] - D1_SoA[d * H_N3 + i + di]);
    }

/*     for (int gi = 0; gi < geo::group_count(); gi++) {
        safe_real sum = 0.0;
        for (int n = 0; n < geo::group_size(gi); n++) {
            const auto pair = geo::group_pair(gi, n);
            sum += q[pair.second][i + pair.first];
        }
        sum /= safe_real(geo::group_size(gi));
        for (int n = 0; n < geo::group_size(gi); n++) {
            const auto pair = geo::group_pair(gi, n);
            q[pair.second][i + pair.first] = sum;
        }
    } */

    return;
}

#endif 