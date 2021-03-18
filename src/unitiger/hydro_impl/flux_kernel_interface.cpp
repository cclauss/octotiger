
#pragma message ("Starting to compile flux interface...")
#include "octotiger/cuda_util/cuda_helper.hpp"
#include "octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp"

#ifdef OCTOTIGER_HAVE_CUDA
#include <cuda_runtime.h>
#include <hpx/execution.hpp>
#include <hpx/synchronization/once.hpp>

#include <hpx/config.hpp>
#ifdef HPX_COMPUTE_HOST_CODE
#pragma message("Host pass!!")
#elif defined(HPX_COMPUTE_DEVICE_CODE)
#pragma message("Device pass!!")
#else
#pragma message("ERROR pass!!")
#endif
#ifdef HPX_COMPUTE_HOST_CODE

hpx::lcos::local::once_flag flag1;

__host__ void init_gpu_masks(bool *masks) {
  auto masks_boost = create_masks();
  cudaMemcpy(masks, masks_boost.data(), NDIM * 1000 * sizeof(bool), cudaMemcpyHostToDevice);
}

__host__ const bool* get_gpu_masks(void) {
    // TODO Create class to handle these read-only, created-once GPU buffers for masks. This is a reoccuring problem
    static bool *masks = recycler::recycle_allocator_cuda_device<bool>{}.allocate(NDIM * 1000);
    hpx::lcos::local::call_once(flag1, init_gpu_masks, masks);
    return masks;
}

__host__ timestep_t launch_flux_cuda(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
    double* device_q,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_f,
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> &combined_x, double* device_x,
    safe_real omega, const size_t nf_, double dx, size_t device_id) {
    timestep_t ts;

    const cell_geometry<3, 8> geo;

    recycler::cuda_device_buffer<double> device_f(NDIM * nf_ * 1000 + 32, device_id);
    const bool *masks = get_gpu_masks();

    recycler::cuda_device_buffer<double> device_amax(7 * NDIM * (1 + 2 * nf_));
    recycler::cuda_device_buffer<int> device_amax_indices(7 * NDIM);
    recycler::cuda_device_buffer<int> device_amax_d(7 * NDIM);
    double A_ = physics<NDIM>::A_;
    double B_ = physics<NDIM>::B_;
    double fgamma = physics<NDIM>::fgamma_;
    double de_switch_1 = physics<NDIM>::de_switch_1;
    int nf_local = physics<NDIM>::nf_;

    dim3 const grid_spec(1, 7, 3);
    dim3 const threads_per_block(2, 8, 8);
    void* args[] = {&(device_q),
      &(device_x), &(device_f.device_side_buffer), &(device_amax.device_side_buffer),
      &(device_amax_indices.device_side_buffer), &(device_amax_d.device_side_buffer),
      &masks, &omega, &dx, &A_, &B_, &nf_local, &fgamma, &de_switch_1};
    launch_flux_cuda_kernel_post(executor, grid_spec, threads_per_block, args);

    // Move data to host
    std::vector<double, recycler::recycle_allocator_cuda_host<double>> amax(7 * NDIM * (1 + 2 * nf_));
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> amax_indices(7 * NDIM);
    std::vector<int, recycler::recycle_allocator_cuda_host<int>> amax_d(7 * NDIM);
    cudaStream_t stream1;
    cudaError_t result;
    result = cudaStreamCreate(&stream1);
       cudaMemcpyAsync(amax.data(),
       device_amax.device_side_buffer, (7 * NDIM * (1 + 2 * nf_)) * sizeof(double),
       cudaMemcpyDeviceToHost, stream1);
    /*hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               cudaMemcpyAsync, amax.data(),
               device_amax.device_side_buffer, (7 * NDIM * (1 + 2 * nf_)) * sizeof(double),
               cudaMemcpyDeviceToHost);*/
    //cudaError_t (*my_memcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    cudaError_t (*my_memcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    my_memcpy = cudaMemcpyAsync;
    std::cout << typeid(my_memcpy).name() << std::endl;
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               my_memcpy, amax.data(),
               device_amax.device_side_buffer, (7 * NDIM * (1 + 2 * nf_)) * sizeof(double),
               cudaMemcpyDeviceToHost);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               my_memcpy, amax_indices.data(),
               device_amax_indices.device_side_buffer, 7 * NDIM * sizeof(int),
               cudaMemcpyDeviceToHost);
    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               my_memcpy, amax_d.data(),
               device_amax_d.device_side_buffer, 7 * NDIM * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
               my_memcpy, combined_f.data(), device_f.device_side_buffer,
               (NDIM * nf_ * 1000 + 32) * sizeof(double), cudaMemcpyDeviceToHost);
    fut.get();

    // Find Maximum
    size_t current_dim = 0;
    for (size_t dim_i = 1; dim_i < 7 * NDIM; dim_i++) {
      if (amax[dim_i] > amax[current_dim]) { 
        current_dim = dim_i;
      }
    }
    std::vector<double> URs(nf_), ULs(nf_);
    const size_t current_max_index = amax_indices[current_dim];
    const size_t current_d = amax_d[current_dim];
    ts.a = amax[current_dim];
    ts.x = combined_x[current_max_index];
    ts.y = combined_x[current_max_index + 1000];
    ts.z = combined_x[current_max_index + 2000];
    const size_t current_i = current_dim;
    current_dim = current_dim / 7;
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    constexpr int compressedH_DN[3] = {100, 10, 1};
    for (int f = 0; f < nf_; f++) {
        URs[f] = amax[21 + current_i * 2 * nf_ + f];
        ULs[f] = amax[21 + current_i * 2 * nf_ + nf_ + f];
    }
    ts.ul = std::move(ULs);
    ts.ur = std::move(URs);
    ts.dim = current_dim;
    return ts;
}
#endif

#endif

// Scalar kernel
template <>
CUDA_GLOBAL_METHOD inline void select_wrapper<double, bool>(
    double_t& target, const bool cond, const double& tmp1, const double& tmp2) {
    target = cond ? tmp1 : tmp2;
}
template <>
CUDA_GLOBAL_METHOD inline double max_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::max(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double min_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::min(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double sqrt_wrapper<double>(const double& tmp1) {
    return std::sqrt(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline double pow_wrapper<double>(const double& tmp1, const double& tmp2) {
    return std::pow(tmp1, tmp2);
}
template <>
CUDA_GLOBAL_METHOD inline double asin_wrapper<double>(const double& tmp1) {
    return std::asin(tmp1);
}
template <>
CUDA_GLOBAL_METHOD inline bool skippable<double>(const double& tmp1) {
    return !tmp1;
}

timestep_t flux_kernel_interface(const hydro::recon_type<NDIM>& Q, hydro::flux_type& F,
    hydro::x_type& X, safe_real omega, const size_t nf_) {
    // input Q, X
    // output F

    timestep_t ts;
    ts.a = 0.0;

    // bunch of small helpers
    static const cell_geometry<3, 8> geo;
    static constexpr auto faces = geo.face_pts();
    static constexpr auto weights = geo.face_weight();
    static constexpr auto xloc = geo.xloc();
    double p, v, v0, c;
    const auto A_ = physics<NDIM>::A_;
    const auto B_ = physics<NDIM>::B_;
    double current_amax = 0.0;
    size_t current_max_index = 0;
    size_t current_d = 0;
    size_t current_dim = 0;

    const auto dx = X[0][geo.H_DNX] - X[0][0];

    std::vector<double> UR(nf_), UL(nf_), this_flux(nf_);
    std::array<double, NDIM> x;
    std::array<double, NDIM> vg;

    for (int dim = 0; dim < NDIM; dim++) {
        const auto indices = geo.get_indexes(3, geo.face_pts()[dim][0]);

        std::array<int, NDIM> lbs = {3, 3, 3};
        std::array<int, NDIM> ubs = {geo.H_NX - 3, geo.H_NX - 3, geo.H_NX - 3};
        for (int dimension = 0; dimension < NDIM; dimension++) {
            ubs[dimension] = geo.xloc()[geo.face_pts()[dim][0]][dimension] == -1 ?
                (geo.H_NX - 3 + 1) :
                (geo.H_NX - 3);
            lbs[dimension] = geo.xloc()[geo.face_pts()[dim][0]][dimension] == +1 ? (3 - 1) : 3;
        }

        // zero-initialize F
        for (int f = 0; f < nf_; f++) {
#pragma ivdep
            for (const auto& i : indices) {
                F[dim][f][i] = 0.0;
            }
        }

        for (int fi = 0; fi < geo.NFACEDIR; fi++) {    // 9
            safe_real ap = 0.0, am = 0.0;              // final am ap for this i
            safe_real this_ap, this_am;                // tmps
            safe_real this_amax;
            const auto d = faces[dim][fi];

            const auto flipped_dim = geo.flip_dim(d, dim);
            // std::cout << "--Face:" << fi << "----------------------------------" << std::endl;
            for (size_t ix = lbs[0]; ix < ubs[0]; ix++) {
                for (size_t iy = lbs[1]; iy < geo.H_NX; iy++) {
                    for (size_t iz = lbs[2]; iz < geo.H_NX; iz++) {
                        if (iy >= ubs[1] || iz >= ubs[2])
                            continue;
                        const size_t i = ix * geo.H_NX * geo.H_NX + iy * geo.H_NX + iz;
                        // why store this?
                        for (int f = 0; f < nf_; f++) {
                            UR[f] = Q[f][d][i];
                            UL[f] = Q[f][flipped_dim][i - geo.H_DN[dim]];
                        }
                        for (int dim = 0; dim < NDIM; dim++) {
                            x[dim] = X[dim][i] + 0.5 * xloc[d][dim] * dx;
                        }
                        vg[0] = -omega * (X[1][i] + 0.5 * xloc[d][1] * dx);
                        vg[1] = +omega * (X[0][i] + 0.5 * xloc[d][0] * dx);
                        vg[2] = 0.0;
                        this_amax = inner_flux_loop<double>(omega, nf_, A_, B_, UR.data(),
                            UL.data(), this_flux.data(), x.data(), vg.data(), ap, am, dim, d, dx,
                            physics<NDIM>::fgamma_, physics<NDIM>::de_switch_1);
                        if (this_amax > current_amax) {
                            current_amax = this_amax;
                            current_max_index = i;
                            current_d = d;
                            current_dim = dim;
                        }
#pragma ivdep
                        for (int f = 0; f < nf_; f++) {
                            // field update from flux
                            F[dim][f][i] += weights[fi] * this_flux[f];
                        }
                    }    // end z
                }        // end y
            }            // end x
        }                // end dirs
    }                    // end dim
    ts.a = current_amax;
    ts.x = X[0][current_max_index];
    ts.y = X[1][current_max_index];
    ts.z = X[2][current_max_index];
    const auto flipped_dim = geo.flip_dim(current_d, current_dim);
    for (int f = 0; f < nf_; f++) {
        UR[f] = Q[f][current_d][current_max_index];
        UL[f] = Q[f][flipped_dim][current_max_index - geo.H_DN[current_dim]];
    }
    ts.ul = UL;
    ts.ur = UR;
    ts.dim = current_dim;
    return ts;
}
#pragma message("Finished pass!!")

