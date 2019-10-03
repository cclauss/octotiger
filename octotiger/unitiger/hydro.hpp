//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_UNITIGER_HYDRO_HPP_
#define OCTOTIGER_UNITIGER_HYDRO_HPP_
#include <vector>

#include "safe_real.hpp"

#define SAFE_MATH_ON
#include "../../octotiger/safe_math.hpp"

#ifdef NOHPX
#include <future>
using std::future;
using std::async;
using std::launch;
#endif

#include "octotiger/unitiger/cell_geometry.hpp"
#include "octotiger/unitiger/util.hpp"

#include <octotiger/cuda_util/cuda_helper.hpp>
#include <octotiger/cuda_util/cuda_scheduler.hpp>

namespace hydro {

using x_type = std::vector<std::vector<safe_real>>;

using flux_type = std::vector<std::vector<std::vector<safe_real>>>;


template<int NDIM>
using recon_type =std::vector<std::vector<std::array<safe_real, NDIM == 1 ? 3 : (NDIM == 2 ? 9 : 27)>>>;


using state_type = std::vector<std::vector<safe_real>>;
}

template<int NDIM, int INX>
struct hydro_computer: public cell_geometry<NDIM, INX> {
	using geo = cell_geometry<NDIM,INX>;

	enum bc_type {OUTFLOW, PERIODIC};

	const hydro::recon_type<NDIM>& reconstruct(hydro::state_type &U, const hydro::x_type&, safe_real );
//#ifdef OCTOTIGER_WITH_CUDA
	const hydro::recon_type<NDIM>& reconstruct_cuda(hydro::state_type &U, const hydro::x_type&, safe_real );
	void reconstruct_ppm(std::vector<std::vector<std::vector<safe_real>>> &Q_SoA,
		hydro::state_type &U, const hydro::x_type&, safe_real, int face_offset, int faces, const std::vector<bool> &smooth);
	void reconstruct_ppm_cpu(std::vector<std::vector<std::vector<safe_real>>> &Q_SoA,
		hydro::state_type &U, safe_real, int face_offset, int faces, const std::vector<bool> &smooth);
	void reconstruct_ppm_partial(std::vector<std::vector<std::vector<safe_real>>> &Q_SoA,
		hydro::state_type &U, safe_real, int face_offset, int faces, const std::vector<bool> &smooth);
//#endif

	safe_real flux(const hydro::state_type& U, const hydro::recon_type<NDIM> &Q, hydro::flux_type &F, hydro::x_type &X, safe_real omega);

	void post_process(hydro::state_type &U, safe_real dx);

	void boundaries(hydro::state_type &U);

	void advance(const hydro::state_type &U0, hydro::state_type &U, const hydro::flux_type &F, const hydro::x_type &X, safe_real dx, safe_real dt,
			safe_real beta, safe_real omega);

	void output(const hydro::state_type &U, const hydro::x_type &X, int num, safe_real);

	void use_angmom_correction(int index, int count);

	void use_smooth_recon(int field);

	std::vector<safe_real> get_field_sums(const hydro::state_type &U, safe_real dx);

	std::vector<safe_real> get_field_mags(const hydro::state_type &U, safe_real dx);

	hydro_computer();

	void set_bc( int face, bc_type bc) {
		bc_[face] = bc;
	}

	void set_bc( std::vector<bc_type>&& bc) {
		bc_ = std::move(bc);
	}

private:

	int nf_;
	int angmom_index_;
	int angmom_count_;
	std::vector<bool> smooth_field_;
	std::vector<bc_type> bc_;
}
;
#ifdef OCTOTIGER_HAVE_CUDA
void reconstruct_kernel_interface_sample(
	octotiger::fmm::struct_of_array_data<std::array<safe_real, 27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &D1,
	std::vector<octotiger::fmm::struct_of_array_data<std::array<safe_real, 27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>>> &Q,
	octotiger::fmm::struct_of_array_data<std::array<safe_real,27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &U,
	octotiger::fmm::struct_of_array_data<std::array<safe_real, 3>, safe_real, 3, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &X);

void reconstruct_ppm_interface(
	octotiger::fmm::struct_of_array_data<std::array<safe_real, 27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &D1,
	std::vector<octotiger::fmm::struct_of_array_data<std::vector<safe_real>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>>> &Q,
	octotiger::fmm::struct_of_array_data<std::array<safe_real,27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &U, int slot,
	int start_face, int end_face
);
#endif

#include <octotiger/unitiger/hydro_impl/hydro.hpp>

#endif /* OCTOTIGER_UNITIGER_HYDRO_HPP_ */
