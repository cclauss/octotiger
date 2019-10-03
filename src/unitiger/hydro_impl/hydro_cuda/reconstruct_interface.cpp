
#include "octotiger/unitiger/hydro.hpp"

#include <octotiger/cuda_util/cuda_helper.hpp>
#include <octotiger/cuda_util/cuda_scheduler.hpp>
#include <octotiger/unitiger/hydro_impl/hydro_cuda/hydro_cuda_kernels.hpp>
#include <octotiger/common_kernel/struct_of_array_data.hpp>

#ifdef OCTOTIGER_HAVE_CUDA
void reconstruct_ppm_interface(
	std::vector<octotiger::fmm::struct_of_array_data<std::array<safe_real, 27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>>> &D1,
	std::vector<octotiger::fmm::struct_of_array_data<std::vector<safe_real>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>>> &Q,
	octotiger::fmm::struct_of_array_data<std::array<safe_real,27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &U, int slot,
	 int start_face, int end_face) {

	// Get interface
	octotiger::util::cuda_helper& gpu_interface =
		octotiger::fmm::kernel_scheduler::scheduler().get_launch_interface(slot);
	// Get staging area
	auto staging_area =
		octotiger::fmm::kernel_scheduler::scheduler().get_hydro_staging_area(slot);
	// Get kernel enviroment
	auto env =
		octotiger::fmm::kernel_scheduler::scheduler().get_hydro_device_enviroment(slot);
	// Move all Arrays
	// Move Arrays back
	// Launch dummy kernel
	// gpu_interface.copy_async(env.device_D1,
	// 	D1.get_pod(),
	// 	octotiger::fmm::d1_size, cudaMemcpyHostToDevice);
	gpu_interface.copy_async(env.device_U,
		U.get_pod(),
		octotiger::fmm::u_size, cudaMemcpyHostToDevice);
	int offset = start_face;
	void* args1[] = {&(env.device_D1),
		&(env.device_U), &offset};
	void* args2[] = {&(env.device_Q1),
		&(env.device_D1),
		&(env.device_U), &offset};

	dim3 const grid_spec(1, end_face - start_face, 12);
	dim3 const threads_per_block(1, 12, 12);
	gpu_interface.execute(
		reinterpret_cast<void*>(&kernel_ppm1),
		grid_spec, threads_per_block, args1, 0);
	gpu_interface.execute(
		reinterpret_cast<void*>(&kernel_ppm2),
		grid_spec, threads_per_block, args2, 0);

				
	for (size_t i = start_face; i < end_face; i++) {
		gpu_interface.copy_async(
		(Q[i]).get_pod(),env.device_Q1 + (octotiger::fmm::q_size) / (15 * sizeof(safe_real)) * i,
		octotiger::fmm::q_size/15, cudaMemcpyDeviceToHost);
	}
	// auto fut = gpu_interface.get_future();
	// fut.get();
	cudaError_t const response =
		gpu_interface.pass_through(
			[](cudaStream_t& stream) -> cudaError_t {
				return cudaStreamSynchronize(stream);
			});
}

void reconstruct_kernel_interface_sample(
	octotiger::fmm::struct_of_array_data<std::array<safe_real, 27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &D1,
	std::vector<octotiger::fmm::struct_of_array_data<std::array<safe_real, 27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>>> &Q,
	octotiger::fmm::struct_of_array_data<std::array<safe_real,27>, safe_real, 27, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &U,
	octotiger::fmm::struct_of_array_data<std::array<safe_real, 3>, safe_real, 3, 2744, 19, octotiger::fmm::pinned_vector<safe_real>> &X) {

    // octotiger::fmm::kernel_scheduler::scheduler().init();
	// // Get Slot
    // int slot = octotiger::fmm::kernel_scheduler::scheduler().get_launch_slot();
	// if (slot == -1) {
	// } else {
	// 	// Get interface
	// 	octotiger::util::cuda_helper& gpu_interface =
	// 		octotiger::fmm::kernel_scheduler::scheduler().get_launch_interface(slot);
	// 	// Get staging area
	// 	auto staging_area =
	// 		octotiger::fmm::kernel_scheduler::scheduler().get_hydro_staging_area(slot);
	// 	// Get kernel enviroment
	// 	auto env =
	// 		octotiger::fmm::kernel_scheduler::scheduler().get_hydro_device_enviroment(slot);
	// 	// Move all Arrays
	// 	// Move Arrays back
	// 	// Launch dummy kernel
	// 	gpu_interface.copy_async(env.device_D1,
	// 		D1.get_pod(),
	// 		octotiger::fmm::d1_size, cudaMemcpyHostToDevice);
	// 	for (size_t i = 0; i < 15; i++) {
	// 		gpu_interface.copy_async(env.device_Q1 + octotiger::fmm::q_size / (15 * sizeof(safe_real)) * i,
	// 			(Q[i]).get_pod(),
	// 			octotiger::fmm::q_size/15, cudaMemcpyHostToDevice);
	// 	}
	// 	gpu_interface.copy_async(env.device_U,
	// 		U.get_pod(),
	// 		octotiger::fmm::u_size, cudaMemcpyHostToDevice);
	// 	gpu_interface.copy_async(env.device_X,
	// 		X.get_pod(),
	// 		octotiger::fmm::x_size, cudaMemcpyHostToDevice);
    //     double local_omega = 1.1;
	// 	int offset = 0;
	// 	void* args[] = {&(env.device_D1),
	// 		&(env.device_Q1),
	// 		&(env.device_U),
	// 		&(env.device_X), &local_omega, &offset};

	// 	dim3 const grid_spec(1, 4, 14);
	// 	dim3 const threads_per_block(1, 14, 14);
	// 	std::cout << "Launching in slot " << slot << "..." << std::endl;
	// 	// gpu_interface.execute(
	// 	// 	reinterpret_cast<void*>(&kernel_reconstruct),
	// 	// 	grid_spec, threads_per_block, args, 0);

	// 	gpu_interface.copy_async(
	// 		D1.get_pod(),env.device_D1,
	// 		octotiger::fmm::d1_size, cudaMemcpyDeviceToHost);
	// 	for (size_t i = 0; i < 15; i++) {
	// 		gpu_interface.copy_async(
	// 			(Q[i]).get_pod(),env.device_Q1 + octotiger::fmm::q_size / (15 * sizeof(safe_real)) * i,
	// 			octotiger::fmm::q_size/15, cudaMemcpyDeviceToHost);
	// 	}
	// 	gpu_interface.copy_async(
	// 		U.get_pod(),env.device_U,
	// 		octotiger::fmm::u_size, cudaMemcpyDeviceToHost);
	// 	gpu_interface.copy_async(
	// 		X.get_pod(),env.device_X,
	// 		octotiger::fmm::x_size, cudaMemcpyDeviceToHost);
	// 	auto fut = gpu_interface.get_future();
	// 	fut.get();
	// }
}
#endif
