
#include "octotiger/unitiger/hydro.hpp"

#include <octotiger/cuda_util/cuda_helper.hpp>
#include <octotiger/cuda_util/cuda_scheduler.hpp>
#include <octotiger/unitiger/hydro_impl/hydro_cuda/hydro_cuda_kernels.hpp>
#include <octotiger/common_kernel/struct_of_array_data.hpp>

void dummy(void) {
#ifdef OCTOTIGER_HAVE_CUDA

    octotiger::fmm::kernel_scheduler::scheduler().init();
	// Get Slot
    int slot = octotiger::fmm::kernel_scheduler::scheduler().get_launch_slot();
	if (slot == -1) {
	} else {
		// Get interface
		octotiger::util::cuda_helper& gpu_interface =
			octotiger::fmm::kernel_scheduler::scheduler().get_launch_interface(slot);
		// Get staging area
		auto staging_area =
			octotiger::fmm::kernel_scheduler::scheduler().get_hydro_staging_area(slot);
		// Get kernel enviroment
		auto env =
			octotiger::fmm::kernel_scheduler::scheduler().get_hydro_device_enviroment(slot);
		// Launch dummy kernel
        double local_omega = 1.1;
		void* args[] = {&(env.device_D1),
			&(env.device_Q1),
			&(env.device_U),
			&(env.device_X), &local_omega};

		dim3 const grid_spec(1, 1, 1);
		dim3 const threads_per_block(1, 12, 12);
		std::cout << "Launching in slot " << slot << "..." << std::endl;
		gpu_interface.execute(
			reinterpret_cast<void*>(&kernel_reconstruct),
			grid_spec, threads_per_block, args, 0);
		auto fut = gpu_interface.get_future();
		fut.get();
		std::cin.get();
	}
#endif

}
