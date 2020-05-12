
__kernel void add_numbersMPI(__global long* group_sum, long init, long final, __local long* local_sum) {

   long tid, gid, total_hilos, register_sum, s;

   tid = get_local_id(0);
	gid = get_global_id(0);
	total_hilos = get_global_size(0);

   register_sum = 0;
	for(long i= init + gid; i <= final; i += total_hilos)
		register_sum += i;

	local_sum[tid] = register_sum;
   
   barrier(CLK_LOCAL_MEM_FENCE);

   for(s=get_local_size(0) / 2; s>0; s>>=1) { // En cada iteracion se opera sobre la mitad de array
		if (tid < s)
			local_sum[tid] += local_sum[tid + s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

   if (tid == 0)
 		group_sum[get_group_id(0)] = local_sum[0];
}
