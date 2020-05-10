#define PROGRAM_FILE "add_numbersMPI.cl"
#define KERNEL_FUNC "add_numbersMPI"

#include <mpi.h>
#include "../utils.h"

int main(int argc, char *argv[]) {

   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, err;
   size_t local_size, global_size, max_workgroup;

   cl_mem sum_buffer;
   cl_int num_groups;

   // variable donde se almacenará el resultado
	long res = 0;

	clock_t t = clock();
   
   MPI_Init(NULL, NULL);

   int world_size, world_rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int M = 64;
	if (argc == 2)
		M = atoi(argv[1]);

   int init = (world_rank * M / world_size) + 1;
   int final = (world_rank + 1) * M / world_size;
   int num_nums = M / world_size;

   /* Create device and context 

   Creates a context containing only one device — the device structure 
   created earlier.
   */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create data buffer 

   • `global_size`: total number of work items that will be 
      executed on the GPU (e.g. total size of your array)
   • `local_size`: size of local workgroup. Each workgroup contains 
      several work items and goes to a compute unit 

   In this example, the kernel is executed by eight work-items divided into 
   two work-groups of four work-items each. Returning to my analogy, 
   this corresponds to a school containing eight students divided into 
   two classrooms of four students each.   

     Notes: 
   • Intel recommends workgroup size of 64-128. Often 128 is minimum to 
   get good performance on GPU
   • On NVIDIA Fermi, workgroup size must be at least 192 for full 
   utilization of cores
   • Optimal workgroup size differs across applications
   */
   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup, NULL);
   
   local_size = max_workgroup / 4;
   num_groups = num_nums / local_size;
   global_size = local_size*num_groups;
   printf("Num groups: %d GlobalSize: %ld LocalSize: %ld\n", num_groups, global_size, local_size);
   sum_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(long), NULL, &err); // <=====OUTPUT
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create a command queue 

   Does not support profiling or out-of-order-execution
   */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sum_buffer); // <=====OUTPUT
   err |= clSetKernelArg(kernel, 1, sizeof(int), &init);
   err |= clSetKernelArg(kernel, 2, sizeof(int), &final);
   err |= clSetKernelArg(kernel, 3, local_size * sizeof(long), NULL);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   /* Enqueue kernel 

   At this point, the application has created all the data structures 
   (device, kernel, program, command queue, and context) needed by an 
   OpenCL host application. Now, it deploys the kernel to a device.

   Of the OpenCL functions that run on the host, clEnqueueNDRangeKernel 
   is probably the most important to understand. Not only does it deploy 
   kernels to devices, it also identifies how many work-items should 
   be generated to execute the kernel (global_size) and the number of 
   work-items in each work-group (local_size).
   */
  cl_event event;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, &event); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   long* part_sum = (long*) calloc(num_groups, sizeof(long));

   clWaitForEvents(1, &event);
   clFinish(queue);

   print_time_exec(event);

   /* Read the kernel's output    */
   err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0, 
         num_groups * sizeof(long), part_sum, 0, NULL, NULL); // <=====GET OUTPUT
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   #pragma omp parallel reduction(+: res)
   for(i=0; i<num_groups; i++)
      res += part_sum[i];

   long sum_tot;
   MPI_Reduce(&res, &sum_tot, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Finalize();

   free(part_sum);
   
   printf("Computed sum = %ld\n", sum_tot);
   printf("Total tiempo: %f s\n", ((double)clock() - t) / CLOCKS_PER_SEC);

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(sum_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}