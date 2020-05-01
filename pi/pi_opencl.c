#define PROGRAM_FILE "pi_opencl.cl"
#define KERNEL_FUNC "pi_opencl"

#include "../utils.h"
#include <limits.h>

#define WG_SIZE 32

int main() {

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t local_size, global_size;

   /* Data and buffers    */
   unsigned int *seeds, *part_dentros, total_dentros;
   cl_mem seeds_buffer, part_dentros_buffer;
   cl_int num_groups;

   static unsigned int M = INT_MAX / 2;

   local_size = WG_SIZE;
   global_size = WG_SIZE * 128;

   /* Initialize seeds */
   seeds = (unsigned int*) calloc(global_size, sizeof(unsigned int));
   for(i=0; i< global_size; i++)
      seeds[i] = time(NULL) ^ i;

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
   num_groups = global_size / local_size;
   printf("Num groups: %d GlobalSize: %ld LocalSize: %ld\n", num_groups, global_size, local_size);
   seeds_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, global_size * sizeof(unsigned int), seeds, &err); // <=====INPUT
   part_dentros_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(unsigned int), NULL, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   free(seeds);

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
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &seeds_buffer); // <=====INPUT
   err |= clSetKernelArg(kernel, 1, local_size * sizeof(unsigned int), NULL);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &part_dentros_buffer); // <=====OUTPUT
   err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &M);
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

   part_dentros = (unsigned int*) calloc(num_groups, sizeof(unsigned int));

   clWaitForEvents(1, &event);
   clFinish(queue);

   print_time_exec(event);

   /* Read the kernel's output    */
   err = clEnqueueReadBuffer(queue, part_dentros_buffer, CL_TRUE, 0, 
         sizeof(unsigned int) * num_groups, part_dentros, 0, NULL, NULL); // <=====GET OUTPUT
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   total_dentros = 0;
   for(j=0; j<num_groups; j++)
      total_dentros += part_dentros[j];

   printf("%.50f\n", (4.0 * total_dentros) / M);

   free(part_dentros);
   
   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseMemObject(seeds_buffer);
   clReleaseMemObject(part_dentros_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}
