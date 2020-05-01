#define PROGRAM_FILE "conv_opencl.cl"
#define KERNEL_FUNC "conv_opencl"

#include "../utils.h"

void random_ints(cl_uint *v, int N) {
	int i;

	srand(time(NULL));
	for(i = 0; i < N; i++)
		v[i] = rand()%10;
	return;
}

int main(int argc, char *argv[]) {

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_uint i, j;
   cl_int err;

   /* Data and buffers*/
   cl_uint *in, *out;
   cl_mem in_buffer, out_buffer;
   cl_int num_groups;
   size_t max_workgroup;

   const int RADIUS = 4;

   int N = 256;
   if (argc == 2)
      N = atoi(argv[1]);

   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup, NULL);

   const size_t size = N * sizeof(cl_uint);
   const size_t local_size = max_workgroup;
   const size_t global_size = N;

   // Alloc space for device copies of a, b, c
	in = (cl_uint*) malloc(size); random_ints(in, N);
	out = (cl_uint*)malloc(size);
   
   program = build_program(context, device, PROGRAM_FILE);

   num_groups = N / local_size;

   printf("Num groups: %d GlobalSize: %ld LocalSize: %ld\n", num_groups, global_size, local_size);
   
   in_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, size, in, &err);
   out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };


   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buffer); // <=====OUTPUT
   err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &N);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &RADIUS);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   cl_event event;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, &event); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("%d\n", err);
      exit(1);
   }
   

   clWaitForEvents(1, &event);
   clFinish(queue);

   print_time_exec(event);


   err = clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, 
                  size, out, 0, NULL, NULL); // <=====GET OUTPUT
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

	for(i=0; i<100; i++) 
      printf("%d, ", in[i]);

   printf("\n\n");
   for(j=0; j<100; j++)
         printf("%d, ", out[j]);
   printf("\n");

   free(in);
   free(out);
   
   clReleaseKernel(kernel);
   clReleaseMemObject(in_buffer);
   clReleaseMemObject(out_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}
