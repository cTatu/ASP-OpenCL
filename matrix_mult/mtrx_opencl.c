#define PROGRAM_FILE "mtrx_opencl.cl"
#define KERNEL_FUNC "mtrx_opencl"

#include "../utils.h"


void print_mtrx(cl_uint* matrix, int M, int oldM){
   #ifdef DEBUG
      for(int i = 0; i < oldM; i++){
         for(int j = 0; j < oldM; j++)
            printf("%d\t", matrix[i*M + j]);
         printf("\n");
      }
      printf("\n");
   #endif
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
   cl_uint *matrixes, *matrix_a, *matrix_b, *matrix_c;
   cl_mem matrix_a_buffer, matrix_b_buffer, matrix_c_buffer;
   cl_int num_groups;
   size_t max_workgroup;

   int M = 5;
   if (argc == 2)
      M = atoi(argv[1]);

   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup, NULL);
   const cl_int TILE_SIZE = sqrt(max_workgroup);

   int oldM = M;
   if (M < TILE_SIZE)
      M = TILE_SIZE;
   else if (M % TILE_SIZE > 0)
      M += TILE_SIZE - (M % TILE_SIZE);
   
   printf("New padded M: %d\n", M);

   matrixes = (cl_uint*) calloc(M*M*3, sizeof(cl_uint));

   matrix_a = &matrixes[0 * M*M];
   matrix_b = &matrixes[1 * M*M];
   matrix_c = &matrixes[2 * M*M];

   for(i = 0; i < oldM; i++){
      for(j = 0; j < oldM; j++){
         matrix_a[i*M + j] = i*oldM + j+1;
         matrix_b[j*M + i] = i*oldM + j+1;
      }
   }

   
   print_mtrx(matrix_a, M, oldM);
   
   print_mtrx(matrix_b, M, oldM);
   

   const size_t local_size[2] = { TILE_SIZE, TILE_SIZE };
   const size_t global_size[2] = { M, M };
   
   program = build_program(context, device, PROGRAM_FILE);

   num_groups = M / TILE_SIZE;

   printf("Num groups: %d GlobalSize: %d LocalSize: %d\n", num_groups, M*M, TILE_SIZE*TILE_SIZE);
   
   matrix_a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, M*M * sizeof(cl_uint), matrix_a, &err);
   matrix_b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, M*M * sizeof(cl_uint), matrix_b, &err);
   matrix_c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M*M * sizeof(cl_uint), NULL, &err);
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

   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix_a_buffer);
   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &matrix_b_buffer);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &matrix_c_buffer); // <=====OUTPUT
   err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &M);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   cl_event event;
   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, 
         local_size, 0, NULL, &event); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("%d\n", err);
      exit(1);
   }
   

   clWaitForEvents(1, &event);
   clFinish(queue);

   print_time_exec(event);

   err = clEnqueueReadBuffer(queue, matrix_c_buffer, CL_TRUE, 0, 
         M*M * sizeof(cl_uint), matrix_c, 0, NULL, NULL); // <=====GET OUTPUT
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   print_mtrx(matrix_c, M, oldM);

   free(matrixes);
   
   clReleaseKernel(kernel);
   clReleaseMemObject(matrix_a_buffer);
   clReleaseMemObject(matrix_b_buffer);
   clReleaseMemObject(matrix_c_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}
