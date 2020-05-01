

__kernel void mtrx_opencl(const __global int* A,
                      const __global int* B,
                      __global int* C,
                      const int M) {
    
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
 
    int acc = 0;
    for (int k=0; k < M; k++)
      acc += A[k*M + globalRow] * B[k*M + globalCol];
 
    C[globalCol*M + globalRow] = acc;
}
