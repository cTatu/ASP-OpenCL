

__kernel void conv_opencl(const __global int* in,
                      __global int* out,
                      const int N, const int RADIUS) {
    
    const int gid = get_global_id(0);

    const int k[5] = { -2, -1, 0, 1, 2 };
 
    int res = 0;
    if(gid >= RADIUS && gid < N - RADIUS) {
      for (int offset = 0; offset < 2*RADIUS+1; offset++){
        int j = gid+offset-RADIUS;
		    res += in[j] * k[(N-j) % 5];
      }
    }
    
    out[gid] = res;
}
