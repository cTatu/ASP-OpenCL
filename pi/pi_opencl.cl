
uint wang_hash(uint seed)
{
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
 }

  float random_num(uint* seed)                
 {
     uint maxint=0;
     maxint--; // not ok but works
     uint rndint=wang_hash(*seed);
     *seed=rndint;
     return ((float)rndint)/(float)maxint;
 }

__kernel void pi_opencl(__global unsigned int * seeds,
               __local unsigned int* local_result,
               __global unsigned int* group_dentros,
               unsigned int M) {

   uint gid, num_puntos, part_dentro, seed;

   gid = get_global_id(0);

   num_puntos = M / get_global_size(0);
   if (gid == 0)
      num_puntos += M % get_global_size(0);

   seed = seeds[gid];

   part_dentro = 0;
   for(uint i=0; i < num_puntos; i++){
      float x = random_num(&seed);
      float y = random_num(&seed);
      if (x*x + y*y <= 1)
         part_dentro++;
   }

   uint local_indx = get_local_id(0);
   local_result[local_indx] = part_dentro;

   barrier(CLK_LOCAL_MEM_FENCE);

   for (uint s=get_local_size(0)/2; s>0; s>>=1) {
      if (local_indx < s)
         local_result[local_indx] += local_result[local_indx + s];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(local_indx == 0)
      group_dentros[get_group_id(0)] = local_result[0];
}
