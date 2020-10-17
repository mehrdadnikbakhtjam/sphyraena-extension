#include "sphyraena2.h" 

int sphyraena_cleanup(sphyraena *s)
{
	
cudaError_t r;

	
if(s->pinned_memory) {
		
if((r = cudaFreeHost(s->data_cpu)) != cudaSuccess) {			
fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));		
	return SPHYRAENA_ERR_CUDAFREEHOST;
		
}

	
	
	if((r = cudaFreeHost(s->stmt_cpu)) != cudaSuccess) {
	
		fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
	
		return SPHYRAENA_ERR_CUDAFREEHOST;
		}
	}
	
else {
	
	free(s->data_cpu);
			
        free(s->stmt_cpu);

     }

	
        if((r = cudaFree((void*)s->data_gpu)) != cudaSuccess) {
             
                fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
         
                return SPHYRAENA_ERR_CUDAFREE;
 
       }

	
return SPHYRAENA_SUCCESS;

}
