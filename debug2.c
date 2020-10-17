#include "sphyraena2.h"
void sphyraena_timer_start()
{
        gettimeofday(&sphyraena_starttime, NULL);
}

double sphyraena_timer_stop()
{
        gettimeofday(&sphyraena_endtime, NULL);
        double start = sphyraena_starttime.tv_sec +
                (double)sphyraena_starttime.tv_usec * .000001;
        double end = sphyraena_endtime.tv_sec +
                (double)sphyraena_endtime.tv_usec * .000001;
	return (end - start);
}

double sphyraena_timer_end(const char* label)
{
	double time = sphyraena_timer_stop();
        printf("%s time: %f seconds\n", label, time);
	return time;
}

cudaError_t sphyraena_print_error()
{
	cudaError_t r = cudaGetLastError();

	if(r != cudaSuccess)
		fprintf(stderr, "cuda error: %s\n",
			cudaGetErrorString(r));

	return r;
}