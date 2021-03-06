/* vim: set filetype=c : */
/* vim: set ts=6 : */
#include <cuda.h>

extern "C" {
#include "sphyraena2.h"
}

// these are operator templates. I did this this way so that I could use the same code but
// switch the operator
#define SPHYRAENA_VM_COMPARE(x)						\
	type = max(reg[op->p3].type, reg[op->p1].type);		\
	switch(type) {								\
		case SPHYRAENA_INT :						\
			jump = (castToInt(&reg[op->p3]) x castToInt(&reg[op->p1]));                                   \
			break;							\
		case SPHYRAENA_FLOAT :					\
			jump = (castToFloat(&reg[op->p3]) x castToFloat(&reg[op->p1]));		                           \
			break;							\
		case SPHYRAENA_INT64 :					\
			jump = (castToInt64(&reg[op->p3]) x castToInt64(&reg[op->p1]));		                           \
			break;							\
		case SPHYRAENA_DOUBLE :					\
			jump = (castToDouble(&reg[op->p3]) x castToDouble(&reg[op->p1]));	                           \
			break;							\
	}										\
	if(jump)									\
		pc = op->p2;							\                                                      
	else										\
		pc++;                                            \                                                             
                                                                                                                                                                                                                                                   

#define SPHYRAENA_VM_MATH(x)						\
	type = max(reg[op->p2].type, reg[op->p1].type);		\
	reg[op->p3].type = type;						   \
	switch(type) {								   \
		case SPHYRAENA_INT :						   \
			reg[op->p3].mem.i = (castToInt(&reg[op->p2]) x castToInt(&reg[op->p1]));	                                    \
			break;							   \
		case SPHYRAENA_FLOAT :					   \
			reg[op->p3].mem.f = (castToFloat(&reg[op->p2]) x castToFloat(&reg[op->p1]));	                                    \
			break;							   \
		case SPHYRAENA_INT64 :					   \
			reg[op->p3].mem.li = (castToInt64(&reg[op->p2]) x castToInt64(&reg[op->p1]));	                              \
			break;							   \
		case SPHYRAENA_DOUBLE :					   \
			reg[op->p3].mem.d = (castToDouble(&reg[op->p2]) x castToDouble(&reg[op->p1]));                                 \
			break;							   \
	}										   \

// these could probably be done in functions but you would have to pass a lot of arguments
// this makes it a bit simpler
#define SPHYRAENA_VM_LOGICAL_LOAD(x)					\
	switch(reg[op->x].type) {						\
		case SPHYRAENA_INT :						\
			val_##x = reg[op->x].mem.i;				\
			break;							\
		case SPHYRAENA_FLOAT :					\
			val_##x = (reg[op->x].mem.f == 0) ? 0 : 1;	\
			break;							\
		case SPHYRAENA_INT64 :					\
			val_##x = (reg[op->x].mem.li == 0) ? 0 : 1;	\
			break;							\
		case SPHYRAENA_DOUBLE :					\
			val_##x = (reg[op->x].mem.d == 0) ? 0 : 1;	\
			break;							\
	}

#define SPHYRAENA_VM_LOGICAL(x)			     \
	SPHYRAENA_VM_LOGICAL_LOAD(p1)			\
	SPHYRAENA_VM_LOGICAL_LOAD(p2)			\
	reg[op->p3].type = SPHYRAENA_INT;		\
	reg[op->p3].mem.i = (val_p1 x val_p2);


sphyraena_mem h_list[5000000][3],h_sorted[5000000][3];
__device__ void copyDataToReg(sphyraena_mem *reg, char *data, char* d, int column, int row);
__device__ int castToInt(sphyraena_mem *m);
__device__ float castToFloat(sphyraena_mem *m);
__device__ i64 castToInt64(sphyraena_mem *m);
__device__ double castToDouble(sphyraena_mem *m);
__device__ void load(int *i,int k,bool swap);
__device__ void compare(int x,int *i,int *j,int k,bool swap);
__device__ void merge_gpu(int start,int mid,int end, bool );

// the sql statement 
__constant__ sphyraena_stmt cstmt;

// the data block meta information
__constant__ sphyraena_data_gpu cdata;

__shared__ int result_start;
__shared__ int block_start;
__shared__ int reductionblock[256/*SPHYRAENA_THREADSPERBLOCK*/]; 
//__shared__ sphyraena_mem registerspace[SPHYRAENA_THREADSPERBLOCK * SPHYRAENA_REGISTERS];
//__device__ sphyraena_mem gregisterspace[SPHYRAENA_THREADSPERBLOCK * SPHYRAENA_REGISTERS];
//__shared__ sphyraena_mem reg_shared[SPHYRAENA_GLOBAL_REGISTERS];

__device__ sphyraena_mem reg_global[SPHYRAENA_GLOBAL_REGISTERS],list[5000000][3],sorted[5000000][3];
__device__ sphyraena_mem reg_global_sum[SPHYRAENA_GLOBAL_REGISTERS];
__device__ unsigned long long int count_global[SPHYRAENA_GLOBAL_REGISTERS];

__device__ int counter, gidx;
__device__ unsigned int block_order,block_order0[SPHYRAENA_GLOBAL_REGISTERS]; 
__device__ float avg_final; 



texture<float, 1, cudaReadModeElementType> texRef;

__global__ void vmKernel(char *data, int start_row, int num_blocks, int rows)
{

	// used for instances where a single thread handles multiple rows, for large data sets
	int curr_row = 0, mflag=0, idx=0;
	block_start = 0;

	// sets the global registers used for aggregates
	if(threadIdx.x < SPHYRAENA_GLOBAL_REGISTERS) {
		//reg_shared[threadIdx.x].mem.li = 0;
		if(blockIdx.x == 0) {
			reg_global[threadIdx.x].mem.li = 0;
			reg_global[threadIdx.x].type = SPHYRAENA_NULL;

		}
	}
       

new_row:

//here blockidx range is 0<=blockidx<num_blocks,curr_row is the number of num_blocks and threadidx range is 0<=threadIdx<256;in fact num_blocks*blockDim.x threads are iterating for curr_row times
	int row = (blockIdx.x + num_blocks * curr_row) * blockDim.x + threadIdx.x + start_row;

	sphyraena_mem reg[SPHYRAENA_REGISTERS];
	//sphyraena_mem *reg = (sphyraena_mem*) &registerspace[threadIdx.x * SPHYRAENA_REGISTERS];
	//sphyraena_mem *reg = (sphyraena_mem*) &gregisterspace[threadIdx.x * SPHYRAENA_REGISTERS];
     
	// program counter for opcode program
      int pc = 0;

	// stride of results block
	int stride = 0;
	// hardcoded registers for logical operations
	int val_p1, val_p2;
	// pointer to current operation
	sphyraena_op *op;

#ifdef COLUMNROW  
	char *d = data;
#else
	char *d = data + row * cdata.stride;
#endif

	// pointer to results block
	char *r = (char*) &results->r;
	// used to store the current variable type
	int type;
	int jump = 0;
	int aggfinal_num = 0;
	int i, j;

	//****this if checks the overflow of the threads;at this point the excessive threads will be stopped 
      if(row >= cdata.rows)
         goto halt;



// there are two modes of operation: divergent and synchronized
// the divergent block allows threads to diverge based on opcodes
// divergent is normally used, then we jump to synchronized on OP_Next
divergent:
	while(pc < SPHYRAENA_MAX_OPS) {
		op = &cstmt.op[pc];

		// this is a massive switch statement of opcodes. It is in alphabetical order other than
		// OP_Column, which is called frequently enough to make see a performance distance in moving
		// it to the front. Depending on the opcodes and size of the opcode programs the total switching
		// overhead for a query could be as high as 30%. On a CPU you could avoid this overhead with a 
		// jump table accessed with the opcode, but this is not supported on current (Tesla C1060)
		// hardware, so we are stuck with the overhead, but this is a major optimization for future
		// hardware.
		// The opcodes are documented on the SQLite website
		switch(op->opcode)
		{
			case OP_Column :  
			     copyDataToReg(&reg[op->p3], data, d, op->p2, row);
				//fprintf(stderr, "Column reg[%i] = %i  %i\n", op->p3, ((int*)(d + data->offsets[op->p2]))[0]);
				pc++;
				break;

			case OP_Add :
				SPHYRAENA_VM_MATH(+)
				pc++;
				break;

			case OP_AddImm :
				reg[op->p1].mem.i = castToInt(&reg[op->p1]) + op->p2;
				pc++;
				break;

			case OP_AggStep :
				switch(op->p4.i) {
					case SPHYRAENA_AGG_COUNT :
						reg[op->p3].mem.i = 1;
						break;

					case SPHYRAENA_AGG_SUM :
					case SPHYRAENA_AGG_MIN :
					case SPHYRAENA_AGG_MAX :
					case SPHYRAENA_AGG_AVG :
						reg[op->p3].mem.i = reg[op->p2].mem.i;//reg[op->p3].mem.li = reg[op->p2].mem.li;
						reg[op->p3].type = reg[op->p2].type;
						break;
				}
				pc++;
				break;

			case OP_And :
				SPHYRAENA_VM_LOGICAL(&&)
				pc++;
				break;

			case OP_BitAnd :
				reg[op->p3].mem.li = reg[op->p1].mem.li & reg[op->p2].mem.li;
				pc++;
				break;

			case OP_BitNot :
				reg[op->p2].mem.li = ~reg[op->p1].mem.li;
				pc++;
				break;

			case OP_BitOr :
				reg[op->p3].mem.li = reg[op->p1].mem.li | reg[op->p2].mem.li;
				pc++;
				break;

			case OP_Copy :
                        for(i = op->p2; i < op->p2 + op->p3; i++){
                           reg[i].type = reg[op->p1+i-op->p2].type;
				   reg[i].mem.li = reg[op->p1+i-op->p2].mem.li;
                           }
				   pc++;
				   break;

			case OP_SCopy :
				reg[op->p2].type = reg[op->p1].type;
				reg[op->p2].mem.li = reg[op->p1].mem.li;
				pc++;
				break;

			case OP_Divide :
				SPHYRAENA_VM_MATH(/)
				pc++;
				break;

			case OP_Eq :
				SPHYRAENA_VM_COMPARE(==)
				break;

			case OP_Ge :
				SPHYRAENA_VM_COMPARE(>=)
				break;

			case OP_Gt :
				SPHYRAENA_VM_COMPARE(>)
				break;

                  case OP_Goto :
                        pc = op->p2;
                        break;

			case OP_Halt :
				goto finish;

                  case OP_Init :
                        pc = op->p2;
                        break;

			case OP_If :
				SPHYRAENA_VM_LOGICAL_LOAD(p1)
				if(val_p1)
					pc = op->p2;
				else
					pc++;
				break;

			/*case OP_IfNeg :
				if(reg[op->p1].mem.i < 0)
					pc = op->p2;
				else
					pc++;
				break;*/

			case OP_IfNot :
				SPHYRAENA_VM_LOGICAL_LOAD(p1)
				if(!val_p1)
					pc = op->p2;
				else
					pc++;
				break;

			case OP_IfPos :
				if(reg[op->p1].mem.i > 0)
					pc = op->p2;
				else
					pc++;
				break;
			
			/*case OP_IfZero :
				if(reg[op->p1].mem.i == 0)
					pc = op->p2;
				else
					pc++;
				break;*/

			case OP_Int64 :
				reg[op->p2].type = SPHYRAENA_INT64;
				reg[op->p2].mem.li = op->p4.li;
				pc++;
				break;

			case OP_Integer :
				reg[op->p2].type = SPHYRAENA_INT;
				reg[op->p2].mem.i = op->p1;
				pc++;
				break;

			case OP_Le :
				SPHYRAENA_VM_COMPARE(<=)
				break;

			case OP_Lt :
				SPHYRAENA_VM_COMPARE(<)
				break;

			case OP_Multiply :
				SPHYRAENA_VM_MATH(*)
				pc++;
				break;

			case OP_Ne :
				SPHYRAENA_VM_COMPARE(!=)
				break;

			case OP_Next :
                        next:
                        __syncthreads();
                        if(gidx==0){
				  goto finish;}
                        else if(mflag==1)
                        {
                        pc++;
                        break;
                        }
                        else 
                        goto finish;

			case OP_Not :
				SPHYRAENA_VM_LOGICAL_LOAD(p1)
				reg[op->p2].type = SPHYRAENA_INT;
				reg[op->p2].mem.i = !val_p1;
				pc++;
				break;

			case OP_Null :
				reg[op->p2].mem.li = 0;
				reg[op->p2].type = SPHYRAENA_NULL;
				pc++;
				break;

			case OP_Or :
				SPHYRAENA_VM_LOGICAL(||)
				pc++;
				break;

			case OP_Real :
				reg[op->p2].type = SPHYRAENA_DOUBLE;
				reg[op->p2].mem.d = op->p4.d;
				pc++;
				break;

			case OP_Remainder :
				break;

			case OP_ResultRow :
				//fprintf(stderr, "resultrow    %i   %i\n", reg[7].mem.i, reg[8].mem.i);
				
				if(block_start != -1)
					j = atomicAdd(&block_start, 1);


				for(i = op->p1; i < op->p1 + op->p2; i++)
                         {
					results->types[i - op->p1] = reg[i].type;
					results->offsets[i - op->p1] = stride;
					switch(reg[i].type) {
						case SPHYRAENA_INT :
							stride += sizeof(int);
							break;
						case SPHYRAENA_FLOAT :
							stride += sizeof(float);
							break;
						case SPHYRAENA_INT64 :
							stride += sizeof(i64);
							break;
						case SPHYRAENA_DOUBLE :
							stride += sizeof(double);
							break;
					}
				}

				// this is a slight abuse of syncthreads. According to the documentation
				// syncthreads should never be called in a conditional not executed uniformly
				// across a threadblock. It turns out that synchthreads still works.
				// I implemented it this way because I had to use atomic functions for the reduction,
				// since even though every thread in the threadblock executes OP_Next, syncthreads
				// does not wait for divergent threads to catch up, so a normal reduction cannot be
				// performed. In the event that the implementation changes, this code can be moved
				// to OP_Next, and a proper coordinated reduction can be performed.
				__syncthreads();

				if(j == 0 && block_start != 0) {
					result_start = atomicAdd(&results->rows, block_start);
					block_start = 0;
				}

				__syncthreads();

				// round stride up to a power of 2
				/*stride--;
				stride |= stride >> 1;
				stride |= stride >> 2;
				stride |= stride >> 4;
				stride |= stride >> 8;
				stride |= stride >> 16;
				stride++;
				stride *= 4;*/

				results->stride = stride;
				results->columns = op->p2;

				r += (result_start + j) * stride;

				for(i = op->p1; i < op->p1 + op->p2; i++) {
					switch(reg[i].type) {
						case SPHYRAENA_INT :
							((int*)r)[0] = reg[i].mem.i;
							r += sizeof(int);
							break;
						case SPHYRAENA_FLOAT :
							((float*)r)[0] = reg[i].mem.i;
							r += sizeof(float);
							break;
						case SPHYRAENA_INT64 :
							((i64*)r)[0] = reg[i].mem.li;
							r += sizeof(i64);
							break;
						case SPHYRAENA_DOUBLE :
						((double*)r)[0] = reg[i].mem.d;
							r += sizeof(double);
							break;
					}
				}

				pc++;
				break;

			case OP_Rowid :
				// TODO change to actual pkey
				copyDataToReg(&reg[op->p2], data, d, 0, row);
				pc++;
				break;

			case OP_ShiftLeft :
				reg[op->p3].type = reg[op->p2].type;
				reg[op->p3].mem.li = reg[op->p2].mem.li << op->p1;
				pc++;
				break;

			case OP_ShiftRight :
				reg[op->p3].type = reg[op->p2].type;
				reg[op->p3].mem.li = reg[op->p2].mem.li >> op->p1;
				pc++;
				break;

			case OP_Subtract :
				SPHYRAENA_VM_MATH(-);
				pc++;
				break;

                        case OP_MakeRecord :
                              int iidx = 0;
                              mflag=1;
                              idx = atomicAdd(&gidx,1);
			      for(int i = op->p1; i < op->p1 + op->p2; i++) {
	                          switch(reg[i].type) {
				          case SPHYRAENA_INT :
							list[idx][iidx].mem.i = reg[i].mem.i;
							break;
					  case SPHYRAENA_FLOAT :
							list[idx][iidx].mem.f = reg[i].mem.f;
							break;
				          case SPHYRAENA_DOUBLE :
					      	list[idx][iidx].mem.d = reg[i].mem.d;
							break;
					}
                                  list[idx][iidx].type = reg[i].type;
                                  iidx++;
                                   }
                              __syncthreads();
                              pc++;
			      break;
                           
                        case OP_SorterSort :
                              __syncthreads();
                              pc++;
                              goto halt;
                        

                      /*case OP_SorterNext :
                              int c;
                              c= atomicAdd(&counter,1);
                              __syncthreads();
                              if(c < gidx){
                                 pc=op->p2;
                              }
                              else 
                                 pc++;

                              break;*/

//the implementation of OP_Seek** cases is a try to provide supporting of primary key in where section of the select commands
                        case OP_SeekGT :
                              copyDataToReg(&reg[op->p2], data, d, 0, row);
                              if(reg[op->p2].mem.i<=reg[op->p3].mem.i)
                                 goto finish;
                              else
                                 {
                                 pc++;
                                 break;
                                 }

                        case OP_SeekGE :
                              copyDataToReg(&reg[op->p2], data, d, 0, row);
                              if(reg[op->p2].mem.i<reg[op->p3].mem.i)
                                goto finish;
                              else
                                 {
                                  pc++;
                                  break;
                                 }

                        case OP_SeekLT :  
                              copyDataToReg(&reg[op->p2], data, d, 0, row);                     
                              if(reg[op->p2].mem.i>=reg[op->p3].mem.i)
                                goto finish;
                              else
                                  {
                                   pc++;
                                   break;
                                  } 

                        case OP_SeekLE : 
                              copyDataToReg(&reg[op->p2], data, d, 0, row);                      
                              if(reg[op->p2].mem.i>reg[op->p3].mem.i)
                                goto finish;
                              else
                                  {
                                   pc++;
                                   break;
                                  } 
                                        
                        default :
				 pc++;
				 break;
		}


	}
			


// this is the coordinated opcode block, for when it is essential that every thread acts in
// concert, as in the global aggregate reductions, note that several other opcodes are implemented
// in this block. These are used for post-aggregate operations, such as AVG(col1) + AVG(col2).
// Since all operation is done within a single kernel launch it is necessary to use atomics for
// the aggregate reductions, and establish threadblock order for ex post operations. Because
// of the threadblock ordering, performing multiple aggregates in the same query probably wont work,
// since there is no way to synchronize threadblocks. This should be avoided. All these are reasons
// that a multiple kernel launch model would be better, but that would erase SQLite registers,
// so a good deal more work is needed to accomplish that.
coordinated:

	while(pc < SPHYRAENA_MAX_OPS) {
		op = &cstmt.op[pc];

		switch(op->opcode)
		{

			case OP_AggFinal :
                                aggfinal_num++;

				switch(op->p4.i) {
					case SPHYRAENA_AGG_COUNT :
						// find the the next lowest power of 2 from the thread block size, including the current size
						i = SPHYRAENA_THREADSPERBLOCK;
						SPHYRAENA_ROUNDTOPWR2(i)
						j = i;
						i = i >> 1;
						i = 128;
				
						reductionblock[threadIdx.x] = reg[op->p1].mem.i;


						for( ; i > 0; i >>= 1, j >>= 1)
                                     {
							if(i >= 32)
								__syncthreads();
							if(threadIdx.x >= i && threadIdx.x < j)
								reductionblock[i - (threadIdx.x - i) - 1] += reductionblock[threadIdx.x];
						}

						//int tid = threadIdx.x;

						/*if(tid >= i)
							reductionblock[i - (tid - i) - 1] += reductionblock[tid];
						__syncthreads();
						if(i >= 512) {
							if(tid < 256)
								reductionblock[tid] += reductionblock[tid + 256];
							__syncthreads();
						}
						if(i >= 256) {
							if(tid < 128)
								reductionblock[tid] += reductionblock[tid + 128];
							__syncthreads();
						}
						if(i >= 128) {
							if(tid < 64)
								reductionblock[tid] += reductionblock[tid + 64];
							__syncthreads();
						}
						if(tid < 32) {
							if(i >= 64)
								reductionblock[tid] += reductionblock[tid + 32];
							if(i >= 32)
								reductionblock[tid] += reductionblock[tid + 16];
							if(i >= 16)
								reductionblock[tid] += reductionblock[tid +  8];
							if(i >=  8)
								reductionblock[tid] += reductionblock[tid +  4];
							if(i >=  4)
								reductionblock[tid] += reductionblock[tid +  2];
							if(i >=  2)
								reductionblock[tid] += reductionblock[tid +  1];
						}*/

						if(threadIdx.x == 0) {
							int count = atomicAdd(&reg_global[op->p1].mem.i, reductionblock[0]);
							reg[op->p1].mem.i = count + reductionblock[0];
							reg[op->p1].type = SPHYRAENA_INT;
						}
						break;

					case SPHYRAENA_AGG_SUM :
                                i = SPHYRAENA_THREADSPERBLOCK;
                                SPHYRAENA_ROUNDTOPWR2(i)
                                j = i;
                                i = i >> 1;

                                      reductionblock[threadIdx.x] = reg[op->p1].mem.i;

                                for( ; i > 0; i >>= 1, j >>= 1) {
                       
							if(i >= 32)
                                          	__syncthreads();

                                          if(threadIdx.x >= i && threadIdx.x < j)
                                                reductionblock[i - (threadIdx.x - i) - 1] += reductionblock[threadIdx.x];

                                    }

						if(threadIdx.x == 0) {
					        float it = atomicAdd(&reg_global[op->p1].mem.i, reductionblock[0]);
					        reg[op->p1].mem.i = it + reductionblock[0];
						   reg[op->p1].type = SPHYRAENA_FLOAT;
                                    }
						break;

					case SPHYRAENA_AGG_MIN :
                                i = SPHYRAENA_THREADSPERBLOCK;
                                SPHYRAENA_ROUNDTOPWR2(i)
                                j = i;
                                i = i >> 1;

                                reductionblock[threadIdx.x] = reg[op->p1].mem.i;

                                for( ; i> 0; i>>= 1, j>>= 1) 
                                    {
						    if(i >= 32)
                                        __syncthreads();
                                     if(threadIdx.x >= i && threadIdx.x < j)
  
SPHYRAENA_MIN(reductionblock[i - (threadIdx.x - i) - 1], reductionblock[threadIdx.x])
                                    }

                                     if(threadIdx.x == 0){
                                        i = atomicMin(&reg_global[op->p1].mem.i, reductionblock[0]);
                                          SPHYRAENA_MIN(i, reductionblock[0]);
                                          reg[op->p1].mem.i = i;
                                          reg[op->p1].type = SPHYRAENA_INT;
                                    }
                                    break;


                              case SPHYRAENA_AGG_MAX :
                                   i = SPHYRAENA_THREADSPERBLOCK;
                                   SPHYRAENA_ROUNDTOPWR2(i)
                                   j = i;
                                   i = i >> 1;

                                   reductionblock[threadIdx.x] = reg[op->p1].mem.i;

                                   for( ;i > 0; i>>= 1, j>>= 1)  
                                      {
							if(i >= 32)
                                          __syncthreads();
                                      if(threadIdx.x >= i && threadIdx.x < j)
                                                SPHYRAENA_MAX(reductionblock[i - (threadIdx.x - i) - 1], reductionblock[threadIdx.x])
                                      }

                                       if(threadIdx.x == 0) {
                                          i = atomicMax(&reg_global[op->p1].mem.i, reductionblock[0]);
                                          SPHYRAENA_MAX(i, reductionblock[0]);
                                          reg[op->p1].mem.i = i;
                                          reg[op->p1].type = SPHYRAENA_INT;
                                    }
                                  break;

					case SPHYRAENA_AGG_AVG :
                                i = SPHYRAENA_THREADSPERBLOCK;
                                SPHYRAENA_ROUNDTOPWR2(i)
                                j = i;
                                i = i >> 1;

                                reductionblock[threadIdx.x] = (reg[op->p1].type != SPHYRAENA_NULL) ? 1 : 0;

                                for( ; i > 0; i >>= 1, j >>= 1)  
                                   {
							if(i >= 32)
                                         __syncthreads();
                                         if(threadIdx.x >= i && threadIdx.x < j)
						        reductionblock[i - (threadIdx.x - i) - 1] += reductionblock[threadIdx.x];
                                    }

						   int sum;

                                    if(threadIdx.x == 0) {
                                      sum = atomicAdd(&reg_global_sum[op->p1].mem.i, reductionblock[0]);
							sum += reductionblock[0];
                                    }


                                    i = SPHYRAENA_THREADSPERBLOCK;
                                    SPHYRAENA_ROUNDTOPWR2(i)
                                    j = i;
                                    i = i >> 1;

						// int overflow?
                                    reductionblock[threadIdx.x] = reg[op->p1].mem.i;

                                    for( ; i > 0; i >>= 1, j >>= 1) {
							if(i >= 32)
                                          	__syncthreads();
                                          if(threadIdx.x >= i && threadIdx.x < j)
								reductionblock[i - (threadIdx.x - i) - 1] += reductionblock[threadIdx.x];
                                    }

						float avg;

                                    if(threadIdx.x == 0) {
                                          avg = (float)atomicAdd(&reg_global[op->p1].mem.i, reductionblock[0]);

							avg += reductionblock[0];
							avg /= sum;
							reg[op->p1].mem.f = avg;
							reg[op->p1].type = SPHYRAENA_FLOAT;
  
                                    }          
                                    break;
				
                            }

			     pc++;
		             break;
					
			case OP_Column :
			     goto divergent;

			case OP_Copy :
			case OP_SCopy :
                             reg[op->p2].type = reg[op->p1].type;
                             reg_global[op->p2].mem.i = reg_global[op->p1].mem.i;
		             pc++;
		             break;

			case OP_Halt :
			     goto finish;

			case OP_ResultRow :

				if(threadIdx.x == 0)
					//i = atomicInc(&block_order, gridDim.x);
					i = atomicAdd(&block_order, 1);
				else
					i = 0;

				if(i == gridDim.x - 1) {
					block_order = 0;


					for(i = op->p1; i < op->p1 + op->p2; i++) {
						results->types[i - op->p1] = reg[i].type;
						results->offsets[i - op->p1] = stride;

						switch(reg[i].type) {
							case SPHYRAENA_INT :
								stride += sizeof(int);
								break;
							case SPHYRAENA_FLOAT :
								stride += sizeof(float);
								break;
							case SPHYRAENA_INT64 :
								stride += sizeof(i64);
								break;
							case SPHYRAENA_DOUBLE :
								stride += sizeof(double);
								break;
						}
					}

					/*stride--;
					stride |= stride >> 1;
					stride |= stride >> 2;
					stride |= stride >> 4;
					stride |= stride >> 8;
					stride |= stride >> 16;
					stride++;*/

					results->stride = stride;
					results->columns = op->p2;
					results->rows = 1;

					for(i = op->p1; i < op->p1 + op->p2; i++) {
						switch(reg[i].type) {
							case SPHYRAENA_INT :
								((int*)r)[0] = reg[i].mem.i;//reg_global[i].mem.i;
								r += sizeof(int);
								break;
							case SPHYRAENA_FLOAT :
								((float*)r)[0] = reg[i].mem.f;//reg_global[i].mem.i;
								r += sizeof(float);
								break;
							case SPHYRAENA_INT64 :
								((i64*)r)[0] = reg[i].mem.li;//reg_global[i].mem.li;
								r += sizeof(i64);
								break;
							case SPHYRAENA_DOUBLE :
							     ((double*)r)[0] = reg[i].mem.d;//reg_global[i].mem.d;
								r += sizeof(double);
								break;
						}
					}

				}
				pc++;
				break;

			default :
				pc++;
				break;
		}
	}

finish:
__syncthreads();
	curr_row++;
	if(curr_row < rows)
		goto new_row;

halt:
__syncthreads();	
return;
}

__global__ void mergesort_gpu(int chunk,bool swap)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * chunk;
    if(start >= gidx) return;
    int mid, end;
    mid = min(start + chunk/2, gidx);
    end = min(start + chunk, gidx);
    merge_gpu(start, mid, end, swap);
    if(threadIdx.x==0 && blockIdx.x==0)   
    printf("sorted[0][0]:%f,sorted[1][0]:%f,sorted[2][0]:%f,sorted[3][0]:%f mergesort_gpu\n",sorted[0][0].mem.f,sorted[1][0].mem.f,sorted[2][0].mem.f,sorted[3][0].mem.f);
}

__device__ void copyDataToReg(sphyraena_mem *reg, char *data, char* d, int column, int row)
{
	reg->type = cdata.types[column];

#ifdef COLUMNROW
	/*if(reg->type <= SPHYRAENA_FLOAT) {
		reg->mem.f = tex1Dfetch(texRef, (cdata.offsets[column] + row * sizeof(int)) / sizeof(int));
	}
	else {
		reg->mem.segment.hi = tex1Dfetch(texRef,
			(cdata.offsets[column] + row * sizeof(i64)) / sizeof(int));
		reg->mem.segment.lo = tex1Dfetch(texRef,
			(cdata.offsets[column] + row * sizeof(i64) + sizeof(int)) / sizeof(int));
	}*/

	char* p = d + cdata.offsets[column] + row * sizeof(int);

	switch(cdata.types[column]) {
		case SPHYRAENA_INT :
			reg->mem.i = ((int*)p)[0];
			break;
		case SPHYRAENA_FLOAT :
			reg->mem.f = ((float*)p)[0];
			break;
		case SPHYRAENA_INT64 :
			reg->mem.li = ((i64*)p)[0];
			break;
		case SPHYRAENA_DOUBLE :
			reg->mem.d = ((double*)p)[0];
			break;
	}
#else
	/*reg->mem.i = tex1Dfetch(texRef, (row * cdata.stride + cdata.offsets[column]) / sizeof(int));
	if(reg->type > SPHYRAENA_FLOAT)
		reg->mem.segment.lo = tex1Dfetch(texRef,
			(row * cdata.stride + cdata.offsets[column]) / sizeof(int) + sizeof(int));*/


	//printf("%i\n", reg->mem.i);
	char* p = d + cdata.offsets[column];

	switch(cdata.types[column]) {
		case SPHYRAENA_INT :
			reg->mem.i = ((int*)p)[0];
			break;
		case SPHYRAENA_FLOAT :
			reg->mem.f = ((float*)p)[0];
			break;
		case SPHYRAENA_INT64 :
			reg->mem.li = ((i64*)p)[0];
			//reg->mem.li = 0;
			//reg->mem.segment.lo = ((int*)p)[0];
			//reg->mem.segment.hi = ((int*)p)[1];
			break;
		case SPHYRAENA_DOUBLE :
			reg->mem.d = ((double*)p)[0];
			break;
	}
#endif

}

__device__ int castToInt(sphyraena_mem *m)
{
	switch(m->type) {
		case SPHYRAENA_INT :
			return m->mem.i;
		case SPHYRAENA_FLOAT :
			return (int)m->mem.f;
		case SPHYRAENA_INT64 :
			return (int)m->mem.li;
		case SPHYRAENA_DOUBLE :
			return (int)m->mem.d;
	}
	return 0;
}

__device__ float castToFloat(sphyraena_mem *m)
{
	switch(m->type) {
		case SPHYRAENA_FLOAT :
			return m->mem.f;
		case SPHYRAENA_INT :
			return (float)m->mem.i;
		case SPHYRAENA_INT64 :
			return (float)m->mem.li;
		case SPHYRAENA_DOUBLE :
			return (float)m->mem.d;
	}
	return 0;
}

__device__ i64 castToInt64(sphyraena_mem *m)
{
	switch(m->type) {
		case SPHYRAENA_INT64 :
			return m->mem.li;
		case SPHYRAENA_INT :
			return (i64)m->mem.i;
		case SPHYRAENA_FLOAT :
			return (i64)m->mem.f;
		case SPHYRAENA_DOUBLE :
			return (i64)m->mem.d;
	}
	return 0;
}

__device__ double castToDouble(sphyraena_mem *m)
{
	switch(m->type) {
		case SPHYRAENA_DOUBLE :
			return m->mem.d;
		case SPHYRAENA_INT :
			return (double)m->mem.i;
		case SPHYRAENA_FLOAT :
			return (double)m->mem.f;
		case SPHYRAENA_INT64 :
			return (double)m->mem.li;
	}
	return 0;
}

__device__ void load(int *i,int k,bool swap){
        for(int c=0;c<3;c++){
           if(swap){
             switch (sorted[*i][c].type) {
               case SPHYRAENA_INT :
                    list[k][c].mem.i = sorted[*i][c].mem.i;
                    break;
               case SPHYRAENA_FLOAT :
                    list[k][c].mem.f = sorted[*i][c].mem.f;
                    break;
               case SPHYRAENA_DOUBLE :
                    list[k][c].mem.d = sorted[*i][c].mem.d;
                    break;
             }
             list[k][c].type=sorted[*i][c].type;
           }
              
           else {
             switch (list[*i][c].type) {
               case SPHYRAENA_INT :
                    sorted[k][c].mem.i = list[*i][c].mem.i;
                    break;
               case SPHYRAENA_FLOAT :
                    sorted[k][c].mem.f = list[*i][c].mem.f;
                    break;
               case SPHYRAENA_DOUBLE :
                    sorted[k][c].mem.d = list[*i][c].mem.d;
                    break;
             }
             sorted[k][c].type=list[*i][c].type;
          }
        }
 *i=*i+1;
} 

__device__ void compare(int x,int *i,int *j,int k,bool swap){
                   nextlevel : 
                       if(x>=3)
                          load(i,k,swap);
                       else if(swap)    
                              switch(sorted[*i][x].type){
                                     case SPHYRAENA_INT :
                                          if (sorted[*i][x].mem.i<sorted[*j][x].mem.i)
                                             load(i,k,swap);
                                          else if (sorted[*j][x].mem.i<sorted[*i][x].mem.i)
                                              load(j,k,swap);
                                          else {
                                                x++;
                                                goto nextlevel;
                                               } 
                                          break;

                                     case SPHYRAENA_FLOAT :
                                          if (sorted[*i][x].mem.f<sorted[*j][x].mem.f)
                                             load(i,k,swap);
                                          else if (sorted[*j][x].mem.f<sorted[*i][x].mem.f)
                                                  load(j,k,swap);
                                          else {
                                                x++;
                                                goto nextlevel;
                                               }
                                          break;

                                     case SPHYRAENA_DOUBLE :
                                          if (sorted[*i][x].mem.d<sorted[*j][x].mem.d)
                                             load(i,k,swap);
                                          else if (sorted[*j][x].mem.d<sorted[*i][x].mem.d)
                                                  load(j,k,swap);
                                          else {
                                                x++;
                                                goto nextlevel;
                                               }
                                          break;
                                    }            

                       else 
                             switch(list[*i][x].type){
                                    case SPHYRAENA_INT :
                                         if (list[*i][x].mem.i<list[*j][x].mem.i)
                                            load(i,k,swap);
                                         else if (list[*j][x].mem.i<list[*i][x].mem.i)
                                            load(j,k,swap);
                                         else {
                                               x++;
                                               goto nextlevel;
                                         }
                                         break;
  
                                   case SPHYRAENA_FLOAT :
                                        if (list[*i][x].mem.f<list[*j][x].mem.f)
                                           load(i,k,swap);
                                        else if (list[*j][x].mem.f<list[*i][x].mem.f)
                                                load(j,k,swap);
                                        else {
                                              x++;
                                              goto nextlevel;
                                        }
                                        break;

                                  case SPHYRAENA_DOUBLE :
                                       if (list[*i][x].mem.d<list[*j][x].mem.d)
                                          load(i,k,swap);
                                       else if (list[*j][x].mem.d<list[*i][x].mem.d)
                                               load(j,k,swap);
                                       else {
                                             x++;
                                             goto nextlevel;
                                       }
                                       break;
                              }
                       } 

__device__ void merge_gpu(int start, int mid, int end, bool swap)
{
    int i=start,j=mid,k=start;
    while( i < mid || j < end )
    {
        if(j==end)
             load(&i,k,swap);

        else if(i==mid) 
             load(&j,k,swap);
   
        else 
             compare(0,&i,&j,k,swap);

          k++;
    }
}

extern "C"
void h_load(int *i,int k,bool swap){
           for(int c=0;c<3;c++){
              if(swap) {
                switch (h_sorted[*i][c].type) {
                       case SPHYRAENA_INT :
                            h_list[k][c].mem.i = h_sorted[*i][c].mem.i;
                            break;
                       case SPHYRAENA_FLOAT :
                            h_list[k][c].mem.f = h_sorted[*i][c].mem.f;
                            break;
                       case SPHYRAENA_DOUBLE :
                            h_list[k][c].mem.d = h_sorted[*i][c].mem.d;
                            break;
                }
                h_list[k][c].type=h_sorted[*i][c].type;
              }

              else {
                    switch (h_list[*i][c].type) {
                           case SPHYRAENA_INT :
                                h_sorted[k][c].mem.i = h_list[*i][c].mem.i;
                                break;
                           case SPHYRAENA_FLOAT :
                                h_sorted[k][c].mem.f = h_list[*i][c].mem.f;
                                break;
                           case SPHYRAENA_DOUBLE :
                                h_sorted[k][c].mem.d = h_list[*i][c].mem.d;
                                break;
                    }
                    h_sorted[k][c].type=h_list[*i][c].type;
              }
           }
 *i=*i+1;
} 

extern "C"
void h_compare(int x,int *i,int *j,int k,bool swap){
                   nextlevel : 
                       if(x>=3)
                          h_load(i,k,swap);
                       else if(swap)    
                              switch(h_sorted[*i][x].type){
                                    case SPHYRAENA_INT :
                                         if (h_sorted[*i][x].mem.i<h_sorted[*j][x].mem.i)
                                            h_load(i,k,swap); 
                                         else if (h_sorted[*j][x].mem.i<h_sorted[*i][x].mem.i)
                                            h_load(j,k,swap);
                                         else {
                                               x++;
                                               goto nextlevel;
                                              }
                                         break;

                                    case SPHYRAENA_FLOAT :
                                         if (h_sorted[*i][x].mem.f<h_sorted[*j][x].mem.f)
                                            h_load(i,k,swap);
                                         else if (h_sorted[*j][x].mem.f<h_sorted[*i][x].mem.f)
                                            h_load(j,k,swap);
                                         else {
                                               x++;
                                               goto nextlevel;
                                              }
                                         break;

                                    case SPHYRAENA_DOUBLE :
                                         if (h_sorted[*i][x].mem.d<h_sorted[*j][x].mem.d)
                                            h_load(i,k,swap);
                                         else if (h_sorted[*j][x].mem.d<h_sorted[*i][x].mem.d)
                                                 h_load(j,k,swap);
                                         else {
                                               x++;
                                               goto nextlevel;
                                              }
                                         break;
                                    }            

                       else
                             switch(h_list[*i][x].type){
                                   case SPHYRAENA_INT :
                                        if (h_list[*i][x].mem.i<h_list[*j][x].mem.i)
                                           h_load(i,k,swap);
                                        else if (h_list[*j][x].mem.i<h_list[*i][x].mem.i)
                                           h_load(j,k,swap);

                                        else {
                                              x++;
                                              goto nextlevel;
                                             }
                                        break;

                                   case SPHYRAENA_FLOAT :
                                        if (h_list[*i][x].mem.f<h_list[*j][x].mem.f)
                                           h_load(i,k,swap);
                                        else if (h_list[*j][x].mem.f<h_list[*i][x].mem.f)
                                             h_load(j,k,swap);
                                        else {
                                              x++;
                                              goto nextlevel;
                                             }
                                        break;

                                  case SPHYRAENA_DOUBLE :
                                       if (h_list[*i][x].mem.d<h_list[*j][x].mem.d)
                                          h_load(i,k,swap);
                                       else if (h_list[*j][x].mem.d<h_list[*i][x].mem.d)
                                          h_load(j,k,swap);
                                       else {
                                             x++;
                                             goto nextlevel;
                                            }
                                       break;
                            }
}   

extern "C"
void merge(int start, int mid, int end, bool swap)
{
    int k=start, i=start, j=mid;
    while (i<mid || j<end)
    { 
     if (j==end)
        h_load(&i,k,swap);

     else if (i==mid)
        h_load(&j,k,swap);

     else 
        h_compare(0,&i,&j,k,swap);
        
     k++;
    } 
} 

extern "C"
int sphyraena_vm(sphyraena *s)
{
    cudaError_t r;
    cudaMemcpyToSymbol(cstmt, s->stmt_cpu, 
              sizeof(sphyraena_stmt), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cdata, s->data_cpu,
              sizeof(sphyraena_data_gpu), 0, cudaMemcpyHostToDevice);
    /*cudaMemset(s->results_gpu, 0, sizeof(int));*/
    int zero = 0;
    cudaMemcpyToSymbol(block_order, &zero, sizeof(int),
	      0, cudaMemcpyHostToDevice);

    /*const struct textureReference *texRefPtr;
    r = cudaGetTextureReference(&texRefPtr, "texRef");

    if(r != cudaSuccess) {
            fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
            return SPHYRAENA_ERR_TEXTURE;
    }

    cudaChannelFormatDesc channelDesc =
	    cudaCreateChannelDesc<float>();

    r = cudaBindTexture(0, texRefPtr, (char*)s->data_gpu,
	    &channelDesc, s->data_size);

    //texRef.filterMode = cudaFilterModeLinear;

    if(r != cudaSuccess) {
	    fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
            return SPHYRAENA_ERR_TEXTURE;
    }*/


    int blocks = (s->data_cpu->rows + s->threads_per_block - 1) / s->threads_per_block;
    int thread_rows = 1;
    if(blocks >= 65536) {
		         thread_rows = (int) ceilf((float)blocks / (float)65536);
          	         blocks = (int) ceilf((float)blocks / (float)thread_rows);
    }

    vmKernel<<<blocks, threads>>>((char*)s->data_gpu, 0, blocks, thread_rows);

    /*if((r = cudaGetLastError()) != cudaSuccess) 
        {  fprintf(stderr, "Cuda error: %s", cudaGetErrorString(r));      
            return SPHYRAENA_ERR_KERNEL;
      }*/

    r = cudaThreadSynchronize();

    if(r != cudaSuccess) 
      {
       fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
       return SPHYRAENA_ERR_KERNEL;
    }
    int hostgidx=0;
    cudaMemcpyFromSymbol(&hostgidx, gidx, sizeof(int),
		0, cudaMemcpyDeviceToHost);
    printf("gidx is:%d\n",hostgidx);

    bool swap=false, first=true, began=false;
    int start,mid=0,end=0;
    int chunk_size;
    int chunk_id;
    for(chunk_size=2; chunk_size<2 * hostgidx; chunk_size *= 2){
       int threads_required, blocks_required=0;
       threads_required =
       (hostgidx % chunk_size==0) ? hostgidx/chunk_size : (hostgidx / chunk_size + 1);
       blocks_required = 
       (threads_required + s->threads_per_block - 1) / s->threads_per_block;
       if(threads_required>=5000){
          mergesort_gpu<<<blocks_required,s->threads_per_block >>>(chunk_size, swap);
          r = cudaThreadSynchronize();
          if(r != cudaSuccess) {
	       fprintf(stderr, "Cuda error: %s\n",  cudaGetErrorString(r));
            printf("Chunk:%d\n",chunk_size);
	    return SPHYRAENA_ERR_KERNEL;
	  }
     }
        else {
              if(first){
                cudaMemcpyFromSymbol(h_list, list, 
                     3*hostgidx*sizeof(sphyraena_mem),0, cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(h_sorted, sorted,
                     3*hostgidx*sizeof(sphyraena_mem),0, cudaMemcpyDeviceToHost);
                     
                Printf("h_sorted[0][0]:%f,h_sorted[1][0]:%f, h_sorted[2][0]:%f,h_sorted[3][0]:%f cpu mergesort entrance\n", h_sorted[0][0].mem.f,h_sorted[1][0].mem.f,h_sorted[2][0].mem.f, h_sorted[3][0].mem.f);                                                                                                                                                                                                                                                                       
                Printf("h_list[0][0]:%f,h_list[1][0]:%f, h_list[2][0]:%f,h_list[3][0]:%f cpu mergesort entrance\n", h_list[0][0].mem.f,h_list[1][0].mem.f,h_list[2][0].mem.f, h_list[3][0].mem.f);                                                                                                                                                                                                                                                                       
                first=false;
                began=true;
     }


              for(chunk_id=0; chunk_id*chunk_size<=hostgidx; chunk_id++){
                 start = chunk_id * chunk_size;
                 if(start < hostgidx){
                   mid = min(start + chunk_size/2, hostgidx);
                   end = min(start + chunk_size, hostgidx);       
                   merge(start, mid, end, swap);
                 }
                else
                    break;
              }   
        }
     if(began)
     if(swap)
             printf("h_list[0][0]:%f,h_list[1][0]:%f, h_list[2][0]:%f,h_list[3][0]:%f cpu mergesort entrance\n",h_list[0][0].mem.f,h_list[1][0].mem.f, h_list[2][0].mem.f,h_list[3][0].mem.f);
     
     else
             printf("h_sorted[0][0]:%f,h_sorted[1][0]:%f, h_sorted[2][0]:%f,h_sorted[3][0]:%f cpu mergesort entrance\n",h_sorted[0][0].mem.f,h_sorted[1][0].mem.f,h_sorted[2][0].mem.f,h_sorted[3][0].mem.f);
     
             printf("blocks_required:%d,threads_required:%d ,hostgidx:%d,chunk:%d,swap:%d\n",blocks_required,threads_required,hostgidx,chunk_size,swap);

             swap=!swap;
     }  
     if(!swap){

        printf("h_list[0][0]:%f,h_list[0][1]:%f,h_list[0][2]:%f,
swap:%d after cpu mergesort sort\n",h_list[0][0].mem.f,h_list[0][1].mem.f,h_list[0][2].mem.f,!swap);

        printf("h_list[1][0]:%f,h_list[1][1]:%f,h_list[1][2]:%f,
swap:%d after cpu mergesort sort\n",h_list[1][0].mem.f,h_list[1][1].mem.f,h_list[1][2].mem.f,!swap);

        printf("h_list[2][0]:%f,h_list[2][1]:%f,h_list[2][2]:%f,
       swap:%d after cpu mergesort sort\n",h_list[2][0].mem.f,h_list[2][1].mem.f,
       h_list[2][2].mem.f,!swap);
     }/*don't forget that for integer values you should change h_list[x][y].mem.f to h_list[x][y].mem.i*/

     else{
          printf("h_sorted[0][0]:%f,h_sorted[0][1]:%f ,h_sorted[0][2]:%f,swap:%d after cpu mergesort sort\n",h_sorted[0][0].mem.f,h_sorted[0][1].mem.f,h_sorted[0][2].mem.f,!swap);

          printf("h_sorted[1][0]:%f,h_sorted[1][1]:%f, h_sorted[1][2]:%f,swap:%d after cpu mergesort sort\n",h_sorted[1][0].mem.f,h_sorted[1][1].mem.f,h_sorted[1][2].mem.f,!swap);

          printf("h_sorted[2][0]:%f,h_sorted[2][1]:%f, h_sorted[2][2]:%f,swap:%d after cpu mergesort sort\n",h_sorted[2][0].mem.f,h_sorted[2][1].mem.f,h_sorted[2][2].mem.f,!swap); /*don't forget that for integer values you should change h_sorted[x][y].mem.f to h_sorted[x][y].mem.i*/
    }
  
    grows=hostgidx;
	
    return SPHYRAENA_SUCCESS;
}

// performs the kernel call with streaming blocks, using s->stream_width to determine
// the number of streaming blocks


extern "C"
int sphyraena_vm_streaming(sphyraena *s)
{
	/*if(s->data_cpu->rows < 1000)
		return sphyraena_vm(s);*/
	cudaError_t r;
	cudaMemcpyToSymbol(cstmt, s->stmt_cpu, 
		sizeof(sphyraena_stmt), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cdata, s->data_cpu,
		sizeof(sphyraena_data), 0, cudaMemcpyHostToDevice);
	int zero = 0;
	cudaMemcpyToSymbol(block_order, &zero, sizeof(int),
		0, cudaMemcpyHostToDevice);

	int rows_per_stream = (s->data_cpu->rows + s->stream_width - 1) / s->stream_width;

	int blocks = (rows_per_stream + s->threads_per_block - 1) / s->threads_per_block;
	int block_size = s->data_cpu->stride * rows_per_stream;

	int thread_rows = 0;

	if(blocks >= 65536) {
		thread_rows = (int) ceilf((float)blocks / (float)65536);
		blocks = (int) ceilf((float)blocks / (float)thread_rows);
	}

	//printf("reg size  %i  block_size  %i\n", s->data_cpu->rows * s->data_cpu->stride, block_size);

	//printf("rps %i  blocks %i  block_size %i\n", rows_per_stream, blocks, block_size);

	cudaStream_t stream[s->stream_width];
	for(int i = 0; i < s->stream_width; i++)
		cudaStreamCreate(&stream[i]);

	for(int i = 0; i < s->stream_width; i++)
		cudaMemcpyAsync(s->data_gpu + block_size * i, s->data_cpu->d + block_size * i,
			block_size, cudaMemcpyHostToDevice, stream[i]);

	for(int i = 0; i < s->stream_width; i++) {
		vmKernel<<<blocks, s->threads_per_block, 0, stream[i]>>>((char*)s->data_gpu, i * blocks * s->threads_per_block, blocks, thread_rows);
	}

	for(int i = 0; i < s->stream_width; i++)
		cudaStreamDestroy(stream[i]);

	if((r = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
		return SPHYRAENA_ERR_KERNEL;
	}

	r = cudaThreadSynchronize();

	if(r != cudaSuccess) {
		fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(r));
		return SPHYRAENA_ERR_KERNEL;
	}

	return SPHYRAENA_SUCCESS;
}
