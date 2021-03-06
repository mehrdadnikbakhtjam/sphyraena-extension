﻿CC = /usr/bin/gcc
CUDA = /usr/local/cuda
NVCC = $(CUDA)/bin/nvcc #/usr/local/cuda/bin/nvcc
FLAGS = -Wall -Wextra -pedantic -std=c99
SQLITE_DIR = ../sqlite
SQLITE_VER = 3.31.1
SQLITESRC_DIR = ../sqlite-$(SQLITE_VER)/src
CUDA_INCLUDE = $(CUDA)/include
CUDA_LIBRARY = $(CUDA)/lib
DB = ../db/main.db

CUSTOM_FLAGS = -DSPHY_DEBUG -DRUNTEST

DEBUG_FLAGS = $(FLAGS) $(CUSTOM_FLAGS) -g3 -I$(SQLITE_DIR) -I$(SQLITESRC_DIR) -I$(CUDA_INCLUDE)
OPT_FLAGS = -O2 -xHost -ipo
LD = -lm -L$(CUDA_LIBRARY) -lcudart

FILES = cleanup2.c debug2.c driver2.c init2.c prepare_data2.c select2.c test2.c transfer_data2.c
DFILES = $(patsubst %.c,debug/%.o,$(FILES))


all: debug run

debug: debug/sphyraena2


debug/sqlite3.o: $(SQLITE_DIR)/sqlite3.c
	$(CC) $(SQLITE_DIR)/sqlite3.c -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_DEPRECATED -w -g -c -o debug/sqlite3.o

debug/opcodes.o: $(SQLITE_DIR)/opcodes.c
	$(CC) $(SQLITE_DIR)/opcodes.c -g -c -o debug/opcodes.o

$(DFILES): debug/%.o: %.c *.h
	$(CC) $(DEBUG_FLAGS) -c $< -o $@

debug/vm-ext.o: vm-ext.cu
	$(NVCC) $(CUSTOM_FLAGS) -I$(SQLITE_DIR) -I$(SQLITESRC_DIR) --ptxas-options="-v" -arch=sm_13 -g -c vm-ext.cu -o debug/vm-ext.o

debug/sphyraena2: $(DFILES) debug/sqlite3.o debug/vm-ext.o debug/opcodes.o
	$(CC) $(DEBUG_FLAGS) $(LD) $(DFILES) debug/opcodes.o debug/sqlite3.o debug/vm-ext.o -o debug/sphyraena2


clean:
	rm -f debug/* 

run: debug/sphyraena2
	export LD_LIBRARY_PATH=$(CUDA_LIBRARY) && debug/sphyraena2 -p -m -d $(DB)

profile:
	export LD_LIBRARY_PATH=$(CUDA_LIBRARY) && $(CUDA)/cudaprof/bin/cudaprof &



.PHONY: all debug clean run profile
