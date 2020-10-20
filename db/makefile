GSL_VER = 2.5

gendb: gendb-ext.c
	gcc -O2 -Wall -Wextra gendb-ext.c -I../sqlite -I../gsl -c -o gendb-ext.o
	gcc gendb-ext.o ../sqlite/sqlite3.o -ldl -lpthread -lgsl -lgslcblas -L../gsl/.libs -L../gsl/cblas/.libs -o gendb-ext
