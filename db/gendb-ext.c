#include <stdio.h>
#include "sqlite3.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


// returned row
static int callback(void *NotUsed, int argc, char **argv, char **azColName){
	int i;
	for(i=0; i<argc; i++){
		printf("%s = %s\t", azColName[i], argv[i] ? argv[i] : "NULL");
	}
	printf("\n");
	return 0;
}

// create test_int table in database
void create_int_tables(sqlite3 *db) {
	int r;
	char *err = 0;

	r = sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS test_int (id INTEGER PRIMARY KEY, uniform INTEGER, normal5 INTEGER, normal20 INTEGER, t20 INTEGER)",
		callback, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
}

// create test table in database
void create_tables(sqlite3 *db) {
	int r;
	char *err = 0;

	r = sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, uniformi INTEGER, normali5 INTEGER, normali20 INTEGER, uniformf FLOAT, normalf5 FLOAT, normalf20 FLOAT)",
		callback, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
}

// insert rows into test with varying random distributions
void insert_data(sqlite3 *db, int rows, gsl_rng *ran) {
	char* err = 0;
	int i;
	int r;

	r = sqlite3_exec(db, "PRAGMA synchronous = 0", NULL, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}

	fprintf(stderr, "executed pragma\n");

	r = sqlite3_exec(db, "BEGIN TRANSACTION", NULL, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}

	fprintf(stderr, "started transaction\n");

	sqlite3_stmt *stmt;
	sqlite3_prepare_v2(db,
		"INSERT INTO test (id, uniformi, normali5, normali20, uniformf, normalf5, normalf20) VALUES (?,?,?,?,?,?,?)", 
		512,
		&stmt,
		0);

	for(i = 0; i < rows; i++) {
		sqlite3_bind_int(stmt, 1, i);
		sqlite3_bind_int(stmt, 2, (int)gsl_ran_flat(ran, -100, 100));
		sqlite3_bind_int(stmt, 3, (int)gsl_ran_gaussian(ran, 5));
		sqlite3_bind_int(stmt, 4, (int)gsl_ran_gaussian(ran, 20));
		sqlite3_bind_int(stmt, 5, (float)gsl_ran_flat(ran, -100, 100));
		sqlite3_bind_int(stmt, 6, (float)gsl_ran_gaussian(ran, 5));
		sqlite3_bind_int(stmt, 7, (float)gsl_ran_gaussian(ran, 20));

		int r = sqlite3_step(stmt);

		if(r != SQLITE_DONE && r != SQLITE_OK) {
			fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
			sqlite3_free(err);
			return;
		}

		sqlite3_reset(stmt);
		
		if(i % 10000 == 0)
			printf("%i\n", i);
	}

	sqlite3_finalize(stmt);

	r=sqlite3_exec(db, "CREATE TEMPORARY TABLE tmp AS SELECT * FROM test", 0, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
        r=sqlite3_exec(db, "UPDATE tmp SET id = NULL;", 0, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
        for(int i=0;i<24;i++){
        r=sqlite3_exec(db, "INSERT INTO test SELECT * FROM tmp;", 0, 0, &err); 

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
}
        r=sqlite3_exec(db, "DROP TABLE tmp;", 0, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
	r=sqlite3_exec(db, "COMMIT", 0, 0, &err);

	if(r != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", err);
		sqlite3_free(err);
	}
}



int main(int argc, char **argv){
	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	if(argc != 2) {
		fprintf(stderr, "%s <database name>\n", argv[0]);
		exit(1);
	}

	const gsl_rng_type *type;
	gsl_rng *ran;

	type = gsl_rng_ranlxd2;
	ran = gsl_rng_alloc(type);
	gsl_rng_set(ran, time(0));

	rc = sqlite3_open(argv[1], &db);
	if( rc ){
		fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
		sqlite3_close(db);
		exit(1);
	}

	create_tables(db);

	fprintf(stderr, "tables created\n");

	insert_data(db, 100000, ran);
	

	if(rc != SQLITE_OK) {
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	}

	sqlite3_close(db);
	gsl_rng_free(ran);

	return 0;
}
