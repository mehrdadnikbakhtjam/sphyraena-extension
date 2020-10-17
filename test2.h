#pragma once

const char *sphyraena_test_cases[] = {
        "SELECT uniformi, normali5, normali20 FROM test WHERE uniformi > 60 AND normali5 < 0 order by uniformi, normali5, normali20",
        "SELECT uniformf, normalf5, normalf20 FROM test WHERE uniformf > 60 AND normalf5 < 0 order by uniformf, normalf5, normalf20",
        "SELECT uniformi, normali5, normali20 FROM test WHERE uniformi > -60 AND normali5 < 5 order by uniformi, normali5, normali20",
        "SELECT uniformf, normalf5, normalf20 FROM test WHERE uniformf > -60 AND normalf5 < 5 order by uniformf, normalf5, normalf20", 
	"SELECT uniformi, normali5, normali20 FROM test WHERE (normali20 + 40) > (uniformi - 10) order by uniformi, normali5, normali20", 
	"SELECT uniformf, normalf5, normalf20 FROM test WHERE (normalf20 + 40) > (uniformf - 10) order by uniformf, normalf5, normalf20",
	"SELECT uniformi, normali5, normali20 FROM test WHERE normali5 * normali20 BETWEEN -5 AND 5 order by uniformi, normali5, normali20", 
	"SELECT uniformf, normalf5, normalf20 FROM test WHERE normalf5 * normalf20 BETWEEN -5 AND 5 order by uniformf, normalf5, normalf20", 
	"SELECT uniformi, normali5, normali20 FROM test WHERE NOT uniformi OR NOT normali5 OR NOT normali20 order by uniformi, normali5, normali20", 
	"SELECT uniformf, normalf5, normalf20 FROM test WHERE NOT uniformf OR NOT normalf5 OR NOT normalf20 order by uniformf, normalf5, normalf20"};
const int sphyraena_num_tests = 10;


const int sphyraena_num_tests_size = 5;
const char *sphyraena_test_cases_size[] = {
	"SELECT uniformf, normalf5, normalf20 FROM test WHERE id < 4000 order by uniformf, normalf5, normalf20",
        "SELECT uniformf, normalf5, normalf20 FROM test WHERE id < 20000 order by uniformf, normalf5, normalf20",
        "SELECT uniformf, normalf5, normalf20 FROM test WHERE id < 100000 order by uniformf, normalf5, normalf20",
        "SELECT uniformf, normalf5, normalf20 FROM test WHERE id < 500000 order by uniformf, normalf5, normalf20",
        "SELECT uniformf, normalf5, normalf20 FROM test WHERE id < 2500000 order by uniformf, normalf5, normalf20"};
