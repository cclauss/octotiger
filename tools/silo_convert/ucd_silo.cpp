/*
 * ucd_silo.cpp
 *
 *  Created on: Oct 21, 2019
 *      Author: dmarce1
 */




#include "./silo_convert.hpp"
#include <string.h>

ucd_silo::ucd_silo(const std::string filename) {
	db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Octo-Tiger", SILO_DRIVER);
}

void ucd_silo::add_mesh(std::string dir, DBquadmesh *mesh) {
	dtime = mesh->dtime;
	time = mesh->time;
	cycle = mesh->cycle;

	printf( "%s\n", dir.c_str());

}

void ucd_silo::add_var(std::string dir, DBquadvar *var) {

}

ucd_silo::~ucd_silo() {



	DBClose(db);
}

void ucd_silo::set_vars( double omega, int n_species, const std::vector<double>& atomic_mass, const std::vector<double>& atomic_number) {
	int one = 1;
	long long int _nspecies = n_species;
	DBWrite( db, "omega", &omega, &one, 1, DB_DOUBLE);
	DBWrite( db, "n_species", &_nspecies, &one, 1, DB_LONG_LONG);
	DBWrite( db, "atomic_mass", atomic_mass.data(), &n_species, 1, DB_DOUBLE);
	DBWrite( db, "atomic_number", atomic_number.data(), &n_species, 1, DB_DOUBLE);

}

