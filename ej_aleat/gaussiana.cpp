#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fstream>

using namespace std;

int main(){
	const gsl_rng_type * T;
	gsl_rng * r;
	
	gsl_rng_env_setup(); //Inicializacion de las gsl
	
  	T = gsl_rng_mt19937; //gsl_rng_default; // Generador a utilizar
	r = gsl_rng_alloc (T); //Comprueba si hay suficiente memoria para emplear el generador
	gsl_rng_set(r,35); // Otra forma de cambiar la semilla, pero despues de llamar al generador. Elegir este caso o el anterior
	
	int i, j;
	double x;
	ofstream fout;
	
	fout.open("data.txt");
	for(i=0;i<=1000;i++){
		x=0;
		fout << 0 << "	" << x << endl;
		for(j=1;j<=1000;j++){
			x += gsl_ran_gaussian(r,35);
			fout << j << "	" << x << endl;
		}
	}
	
	return 0;
}
