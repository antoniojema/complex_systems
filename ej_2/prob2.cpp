#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fstream>

using namespace std;

const double sigma = 1;
const int Ncaminantes =1000;
const int Niteraciones = 1000;

int main(){
	/*************************** GSL ******************************/
	const gsl_rng_type * T;
	gsl_rng * r;
	gsl_rng_env_setup();
  	T = gsl_rng_mt19937;
	r = gsl_rng_alloc (T);
	gsl_rng_set(r,5);
	
	/************************* PROGRAM ****************************/
	
	double* x = new double[Ncaminantes]; //Posición de 1 caminante
	double* x_ = new double[Ncaminantes];//Suma de las posiciones de 1 caminante
	for(int i=0; i<Ncaminantes; i++){
		x[i]=0;
		x_[i]=0;
	}
	
	ofstream fout, fout2;
	fout.open("prob2med.txt");
	fout2.open("prob2var.txt");
	
	double sum = 0;	//Suma de las posiciones de todos los camenantes
	double sum2 = 0;//Suma de los cuadrados de las posic. de todos los caminantes
	double sum_2;	//Suma de los cuadrados de las acumuladas (x_) de todos los c.
	for(int i=1; i<=Niteraciones; i++){
		sum_2 = 0;
		for(int j=0; j<Ncaminantes; j++){
			x[j] = x[j] + gsl_ran_gaussian(r,sigma);
			x_[j] = x_[j] + x[j];
			
			sum += x[j];
			sum2 += x[j]*x[j];
			sum_2 += x_[j]*x_[j];
		}
		fout << i << "	" << 1.*sum / (Ncaminantes*(i+1)) << endl;
		fout2 << i << "	" << (1.*sum2 - 1.*sum_2/(i+1)) / (Ncaminantes*(i+1)) << endl;
	}
	
	fout.close();
	fout2.close();
	return 0;
}
