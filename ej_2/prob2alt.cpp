#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fstream>

using namespace std;

const double sigma = 1;
const int Ncaminantes =1000;
const int Niteraciones = 1000;
double x[Niteraciones][Ncaminantes];

int main(){
	/*************************** GSL ******************************/
	const gsl_rng_type * T;
	gsl_rng * r;
	gsl_rng_env_setup();
  	T = gsl_rng_mt19937;
	r = gsl_rng_alloc (T);
	gsl_rng_set(r,5);
	
	/************************* PROGRAM ****************************/
	
	int i, j, k;
	double vartot, mediatot, media, var;
	
	for(i=0; i<Niteraciones; i++){
		for(j=0; j<Ncaminantes; j++){
			x[i][j]=0;
		}
	}
	
	for(i=1; i<Niteraciones; i++){
		for(j=0; j<Ncaminantes; j++){
			x[i][j] = x[i-1][j] + gsl_ran_gaussian(r,sigma);
		}
	}
	
	ofstream fout, fout2;
	fout.open("prob2med.txt");
	fout2.open("prob2var.txt");
	
	for(i=0; i<Niteraciones; i++){
		mediatot = 0;
		vartot = 0;
		for(j=0; j<Ncaminantes; j++){
			media = 0;
			var = 0;
			for(k=0; k<=i; k++){
				media += x[k][j];
				var += x[k][j]*x[k][j];
			}
			media = 1.*media/(i+1);
			var = 1.*var/(i+1) - media*media;
			mediatot += media;
			vartot += var;
		}
		mediatot = 1.*mediatot / Ncaminantes;
		vartot = 1.*vartot / Ncaminantes;
		fout << i << "	" << mediatot << endl;
		fout2 << i << "	" << vartot << endl;
	}
	
	fout.close();
	fout2.close();
	return 0;
}
