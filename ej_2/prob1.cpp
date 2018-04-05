#include <iostream>
#include <fstream>

using namespace std;

int main(){
	int iteraciones = 100000;
	int Ndiv = 100;
	int Nac = 1000;
	
	unsigned long int Nmax = 2147483647;
	unsigned int c = 65539, a0 = 0;
	unsigned long long int x = 1;
	double x_;
	ofstream fout1, fout2;
	fout1.open("prob1.txt");
	fout2.open("prob1_.txt");
	
	long int div[Ndiv];
	unsigned long long int acum[Nac];
	for(int j=0; j<Ndiv; j++) div[j] = 0;
	for(int j=0; j<Nac; j++) acum[j] = 0;
	
	x_ = 1.*x/Nmax;
	fout1 << 0 << "	" << x_ << endl;
	bool goon = true;
	unsigned long long int i = 0;
	for(int i=1; i<iteraciones; i++){
		fout2 << x_ << "	";
		x = (c*x + a0) % Nmax;
		x_ = 1.*x/Nmax;
		fout2 << x_ << endl;
		fout1 << i << "	" << x_ << endl;
		
		for(int j=1; j<=Ndiv; j++) if(x%j == 0) div[j-1]++;
		
		int ac = x_*Nac;
		for(int j=ac; j<Nac; j++) acum[j]++;
		
		if(x == 1){
			cout << "La secuencia se repite en la iteración " << i << endl;
			return 0;
		}
	}
	fout1.close();
	fout2.close();
	
	fout1.open("prob1_div.txt");
	for(int j=1; j<=Ndiv; j++) fout1 << j << "	" << div[j-1] << endl;
	fout1.close();
	
	fout1.open("prob1_ac.txt");
	for(int j=0; j<Nac; j++) fout1 << 1.*j/Nac << "	" << 1.*acum[j]/iteraciones << endl;
	fout1.close();
	
	return 0;
}
