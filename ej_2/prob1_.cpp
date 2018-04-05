#include <iostream>
#include <fstream>

using namespace std;

int main(){
	unsigned long int Nmax = 2147483647;
	unsigned int c = 65539, a0 = 0;
	unsigned long long int x = 1;
	unsigned long long int i = 0;
	while(true){
		x = (c*x + a0) % Nmax;
		i++;
		if(x == 1){
			cout << "La secuencia se repite en la iteración " << i << endl;
			return 0;
		}
	}
	return 0;
}
