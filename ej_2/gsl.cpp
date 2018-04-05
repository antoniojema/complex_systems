//Para utilizar las gsl en Linux, deben instalarse los paquetes gsl-bin y libgsl-dev
//Consultar la siguiente web para más información
//https://www.gnu.org/software/gsl/manual/html_node/Random-Number-Generation.html#Random-Number-Generation

#include <stdio.h>
#include <gsl/gsl_rng.h>

int
main (void)
{
	int i, n = 10;
	const gsl_rng_type * T;
	gsl_rng * r;
	
	gsl_rng_env_setup(); //Inicializacion de las gsl
	gsl_rng_default_seed=0; //Permite cambiar la semilla por defecto
	
  	T = gsl_rng_mt19937; //gsl_rng_default; // Generador a utilizar
	r = gsl_rng_alloc (T); //Comprueba si hay suficiente memoria para emplear el generador
	gsl_rng_set(r,35); // Otra forma de cambiar la semilla, pero despues de llamar al generador. Elegir este caso o el anterior
	
	for (i = 0; i < n; i++) 
	{
	double u = gsl_rng_uniform (r);
	unsigned long int h = gsl_rng_uniform_int (r,100);
	printf ("Uniforme [0-1): %.5f 		Entero entre 1 y 100: %lu \n", u, h);
	}
	
	gsl_rng_free (r); //Libera toda la memoria que haya empleado el generador
	
	return 0;
}


