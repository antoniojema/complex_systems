program random
use fgsl

implicit none

integer*8 :: seed
type(fgsl_rng) :: r
type(fgsl_rng_type) :: ini_fgsl

!Inicializo aleatorios con las gsl.
seed=21
ini_fgsl = fgsl_rng_env_setup()
fgsl_rng_default_seed=seed
ini_fgsl = fgsl_rng_default
r = fgsl_rng_alloc(ini_fgsl)

write(*,*) fgsl_rng_uniform(r)

stop
end
