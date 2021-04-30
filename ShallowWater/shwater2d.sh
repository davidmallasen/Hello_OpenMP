#!/bin/csh
foreach n (`seq 1 1 16`)
    echo $n
    env OMP_NUM_THREADS=$n srun -n 1 ./shwater2d_1000.out
end
