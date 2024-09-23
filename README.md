# null_shuttle

Papers:

1. https://library.seg.org/doi/full/10.1190/geo2020-0192.1
2. https://academic.oup.com/gji/article/124/2/372/2058889

TODOs: (Besides the ones annotated in the code)

- Compute Hessian Vector product
- Verify first gradient is good and working
- Make plotting visuals clean
- Run Optimizer and look at the null shuttles

Make sure to install DrWatson before

Setup

```
module load Julia/1.8/5 cudnn-11 nvhpc-mpi Miniconda/3

export JULIA_NUM_THREADS=$SLURM_CPUS_ON_NODE

export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX
```
