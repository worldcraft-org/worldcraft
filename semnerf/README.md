# Running Nerfstudio on OCF HPC 

## Access

   ```bash
   # we have shared credentials
   ssh your_username@hpcctl.ocf.berkeley.edu

   # container should already be at: ~/nerfstudio_latest.sif
   # wrapper script should be at: ~/run-nerfstudio.sh

   # The wrapper script will create ~/nvlibs and copy required libraries
   # This happens automatically the first time you run it
   ```

## Usage

### Interactive Session

```bash
# request GPU node
srun --gres=gpu:1 --partition=ocf-hpc --mem=50G --cpus-per-task=16 --pty bash

# run nerfstudio commands through the wrapper
./run-nerfstudio.sh ns-train --help
./run-nerfstudio.sh ns-process-data --help
```

### Batch Job

https://www.ocf.berkeley.edu/docs/services/hpc/slurm/#h3_using-sbatch



## Important Notes

- **Always use the wrapper script** (`./run-nerfstudio.sh`)  don't run singularity directly cuz it won't work

## Files

- `~/nerfstudio_latest.sif` - the 4.8gb container 
- `~/run-nerfstudio.sh` - wrapper script to handle GPU setup
- `~/nvlibs/` - auto copied nvidia driver libraries 

## Resources

- [Nerfstudio Docs](https://docs.nerf.studio/)
