## Building the Container

  ### On Mac (using Lima)
  ```bash
  limactl start template://apptainer
  limactl shell apptainer
  cd ~
  apptainer pull docker://ghcr.io/nerfstudio-project/nerfstudio:latest
  exit
  ```

  ### Transfer to Cluster
  ```bash
  scp ~/nerfstudio_latest.sif username@hpcctl.ocf.berkeley.edu:~/
  ```
