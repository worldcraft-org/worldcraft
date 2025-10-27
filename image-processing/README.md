# Image Processing
## Extraction of intrinsics & extrinsics
### OpenMVG
- complied and tested on x64 Ubuntu 22.04
- cd to openMVG_Build\software\SfM

#### Editing ./SfM_SequentialPipeline.py
- change OPENMVG_SFM_BIN, CAMERA_SENSOR_WIDTH_DIRECTORY to ur dir
- change the argument "-f" to 1.2 * width/height (whichever is larger) as initial focal length
- run in terminal ./SfM_SequentialPipeline.py <images> <output>
(.\Reduced_ImageDataset_Kitti360\images) (.\\Reduced_ImageDataset_Kitti360\recon)

#### Converting sfm.bin to sfm.json for human readability
- located in \Reduced_ImageDataset_Kitti360\recon\reconstruction_sequential\
- contains the results
- cd to \openMVG_Build\Linux-x86_64-RELEASE
- run ./openMVG_main_ConvertSfM_DataFormat -i path/to/sfm_data.bin -o path/to/output.json

## Semantic mapping
- Pretty self explainatory
- Used Mask2Former through Huggingface's transformer plugin
- Pretrained model from facebook (mask2former-swin-large-mapillary-vistas-semantic) to obtain the per pixel confidence maps
- First part just use the 1st image and test it
- Second part process the whole directory
- Output in semantic mapping in both binaries and images format
