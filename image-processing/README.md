# Image Processing
## Getting Kitti360 dataset
images: https://www.kaggle.com/datasets/greatgamedota/kitti-360-0000?resource=download-directory
other stuff: https://www.cvlibs.net/datasets/kitti-360/download.php (u need to create an account)

## Extraction of intrinsics & extrinsics
### OpenMVG
- complied and tested on x64 Ubuntu 22.04

#### Procedures
##### 1. cd to openMVG_Build\software\SfM

##### 2. Editing ./SfM_SequentialPipeline.py
- change OPENMVG_SFM_BIN, CAMERA_SENSOR_WIDTH_DIRECTORY to ur dir
  - change the argument "-f" to 1.2 * width/height (whichever is larger) as initial focal length
- run in terminal ./SfM_SequentialPipeline.py <images> <output>
  - (.\Reduced_ImageDataset_Kitti360\images) (.\\Reduced_ImageDataset_Kitti360\recon)

##### 3. Converting sfm.bin to sfm.json for human readability
- located in \Reduced_ImageDataset_Kitti360\recon\reconstruction_sequential\
- contains the results
##### 4. cd to \openMVG_Build\Linux-x86_64-RELEASE
##### 5. run ./openMVG_main_ConvertSfM_DataFormat -i path/to/sfm_data.bin -o path/to/output.json

#### intrinsics format
- "focal_length": 550.9857544276782,
- "principal_point": [
 681.1552850314727,
 237.91637654378585
],

#### extrinsics format
- "rotation": [
[0.99507, 0.06208, -0.0773],
[-0.06376, 0.99777, -0.0194],
[0.07595, 0.02426, 0.99681]
],
- "center": [
0.4220822064007432,
0.03939894611747139,
0.5839133704325018
]

## Semantic mapping
### Architecture
- Used Mask2Former through Huggingface's transformer plugin
  - Pretrained model from facebook (mask2former-swin-large-mapillary-vistas-semantic) to obtain the per pixel confidence maps
- Gather outputs from the model pipeline
  - Per-pixel confidence map
  - Per-pixel class IDs map
- Visualize them as an overlay of original image
- Output format
  - Saved semantics maps using numpy's save function
  - images as .pngs 
