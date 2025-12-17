# Urban_Tree_Canopy_Detection
Tree canopy detection in urban area using NAIP imagery, Random Forest classification, and deep learning segmentation.


### Data Source

- **National Agriculture Imagery Program (NAIP)**
- **4-band imagery**: Red, Green, Blue, Near-Infrared (RGB + NIR)
- **Spatial resolution**: ~1 meter
- **Study area**: One U.S. city, covered by 8 NAIP tiles

## NDVI-Based Vegetation Mask
This project begins with a baseline vegetation detection approach using the Normalized Difference Vegetation Index **(NDVI)**. 

#### Notebook
`get_binarymask_ndvi.ipynb`

#### What it does
- Loads NAIP raster tiles
- Computes NDVI using the Red and Near-Infrared (NIR) bands
- Applies a threshold to create a binary vegetation mask
- Saves per-tile NDVI rasters and vegetation masks

#### Outputs
- Stored in `get_binarymask_outputs/`
- Includes:
  - NDVI GeoTIFFs
  - Binary vegetation mask GeoTIFFs
- A stitched, city-wide visualization:
  - `Redlands_NDVI_May2022_NAIP.png`
 
This step provides an interpretable but coarse baseline for vegetation detection.

## Manual Labeling

#### Patch Extraction
- A single NAIP tile was selected
- The tile was divided into **512 × 512 pixel patches**

#### Manual Labeling
- **12 patches** were manually labeled in **QGIS** (GUI-based workflow)
- Binary classification scheme:
  - **1** = Tree canopy
  - **0** = Non-tree

#### Labeled Data Location

data/patches/labeled/

## Random Forest Tree Canopy Classifier

A **Random Forest** classifier was trained using a portion of the manually labeled patches as well as **pixel-level features** derived from the labeled patches.

#### Script
- `rf_canopy_classifier.py`

#### Model Details
- **Input features:**
  - NAIP spectral bands (RGB + NIR)
  - Vegetation indices (e.g., NDVI and related indices)
- **Training data:**
  - Pixels from 12 manually labeled patches (80% training set, 20% test set)
- **Output:**
  - Binary masks of tree canopy prediction

#### Outputs
- **Predicted masks** for all NAIP patches:
  - `data/patches/rf_predictions/`
- **City-wide stitched visualization:**
  - `treeCanopy_RFprediction.png`

This step enables **semi-automatic labeling at scale**, though predictions are spatially noisy due to pixel-based classification.

## Current Status and Next Steps

#### Current Focus
- Random Forest diagnostics
- Error analysis
- Identifying failure modes (e.g., speckle noise, edge confusion)

#### Planned Next Step
- Transition to **U-Net** for semantic segmentation
- Use:
  - Manually labeled patches
  - Carefully selected RF-predicted patches as weak labels
- Goal:
  - Improve spatial coherence and object-level canopy detection


  
