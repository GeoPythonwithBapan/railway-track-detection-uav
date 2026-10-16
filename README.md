# Railway Track Detection from UAV Orthomosaic

UAV-based railway rail extraction and track centerline generation using deep learning and GIS.

## Project Overview

This project presents an end-to-end workflow for extracting railway rails from high-resolution UAV orthomosaic imagery.

The workflow combines:

- UAV orthomosaic processing
- Image tiling
- QGIS-based rail annotation
- Binary mask generation
- Lite-RSNet-inspired semantic segmentation
- Full-orthomosaic prediction
- Raster post-processing
- Skeletonization
- Rail vectorization
- Rail line repair
- Rail pairing
- Railway track centerline generation

## Workflow

UAV Orthomosaic  
↓  
Image Tiling  
↓  
Rail Annotation  
↓  
Binary Mask Generation  
↓  
Lite-RSNet Training  
↓  
Full Image Prediction  
↓  
Rail Mask Post-processing  
↓  
Skeletonization  
↓  
Rail Vectorization  
↓  
Rail Line Repair  
↓  
Rail Pairing  
↓  
Track Centerline Generation  
↓  
Final GIS Output

## Final Output

The workflow produces a GeoPackage containing:

- `Railway Track`
- `Railway Centerline`

The centerline layer includes track identification and length information.

## Technologies

Python | PyTorch | Rasterio | GeoPandas | Shapely | NumPy | SciPy | scikit-image | Skan | QGIS

## Status

Completed end-to-end prototype workflow.



---

## 👨‍💻 Author

**Bapan Dutta**

GIS | Remote Sensing | UAV | AutoCAD | AutoLISP | GIS/CAD Automation

