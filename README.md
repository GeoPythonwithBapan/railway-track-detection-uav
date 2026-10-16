<p align="center">
  <img src="assets/railway_banner.png"
       alt="UAV-Based Railway Track Detection"
       width="100%">
</p>

<h1 align="center">🚆 Railway Track Detection from UAV Orthomosaic</h1>

<p align="center">
  <strong>
    Deep Learning | Computer Vision | Remote Sensing | GIS Automation
  </strong>
</p>

<p align="center">
  UAV-based railway rail extraction and track centerline generation
  using deep learning and GIS.
</p>

---

## 📌 Project Overview

This project presents an end-to-end workflow for extracting railway rails from high-resolution UAV orthomosaic imagery and generating railway track centerlines.

The workflow combines deep learning, computer vision, raster processing, and GIS automation to reduce repetitive manual digitization and produce GIS-ready railway geometries.

### Key Features

- 🛰️ UAV orthomosaic processing
- 🧩 Image tiling and dataset preparation
- ✏️ QGIS-based rail annotation
- 🎭 Binary mask generation
- 🧠 Lite-RSNet-inspired semantic segmentation
- 🔍 Full-orthomosaic rail prediction
- 🛠️ Raster post-processing and skeletonization
- 📐 Rail vectorization and geometry repair
- 🚆 Railway rail pairing and centerline generation
- 🗺️ GeoPackage output for GIS applications

---

## ⚙️ Project Workflow

The complete workflow follows these stages:

```text
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
```

---

## 🖼️ Before and After Processing

This section demonstrates the railway rail extraction process, from the original or raw prediction output to the processed rail lines and generated track centerline.

<!-- Replace these example filenames with your actual screenshot filenames. -->

<table>
  <tr>
    <th align="center">Before Processing</th>
    <th align="center">After Processing</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/before_processing.png"
           alt="Railway extraction before processing"
           width="100%">
    </td>
    <td align="center">
      <img src="assets/after_processing.png"
           alt="Railway extraction after processing"
           width="100%">
    </td>
  </tr>
</table>

*The comparison illustrates the rail extraction and post-processing workflow. Actual results depend on image quality, annotation coverage, model performance, and post-processing parameters.*

---

## 🎯 Final Output

The workflow produces a GeoPackage (`.gpkg`) containing the following layers:

| Layer | Description |
|---|---|
| `Railway Track` | Identified railway rail pairs |
| `Railway Centerline` | Generated centerlines between paired rails |

The centerline layer includes track identification and length information.

The output can be opened in QGIS for visualization, inspection, and further geospatial analysis.

---

## 🧰 Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Raster Processing | Rasterio, NumPy, SciPy |
| Vector GIS | GeoPandas, Shapely |
| Image Processing | OpenCV, scikit-image |
| Skeleton Analysis | Skan |
| GIS Software | QGIS |
| Data Format | GeoTIFF, GeoPackage |

---

## 📁 Repository Structure

```text
railway-track-detection-uav/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   ├── railway_banner.png
│   ├── before_processing.png
│   └── after_processing.png
│
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── dataset/
│   ├── training/
│   ├── prediction/
│   ├── postprocessing/
│   ├── vectorization/
│   ├── repair/
│   └── centerline/
│
├── docs/
│   └── workflow_documentation.md
│
└── examples/
```

---

## 🚀 Project Status

**Status:** End-to-end prototype workflow completed.

The workflow covers model training, rail prediction, raster post-processing, vectorization, rail-line repair, and centerline generation.

Further improvements may focus on prediction accuracy, rail continuity, curved-track handling, rail pairing, and centerline geometric accuracy.

---

## 👨‍💻 Author

**Bapan Dutta**

GIS | Remote Sensing | UAV | AutoCAD | AutoLISP | GIS/CAD Automation
