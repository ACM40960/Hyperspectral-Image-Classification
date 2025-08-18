# 🌍 Hyperspectral Image Classification  
*Feature Extraction & Classification for Urban Natural Resource Monitoring*  

---

## 📖 Project Overview  

Hyperspectral imaging records a complete spectrum at every pixel, creating a three-dimensional data cube with hundreds of narrow wavelength bands. Unlike conventional RGB images, this wealth of spectral detail provides a unique material signature, enabling the accurate detection and classification of land-cover types such as vegetation, soil, water, and urban structures. Such fine-grained mapping makes hyperspectral imaging a valuable tool for monitoring natural resources and supporting urban and environmental planning.  

This project focuses on **feature extraction and classification** of hyperspectral datasets, including *Indian Pines* and *Salinas*. The proposed pipeline incorporates essential **preprocessing steps** (radiometric and atmospheric corrections), **dimensionality reduction** through Principal Component Analysis (PCA), and **feature learning** using Convolutional Neural Networks (CNNs) with spatial attention mechanisms. By integrating both spectral and spatial characteristics, the model generates high-quality pixel-level classification maps that are both accurate and interpretable.  

The system further employs **post-processing filters** to refine predictions and evaluates performance using key metrics such as **overall accuracy, per-class precision, recall, and the Kappa coefficient**. By combining spectral and spatial information, the project demonstrates how advanced machine learning can transform raw hyperspectral data into actionable insights. Beyond strong results on benchmark datasets, this approach highlights the broader potential of hyperspectral imaging for **urban planning, resource management, and environmental monitoring**.  

---

## 📝 Problem Statement  

The goal of this project is to perform **feature extraction and classification of hyperspectral images** in order to provide reliable information for monitoring different natural resources in urban areas. Publicly available benchmark datasets such as *Indian Pines* and *Salinas* are used to evaluate and validate the proposed approach.  

---

## 🎯 Motivation  

Conventional images capture only three colour channels (RGB), which often miss subtle material differences. Hyperspectral imaging records **hundreds of narrow spectral bands** at each pixel, uncovering details that are invisible in standard photos.  

By reducing noise and extracting the most informative features, similar pixels can be grouped and classified into categories such as **vegetation, water, or pavement**. This enables fast and accurate mapping of urban natural resources, supporting planners, scientists, and policymakers in **environmental monitoring and informed decision-making**.  

---

## 🎯 Objectives  

In this project, we set out to:  
1. Study and compare different ways to pick out useful features from hyperspectral images.  
2. Try methods to reduce the huge amount of data into a smaller, meaningful set without losing key information.  
3. Look at different algorithms that can classify land-cover types and check which ones work best.  
4. Build and test models (like machine learning and deep learning) to classify pixels in the images.  
5. Clearly define the problem of hyperspectral image classification for urban areas.  
6. Explore advanced techniques like using neighbourhood similarity and mixing spectral features.  
7. Use well-known datasets (*Indian Pines, Salinas*) to test our methods.  
8. Check performance carefully, measuring accuracy, precision/recall, and overall reliability.  

---

## ⚙️ Methodology  

Our methodology follows a structured pipeline, starting from raw hyperspectral data acquisition to generating accurate land-cover classification maps. The process involves **preprocessing, feature extraction, classification, and post-processing with evaluation metrics**.  

---

### **1. Overview of the Approach**  

Hyperspectral images capture hundreds of spectral bands per pixel. While this richness enables precise identification of materials, it also introduces the challenge of managing **high-dimensional, redundant, and noisy data**.  

To address this, our approach combines:  
- **Principal Component Analysis (PCA)** → reduces dimensionality efficiently while preserving critical spectral information.  
- **Convolutional Neural Networks (CNNs)** with **spatial attention mechanisms** → learn robust spectral-spatial features.  
- **Post-processing techniques** → smoothing filters to refine classification maps.  

This fusion of spectral reduction and spatial learning ensures the system is both **computationally efficient** and **accurate**.  

📌 *Image to Add:* `images/overview_pipeline.png` (Workflow diagram: Input Hyperspectral Cube → PCA → CNN + Attention → Classification → Post-processing → Final Map).  

---

### **a. Data Collection**  

We use well-established benchmark hyperspectral datasets:  

- **Indian Pines Dataset**  
  - AVIRIS sensor, northwest Indiana.  
  - 224 spectral bands (0.4–2.5 μm).  
  - 16 ground-truth classes (crops, grass, woods, urban).  

- **Salinas Scene Dataset**  
  - AVIRIS sensor, Salinas Valley, California.  
  - 224 bands at 3.7 m resolution.  
  - 16 classes (vegetation, crops, soil).  

- **AVIRIS Urban Scenes**  
  - Urban hyperspectral cubes.  
  - Includes man-made (asphalt, rooftops) and natural features.  

📌 *Images to Add:*  
- `images/indian_pines_rgb.png` (Indian Pines pseudo-RGB).  
- `images/salinas_rgb.png` (Salinas pseudo-RGB).  
- `images/dataset_groundtruth.png` (Ground truth maps).  

---

## Methodology

### b. Preprocessing  

The raw hyperspectral data undergoes several key preprocessing steps:

- **Radiometric & Atmospheric Correction**  
  Eliminates distortions caused by sensor noise and atmospheric effects.  

- **Geometric Alignment**  
  Ensures that spectral bands are spatially consistent across the image.  

- **Dimensionality Reduction (PCA)**  
  Principal Component Analysis is applied to extract the most informative features while reducing redundancy.  

<!-- LaTeX reference
$$ Z = XW, \quad W = \arg\max_W \; \det(W^T \Sigma W) $$
-->

![PCA Equation](images/pca_equation.png)

*Images to Add:*  
- `images/preprocessing_flow.png` (overall preprocessing flow).  
- `images/pca_visual.png` (PCA variance captured vs components).  

---

### c. Feature Extraction  

We apply **convolutional layers** to capture both spectral and spatial context.  

<!-- LaTeX reference
$$ y_{i,j}^k = f \left( \sum_m \sum_{u,v} x_{i+u,j+v}^m \cdot w_{u,v}^{m,k} + b^k \right) $$
-->

![Convolution Equation](images/conv_equation.png)

*Images to Add:*  
- `images/cnn_block.png` (CNN block diagram).  
- `images/feature_extraction.png` (spectral vs spatial features).  

---

### d. Classification  

1. **Softmax Layer**  
   Converts the extracted features into probability distributions over classes.  

<!-- LaTeX reference
$$ P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} $$
-->

![Softmax Equation](images/softmax.png)

---

2. **Loss Function (Cross-Entropy)**  
   Optimizes the network by minimizing the error between predicted and true labels.  

<!-- LaTeX reference
$$ L = - \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log \hat{y}_{i,k} $$
-->

![Cross-Entropy Equation](images/cross_entropy.png)

*Images to Add:*  
- `images/classification_pipeline.png` (end-to-end classification process).  

---

### e. Post-Processing & Evaluation  

- **Smoothing Filters** are applied to reduce noise in the classified map.  
- **Evaluation Metrics** include accuracy, precision, recall, and Kappa coefficient.  

<!-- LaTeX reference
Accuracy: $$ Acc = \frac{TP + TN}{TP + TN + FP + FN} $$
Kappa: $$ \kappa = \frac{p_o - p_e}{1 - p_e} $$
-->

![Accuracy Equation](images/accuracy_equation.png)  
![Kappa Equation](images/kappa_equation.png)

*Images to Add:*  
- `images/classified_map.png` (sample classified land-cover map).  
- `images/confusion_matrix.png` (evaluation matrix).  

## 📂 Datasets Used  

- **Indian Pines** – 224 bands, 16 classes.  
- **Salinas Scene** – 224 bands, high-res, 16 classes.  
- **AVIRIS Urban Scenes** – Urban cubes with mixed features.  

Ground truth is available for all datasets, enabling **training, validation, and testing**.  

---

## 📊 Results & Evaluation  

### **1. Accuracy Metrics**  
- **OA, AA, Kappa** used to validate models.  
$$
OA = \frac{\sum_{i=1}^N \mathbf{1}(y_i = \hat{y}_i)}{N}
$$  

📌 *Image:* `images/accuracy_comparison.png`  

### **2. Precision, Recall, F1**  
$$
Precision = \frac{TP}{TP+FP}, \quad Recall = \frac{TP}{TP+FN}, \quad F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision+Recall}
$$  

📌 *Image:* `images/confusion_matrix.png`  

### **3. Sensitivity Analysis**  
- Varying PCA components, CNN depth, and training sample size.  
- Showed stability of results under different configurations.  

📌 *Image:* `images/sensitivity_analysis.png`  

### **4. Visual Results**  
- Ground truth vs predicted classification maps.  
- Overlay of predictions on original hyperspectral cube.  

📌 *Image:* `images/visual_results.png`  

---

## 📌 Project Structure  

