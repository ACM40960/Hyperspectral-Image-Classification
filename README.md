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

The raw hyperspectral data undergoes three key steps to improve quality and reduce computational load:

1. **Radiometric & Atmospheric Correction**  
   Convert raw sensor values to surface reflectance to remove sensor and atmospheric effects:
   $$
   R(\lambda) = \frac{I_{\text{raw}}(\lambda) - I_{\text{dark}}(\lambda)}{I_{\text{white}}(\lambda) - I_{\text{dark}}(\lambda)}
   $$
   where \(I_{\text{white}}\) and \(I_{\text{dark}}\) are calibration references.

2. **Geometric Alignment (Band Registration)**  
   Align each spectral band to a common spatial grid so the same \((x,y)\) refers to the same ground location:
   $$
   I_{\text{aligned}}(x,y,\lambda) = T_\lambda\!\big(I(x,y,\lambda)\big)
   $$
   with \(T_\lambda\) the estimated geometric transform for band \(\lambda\).

3. **Dimensionality Reduction (PCA)**  
   Reduce redundancy across bands by projecting onto the top \(k\) principal components.  
   Given centered data matrix \(X \in \mathbb{R}^{n \times d}\):
   $$
   C = \frac{1}{n-1}X^\top X,\quad C v_i = \lambda_i v_i,\quad
   Z = X V_k
   $$
   where \(V_k = [v_1,\dots,v_k]\). \(Z \in \mathbb{R}^{n \times k}\) retains the most informative spectral variation.### b. Preprocessing

The raw hyperspectral data undergoes three key steps to improve quality and reduce computational load.

#### 1) Radiometric & Atmospheric Correction
Convert raw sensor values to surface reflectance to remove sensor and atmospheric effects:

$$
R(\lambda)
= \frac{I_{\text{raw}}(\lambda) - I_{\text{dark}}(\lambda)}
       {I_{\text{white}}(\lambda) - I_{\text{dark}}(\lambda)}
$$

where \(I_{\text{white}}\) and \(I_{\text{dark}}\) are calibration references.

---

#### 2) Geometric Alignment (Band Registration)
Align each spectral band to a common spatial grid so the same \((x,y)\) refers to the same ground location:

$$
I_{\text{aligned}}(x,y,\lambda) \;=\; T_{\lambda}\!\big(I(x,y,\lambda)\big)
$$

with \(T_{\lambda}\) the estimated geometric transform for band \(\lambda\).

---

#### 3) Dimensionality Reduction (PCA)
Reduce redundancy across bands by projecting onto the top \(k\) principal components.

Given the mean-centered data matrix \(X \in \mathbb{R}^{n \times d}\):

$$
C \;=\; \frac{1}{n-1}\, X^{\top} X
$$

Eigen-decomposition of the covariance:

$$
C\,v_i \;=\; \lambda_i\, v_i
$$

Project onto the top-\(k\) eigenvectors \(V_k = [v_1,\dots,v_k]\):

$$
Z \;=\; X\,V_k
$$


**Images to add:**
- `images/preprocessing_flow.png` – correction → registration → PCA.
- `images/pca_variance.png` – scree plot (explained variance vs. component).
- `images/pca_rgb.png` – pseudo-RGB from first 3 PCs.

---

### c. Feature Extraction

We learn spectral–spatial features using convolutional neural networks:

- **2D CNN (spatial-only kernels per band/stack):**
  $$
  y_{i,j}^{(k)} = f\!\left(\sum_{m}\sum_{u,v} x_{i+u,\,j+v}^{(m)}\, w_{u,v}^{(m,k)} + b^{(k)}\right)
  $$
  captures local spatial patterns.

- **3D CNN (joint spectral–spatial kernels):**
  $$
  y_{x,y,z}^{(k)} = f\!\left(\sum_{i=0}^{D-1}\sum_{j=0}^{H-1}\sum_{\ell=0}^{W-1}
  x_{x+i,\,y+j,\,z+\ell}\, w_{i,j,\ell}^{(k)} + b^{(k)}\right)
  $$
  captures correlations across height–width–bands simultaneously.

Typical block: **Conv → ReLU → (BatchNorm) → Pool** on PCA patches; optional **attention** weights re-scale channels/regions to emphasize informative spectra or neighborhoods.

**Images to add:**
- `images/cnn_block.png` – 2D vs 3D conv blocks.
- `images/patch_extraction.png` – PCA patch extraction from Indian Pines/Salinas.
- `images/attention_sketch.png` – simple channel/spatial attention sketch (optional).

---

### d. Classification

Convolutional features are mapped to land-cover classes:

1. **Pooling** (e.g., max pooling) reduces spatial size while retaining saliency:
   $$
   y_{i,j}^{(k)}=\max_{(u,v)\in\Omega} x_{i+u,\,j+v}^{(k)}
   $$

2. **Fully Connected layer** on flattened features:
   $$
   h = \sigma(Wx + b)
   $$

3. **Softmax** outputs class probabilities over \(K\) classes:
   $$
   P(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
   $$

4. **Training objective (categorical cross-entropy)**:
   $$
   \mathcal{L} = -\sum_{i=1}^{N}\sum_{k=1}^{K} y_{i,k}\,\log\big(\hat y_{i,k}\big)
   $$

**Images to add:**
- `images/classification_pipeline.png` – features → FC → softmax.
- `images/sample_classification_map.png` – predicted map example.

---

### e. Post-Processing & Evaluation

**Post-processing.**  
Apply morphological filtering or majority voting to remove isolated misclassifications and enforce spatial coherence.

**Metrics.**  
We report pixel-level and class-wise performance:

- **Overall Accuracy (OA):**
  $$
  \text{OA}=\frac{\sum_{i=1}^{N}\mathbf{1}\{y_i=\hat y_i\}}{N}
  $$

- **Average Accuracy (AA):**
  $$
  \text{AA}=\frac{1}{K}\sum_{k=1}^{K}\frac{TP_k}{TP_k+FN_k}
  $$

- **Precision / Recall / F1 (per class):**
  $$
  \text{Precision}=\frac{TP}{TP+FP},\qquad
  \text{Recall}=\frac{TP}{TP+FN},\qquad
  F1=2\cdot\frac{\text{Prec}\cdot\text{Rec}}{\text{Prec}+\text{Rec}}
  $$

- **Cohen’s Kappa (\(\kappa\))** adjusts for chance agreement:
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e}
  $$
  where \(p_o\) is observed accuracy and \(p_e\) is expected accuracy by chance.

- **Sensitivity analysis:** vary #PCs, CNN depth/filter sizes, and training sample size to test robustness.

**Images to add:**
- `images/confusion_matrix.png` – confusion matrix heatmap.
- `images/oa_aa_kappa.png` – bar chart of OA/AA/κ.
- `images/groundtruth_vs_pred.png` – ground truth vs predicted maps.
- `images/sensitivity_plots.png` – OA vs #PCs / depth curves.
 

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

