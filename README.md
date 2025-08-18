### 🌍 Hyperspectral Image Classification  
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

# Methodology

This study implements a deep learning–based pipeline for hyperspectral image (HSI) classification using the **Indian Pines dataset** (145 × 145 pixels, 200 spectral bands). The methodology is divided into **two major stages**:  

1. **Preprocessing** – to improve data quality and reduce redundancy.  
2. **CNN Model Training & Classification** – to learn spectral–spatial representations and classify pixels.  

---

## **1. Preprocessing**

The raw hyperspectral data undergoes three key steps to improve quality and reduce computational load:

### 1.1 Radiometric & Atmospheric Correction
Convert raw sensor values to surface reflectance to remove sensor and atmospheric effects:  

$$
R(\lambda) = \frac{I_{\text{raw}}(\lambda) - I_{\text{dark}}(\lambda)}{I_{\text{white}}(\lambda) - I_{\text{dark}}(\lambda)}
$$

where $$\(I_{\text{white}}\)$$ and $$\(I_{\text{dark}}\)$$ are calibration references.

---

### 1.2 Geometric Alignment (Band Registration)
Each spectral band is aligned to a common spatial grid so that the same \((x,y)\) refers to the same ground location:  

$$
I_{\text{aligned}}(x,y,\lambda) = T_{\lambda}\big(I(x,y,\lambda)\big)
$$

where $$\(T_{\lambda}\)$$ is the estimated geometric transform for band \(\lambda\).

---

### 1.3 Dimensionality Reduction (PCA)
To reduce redundancy, data is projected onto the top \(k\) principal components:  

$$
C = \frac{1}{n-1} X^\top X, \quad C v_i = \lambda_i v_i
$$

$$
Z = X V_k
$$

where $$\(X \in \mathbb{R}^{n \times d}\)$$ is the data matrix, $$\(V_k = [v_1, v_2, \dots, v_k]\)$$, and $$\(Z \in \mathbb{R}^{n \times k}\)$$ retains the most informative spectral variation.

---

## **2. CNN Model Architecture**

Two architectures were implemented:  
- **2D CNN** – extracts spatial features per band independently.  
- **3D CNN** – captures joint spectral–spatial correlations.  

Both models were trained using **categorical cross-entropy loss** with the **SGD optimizer**.

---

### Step 2.1: Convolution Layer (Feature Extraction)

For **2D CNN**:

$$
Y(i,j,k) = \sigma \Bigg( \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} \sum_{c=0}^{C-1} 
X(i+m, j+n, c) \cdot W(m,n,c,k) + b_k \Bigg)
$$

For **3D CNN**:

$$
Y(x,y,z,k) = \sigma \Bigg( \sum_{d=0}^{D-1} \sum_{j=0}^{H-1} \sum_{i=0}^{W-1} 
X(x+i, y+j, z+l) \cdot W(i,j,l,k) + b_k \Bigg)
$$

---

### Step 2.2: Pooling Layer (Downsampling)

Reduces spatial size while retaining key features:

$$
Y(i,j,k) = \max_{(m,n)\in \Omega} \; X(i+m, j+n, k)
$$

---

### Step 2.3: Fully Connected Layer

Flattens features for classification:

$$
y = \sigma ( W \cdot x + b )
$$

---

### Step 2.4: Softmax Classifier

Outputs class probabilities:

$$
P(y = k \mid x) = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}
$$

---

### Step 2.5: Cross-Entropy Loss

Optimization objective for training:

$$
L = - \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \; \log \hat{y}_{i,k}
$$

---

## **3. Training & Evaluation**

- **Data Augmentation**: Flips, rotations, and oversampling were applied to address class imbalance.  
- **Optimizer**: Stochastic Gradient Descent (SGD).  
- **Evaluation Metrics**: Overall Accuracy (OA), Average Accuracy (AA), Kappa Coefficient, Precision, Recall, F1-score.  
- **Validation**: Confusion matrix and classification maps were generated to compare predictions with ground truth.  

---

✅ This methodology ensures that preprocessing improves data quality, CNNs capture both spectral and spatial dependencies, and robust evaluation metrics validate classification performance.


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

