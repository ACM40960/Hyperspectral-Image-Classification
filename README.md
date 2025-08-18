## 🌍 Hyperspectral Image Classification using 2D and 3D CNNs

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

This study implements a deep learning–based pipeline for hyperspectral image (HSI) classification using the **Indian Pines dataset** (145 × 145 pixels, 200 spectral bands). The methodology is divided into **two major stages**:  

1. **Preprocessing** – to improve data quality and reduce redundancy.  
2. **CNN Model Training & Classification** – to learn spectral–spatial representations and classify pixels.  

---

### **Overview of the Approach**  

Hyperspectral images capture hundreds of spectral bands per pixel. While this richness enables precise identification of materials, it also introduces the challenge of managing **high-dimensional, redundant, and noisy data**.  

To address this, our approach combines:  

- **Principal Component Analysis (PCA)** → reduces dimensionality efficiently while preserving critical spectral information.  
- **Convolutional Neural Networks (CNNs)** with **spatial attention mechanisms** → learn robust spectral-spatial features.  
- **Post-processing techniques** → smoothing filters to refine classification maps.  

This fusion of spectral reduction and spatial learning ensures the system is both **computationally efficient** and **accurate**.  

📌 *Image to Add:*  
`images/overview_pipeline.png` (Workflow diagram: Input Hyperspectral Cube → PCA → CNN + Attention → Classification → Post-processing → Final Map).  

---

### **1. Data Collection**  

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

## **2. Preprocessing**

Before training the model, the hyperspectral data needed to be cleaned and simplified so that it could be used effectively. Raw hyperspectral images usually contain hundreds of spectral bands, but many of these are either redundant or affected by atmospheric noise and sensor distortions. To tackle this, we first applied radiometric and atmospheric corrections, which convert the raw sensor values into surface reflectance. This ensures that the dataset represents real-world ground conditions rather than being influenced by lighting variations or sensor errors.

Once the data was corrected, we focused on reducing its very high dimensionality. Using Principal Component Analysis (PCA), we were able to compress the dataset while still keeping more than 99% of the useful information. This step not only removed unnecessary bands but also made the training process much faster and more efficient. To prepare the data for the CNN models, we extracted fixed-size patches around each pixel and applied data augmentation techniques like rotations and flips. This helped balance the dataset by giving more representation to smaller classes, ensuring that the model could learn fairly across all land-cover types.



### 2.1 Radiometric & Atmospheric Correction
Convert raw sensor values to surface reflectance to remove sensor and atmospheric effects:  

$$
R(\lambda) = \frac{I_{\text{raw}}(\lambda) - I_{\text{dark}}(\lambda)}{I_{\text{white}}(\lambda) - I_{\text{dark}}(\lambda)}
$$

where $$\(I_{\text{white}}\)$$ and $$\(I_{\text{dark}}\)$$ are calibration references.

Radiometric and atmospheric correction is a crucial first step in hyperspectral image preprocessing. Raw hyperspectral data is often influenced by sensor noise, illumination variations, and atmospheric scattering, which can distort the spectral signatures of materials. To ensure that the measured values truly represent ground reflectance, calibration using white and dark reference panels is applied.  


---

### 2.2 Geometric Alignment (Band Registration)
Each spectral band is aligned to a common spatial grid so that the same \((x,y)\) refers to the same ground location:  

$$
I_{\text{aligned}}(x,y,\lambda) = T_{\lambda}\big(I(x,y,\lambda)\big)
$$

where $$\(T_{\lambda}\)$$ is the estimated geometric transform for band \(\lambda\).

This step ensures that every spectral band is perfectly aligned so that each pixel location $$\((x, y)\)$$ refers to the exact same ground point across all wavelengths. Without alignment, even small distortions between bands could confuse the model and lead to misclassification. By applying geometric transformations, we create a consistent hyperspectral cube where both spectral and spatial information are accurately preserved for further processing.


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

Dimensionality reduction with PCA is applied to tackle the “curse of dimensionality” in hyperspectral data. Since many adjacent spectral bands carry redundant information, PCA projects the original high-dimensional data onto a smaller set of uncorrelated components that capture the majority of variance. This not only reduces computational complexity but also suppresses noise, making the features more robust for classification. In our project, retaining the top principal components preserves essential spectral-spatial details while significantly improving efficiency for downstream CNN training.

---

## **2. CNN Model Architecture**

In our project, two types of convolutional neural networks (CNNs) were explored to capture the spectral–spatial characteristics of hyperspectral images.  

 - **2D CNN** – extracts spatial features per band independently.  

The **2D CNN** was applied on spectral bands after dimensionality reduction using PCA. Each input patch was represented as a 2D image with reduced channels, and the convolutional layers learned spatial textures and edge-like patterns within each band. This approach is computationally less intensive and works well when spectral redundancy has already been reduced, making it suitable for large-scale experiments.  

- **3D CNN** – captures joint spectral–spatial correlations.
  
The **3D CNN**, in contrast, directly processed the hyperspectral data cube by treating both the spectral and spatial dimensions jointly. Instead of analyzing each band independently, the 3D kernels convolved across neighboring pixels and spectral bands simultaneously, enabling the network to capture subtle correlations between wavelength variations and spatial structures. This made the 3D CNN particularly powerful in learning complex class boundaries where spectral signatures alone were insufficient.  

By training both models under the same framework with categorical cross-entropy loss and an SGD optimizer, we could compare their effectiveness. The 2D CNN achieved faster training and lower computational cost, while the 3D CNN produced richer feature representations and higher classification accuracy, especially in spectrally complex regions.


---

### Step 2.1: Convolution Layer (Feature Extraction)

In our project, the convolution layer acted as the fundamental feature extractor for hyperspectral images. 

For **2D CNN**:

$$
Y(i,j,k) = \sigma \Bigg( \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} \sum_{c=0}^{C-1} 
X(i+m, j+n, c) \cdot W(m,n,c,k) + b_k \Bigg)
$$

In our project, the convolution layer acted as the fundamental feature extractor for hyperspectral images. For the **2D CNN**, the convolution was applied on local pixel neighborhoods within each PCA-reduced band. This helped the model learn spatial textures such as edges, shapes, and fine details across the scene, which are important for distinguishing land-cover classes. On the other hand, 

For **3D CNN**:

$$
Y(x,y,z,k) = \sigma \Bigg( \sum_{d=0}^{D-1} \sum_{j=0}^{H-1} \sum_{i=0}^{W-1} 
X(x+i, y+j, z+l) \cdot W(i,j,l,k) + b_k \Bigg)
$$

the **3D CNN** extended this operation to the spectral dimension, where filters scanned across both space and wavelength simultaneously. This allowed the network to capture subtle correlations between spectral signatures and spatial patterns, leading to richer feature maps. Together, these convolution operations ensured that the models could automatically learn discriminative features without relying on handcrafted descriptors.

---

### Step 2.2: Pooling Layer (Downsampling)
 In our project, the pooling layer was introduced to progressively reduce the spatial resolution of the feature maps while retaining the most important information. By applying max-pooling, the model preserved the strongest activations that represent dominant spectral–spatial patterns, while discarding redundant or less significant details. This not only reduced computational complexity but also improved robustness to small variations such as noise or slight misalignments in the hyperspectral data. The pooling operation therefore helped our CNN models focus on the most discriminative features required for accurate land-cover classification.

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

In our project, the fully connected layer acted as the final decision-making stage of the CNN. After convolution and pooling, the extracted spectral–spatial features were flattened into a one-dimensional vector and passed through dense connections. This enabled the network to combine information across all bands and spatial regions, effectively learning high-level class representations. The fully connected layer mapped these features to the output classes (e.g., different land-cover types in the Indian Pines dataset), ensuring that the network could perform accurate pixel-wise classification.

---

### Step 2.4: Softmax Classifier

Outputs class probabilities:

$$
P(y = k \mid x) = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}
$$

In our project, the softmax layer was used as the final classification stage. It converted the raw output scores (logits) from the fully connected layer into normalized probability distributions across all land-cover classes. This allowed each pixel in the hyperspectral image to be assigned to the most likely class, while still providing probability estimates for all possible categories. By using softmax, the model not only predicted the most probable class but also provided confidence levels, which is crucial for evaluating uncertainty in hyperspectral image classification tasks.


---

### Step 2.5: Cross-Entropy Loss

Optimization objective for training:

$$
L = - \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \; \log \hat{y}_{i,k}
$$

For training our CNN models, we employed the cross-entropy loss as the optimization objective. This loss function compares the predicted probability distribution with the true class labels, penalizing the model more heavily when the predicted probability for the correct class is low. By minimizing cross-entropy loss, the network effectively learns to maximize the likelihood of correctly classifying each pixel. In our project, this was particularly important given the large number of classes in hyperspectral datasets like Indian Pines, ensuring that the classifier became more discriminative and accurate over successive training epochs.

---

## **3. Training & Evaluation**

The training process was carefully designed to improve the generalization capability of our CNN models while handling the challenges of hyperspectral data, such as high dimensionality and severe class imbalance.

### 3.1 Data Augmentation  
Hyperspectral datasets like **Indian Pines** often suffer from limited labeled samples per class. To mitigate this, we applied **data augmentation** techniques such as random flips, rotations, and oversampling of minority classes. This not only balanced the dataset but also improved the robustness of the model against variations in spatial orientation and local distortions.

### 3.2 Optimizer – Stochastic Gradient Descent (SGD)  
The model parameters were optimized using **Stochastic Gradient Descent (SGD)**. At each iteration, the weights were updated as:  

$$
w_{t+1} = w_t - \eta \cdot \nabla L(w_t)
$$  

where $$\(w_t\)$$ represents the weight vector at iteration $$\(t\)$$, $$\(\eta\)$$ is the learning rate, and $$\(\nabla L(w_t)\)$$ is the gradient of the cross-entropy loss with respect to the weights.  
We also experimented with **learning rate scheduling** to ensure faster convergence in the early stages of training and stability during later iterations.

### 3.3 Evaluation Metrics  
To assess performance, multiple metrics were employed:  

- **Overall Accuracy (OA):**  
  
  OA = $$\frac$${\text{Number of correctly classified samples}}{\text{Total number of samples}}
  

- **Average Accuracy (AA):**  
  $$
  AA = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{N_k}
  $$  
  where \(TP_k\) is the number of correctly classified samples in class \(k\), and \(N_k\) is the total number of samples in that class.  

- **Kappa Coefficient (\(\kappa\)):**  
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e}
  $$  
  where \(p_o\) is the observed agreement (OA) and \(p_e\) is the expected agreement by chance.  

- **Precision & Recall:**  
  $$
  \text{Precision} = \frac{TP}{TP + FP}, \quad 
  \text{Recall} = \frac{TP}{TP + FN}
  $$  

- **F1-Score:**  
  $$
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$  

These metrics ensured both **global accuracy** and **class-specific performance** were evaluated, which is crucial in imbalanced hyperspectral datasets.  

### 3.4 Validation  
To validate the model, we generated:  
1. **Confusion Matrix** – displaying per-class accuracies and misclassifications.  
2. **Classified Maps** – comparing predicted labels with ground truth maps for visual interpretation.  

This multi-faceted evaluation enabled us to confirm not only the numerical performance of the models but also their ability to produce spatially coherent classification maps suitable for real-world land-cover analysis.

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

OA = $$\frac{\sum_{i=1}^N \mathbf{1}(y_i = \hat{y}_i)}{N}$$  


📌 *Image:* `images/accuracy_comparison.png`  

### **2. Precision, Recall, F1**  

Precision = $$\frac{TP}{TP+FP}, \quad Recall = \frac{TP}{TP+FN}$$  , $$\quad$$ F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision+Recall}
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

