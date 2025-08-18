Project Overview
Hyperspectral imaging records a complete spectrum at every pixel, creating a three dimensional data cube with hundreds of narrow wavelength bands. Unlike conventional RGB images, this wealth of spectral detail provides a unique material signature, enabling the accurate detection and classification of land-cover types such as vegetation, soil, water, and urban structures. Such fine grained mapping makes hyperspectral imaging a valuable tool for monitoring natural resources and supporting urban and environmental planning.
This project focuses on feature extraction and classification of hyperspectral datasets, including Indian Pines and Salinas. The proposed pipeline incorporates essential pre-processing steps (radiometric and atmospheric corrections), dimensionality reduction through Principal Component Analysis (PCA), and feature learning using Convolutional Neural Networks (CNNs) with spatial attention mechanisms. By integrating both spectral and spatial characteristics, the model generates high quality pixel level classification maps that are both accurate and interpretable.
The system further employs post processing filters to refine predictions and evaluates performance using key metrics such as overall accuracy, per class precision, recall, and the Kappa coefficient. By combining spectral and spatial information, the project demonstrates how advanced machine learning can transform raw hyperspectral data into actionable insights. Beyond strong results on benchmark datasets, this approach highlights the broader potential of hyperspectral imaging for real world applications including urban planning, resource management, and environmental monitoring.
Problem Statement
The goal of this project is to perform feature extraction and classification of hyperspectral images in order to provide reliable information for monitoring different natural resources in urban areas. Publicly available benchmark datasets such as Indian Pines and Salinas are used to evaluate and validate the proposed approach.
Motivation
Conventional images capture only three colour channels (RGB), which often miss subtle material differences. Hyperspectral imaging records hundreds of narrow spectral bands at each pixel, uncovering details that are invisible in standard photos.
By reducing noise and extracting the most informative features, similar pixels can be grouped and classified into categories such as vegetation, water, or pavement. This enables fast and accurate mapping of urban natural resources, supporting planners, scientists, and policymakers in environmental monitoring and informed decision-making.
Objectives
In this project, we set out to:
1.	Study and compare different ways to pick out useful features from hyperspectral images.
2.	Try methods to reduce the huge amount of data into a smaller, meaningful set without losing key information.
3.	Look at different algorithms that can classify land-cover types and check which ones work best.
4.	Build and test models (like machine learning and deep learning) to classify pixels in the images.
5.	Clearly define the problem of hyperspectral image classification for urban areas.
6.	Explore advanced techniques like using neighbourhood similarity and mixing spectral features.
7.	Use well-known datasets (Indian Pines, Salinas) to test our methods.
8.	Check performance carefully, measuring accuracy, precision/recall, and overall reliability.
Methodology
1. Overview of the Approach
Hyperspectral images carry a wealth of information by capturing hundreds of spectral bands per pixel. While this richness enables precise identification of materials like vegetation, soil, water, and urban structures, it also introduces the challenge of managing high-dimensional, redundant, and noisy data.
To address this, our approach combines:
•	Principal Component Analysis (PCA) to reduce dimensionality efficiently while preserving critical spectral information.
•	Convolutional Neural Networks (CNNs) enhanced with spatial attention mechanisms to learn robust spectral-spatial features.
•	Post-processing techniques (e.g., smoothing filters) to refine the produced classification maps.
This fusion of spectral reduction and spatial learning ensures that the system remains computationally efficient and delivers accurate, interpretable land-cover mapping for real-world urban environments.
Image to Add: A clear, high-level workflow diagram—Input Hyperspectral Cube → PCA → Attention-augmented CNN → Classification → Post-processing → Final Map.
Filename suggestion: images/overview_pipeline.png

a. Data Collection
For this project, we use well-established benchmark hyperspectral datasets that are widely adopted in the remote sensing and machine learning communities. These datasets not only provide the high-dimensional spectral cubes required for feature extraction but also include ground-truth annotations that allow for robust training and reliable evaluation of classification models.
•	Indian Pines Dataset
o	Captured by the AVIRIS sensor over agricultural and urban areas in northwest Indiana.
o	Contains 224 spectral bands in the wavelength range of 0.4–2.5 μm.
o	Includes 16 ground-truth land-cover classes such as crops, grass, woods, and built-up structures.
o	Commonly used for benchmarking due to its mix of agricultural fields and man-made surfaces.
•	Salinas Scene Dataset
o	Also collected by the AVIRIS sensor, covering part of the Salinas Valley, California.
o	Features 224 spectral bands at a high spatial resolution (3.7 m per pixel).
o	Ground truth is available for 16 classes, including different crop types, soil, and vegetation.
o	Its large size and detailed annotation make it particularly suitable for testing deep learning models.
•	AVIRIS Urban Scenes
o	Provide hyperspectral cubes over urban environments.
o	Capture man-made materials (asphalt, concrete, rooftops) alongside natural features.
o	Used to evaluate how well models generalize to heterogeneous urban areas.
Together, these datasets ensure that the methodology is tested on a diverse set of conditions—from agricultural fields to dense urban zones. Their availability with labeled ground truth makes them ideal for both training supervised models and benchmarking classification performance.
🔹 Images to Add:
1.	Indian Pines RGB composite (pseudo-color image showing land-cover structure).
File: images/indian_pines_rgb.png
2.	Salinas dataset RGB view highlighting agricultural patterns.
File: images/salinas_rgb.png
3.	Sample ground-truth maps (side-by-side with data cubes).
File: images/dataset_groundtruth.png
## Methodology

Our methodology follows a structured pipeline, starting from raw hyperspectral data acquisition to generating accurate land-cover classification maps. The process involves **preprocessing, feature extraction, classification, and post-processing with evaluation metrics**.

---

### **b. Preprocessing**

The raw hyperspectral cubes often contain noise, distortions, and redundancies. To ensure data quality and reduce computational complexity, several steps are applied:

1. **Radiometric and Atmospheric Correction**  
   Raw pixel values are converted into surface reflectance by calibrating against dark and white reference values:  

   $$
   R(\lambda) = \frac{I_{raw}(\lambda) - I_{dark}(\lambda)}{I_{white}(\lambda) - I_{dark}(\lambda)}
   $$

   This eliminates sensor-induced and atmospheric variations.

2. **Geometric Alignment**  
   Bands across hundreds of wavelengths must be spatially aligned to avoid pixel misregistration. Standard geometric correction methods are applied to ensure spectral-spatial consistency.

3. **Dimensionality Reduction (PCA)**  
   Hyperspectral data contains high redundancy. Principal Component Analysis (PCA) projects the data onto fewer orthogonal components:  

   $$
   Z = X W
   $$

   where \(X\) is the data matrix, \(W\) the eigenvector matrix of the covariance matrix of \(X\), and \(Z\) the transformed representation.  
   PCA ensures that only the most informative features are preserved while reducing noise.

📌 **Images to Add:**  
- Diagram of preprocessing pipeline (correction → alignment → PCA).  
- PCA variance explained plot.  

---

### **c. Feature Extraction**

Hyperspectral cubes capture both spectral and spatial information. To utilize both:

1. **Convolutional Neural Networks (CNNs)**  
   Feature extraction begins with convolutional layers:  

   $$
   y_{i,j}^k = f\left( \sum_m \sum_{u,v} x_{i+u, j+v}^m \cdot w_{u,v}^{m,k} + b^k \right)
   $$

   where \(x\) is the input, \(w\) are convolutional filters, \(b\) is bias, and \(f\) is a non-linear activation function (e.g., ReLU).  

   - **2D CNNs** process spatial patches across spectral bands.  
   - **3D CNNs** capture spectral-spatial correlations simultaneously.  

2. **Attention Mechanism (if used)**  
   Spectral attention modules weigh the importance of different wavelength bands, enhancing relevant features while suppressing noise.  

📌 **Images to Add:**  
- CNN feature extraction block diagram.  
- Visualization of spectral vs. spatial feature extraction.  

---

### **d. Classification**

Once features are extracted, each pixel is assigned a class label.

1. **Softmax Layer for Pixel Classification**  

   $$
   P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
   $$

   where \(z_k\) are the logits for class \(k\). This provides probabilities for all possible land-cover classes.  

2. **Loss Function (Cross-Entropy)**  

   $$
   L = - \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log \hat{y}_{i,k}
   $$

   where \(y_{i,k}\) is the true label and \(\hat{y}_{i,k}\) is the predicted probability. This ensures the network is optimized to minimize classification error.  

📌 **Images to Add:**  
- Flowchart showing feature vectors entering softmax classifier.  
- Example classification map output.  

---

### **e. Post-Processing & Evaluation**

After classification, refinement and evaluation steps are applied:

1. **Post-Processing**  
   - Morphological filters are applied to smooth classification maps.  
   - Small isolated misclassified regions are removed, ensuring spatial coherence.  

2. **Evaluation Metrics**  
   The performance is quantified using standard remote sensing metrics:  

   - **Overall Accuracy (OA):**  
     $$
     OA = \frac{\text{Number of correctly classified pixels}}{\text{Total number of pixels}}
     $$  

   - **Average Accuracy (AA):**  
     Mean of per-class accuracies, useful when class imbalance exists.  

   - **Kappa Coefficient (κ):**  
     $$
     \kappa = \frac{p_o - p_e}{1 - p_e}
     $$  
     where \(p_o\) is the observed agreement and \(p_e\) is the expected agreement by chance.  

📌 **Images to Add:**  
- Confusion matrix heatmap.  
- Graph of OA, AA, and κ comparisons across datasets.  
- Side-by-side classification maps (ground truth vs. predicted).  

---


📂 Datasets Used
We used well-known benchmark hyperspectral datasets to evaluate the proposed system:
1.	Indian Pines
o	Captured by AVIRIS sensor over agricultural/urban areas in northwest Indiana.
o	Contains 224 spectral bands (0.4–2.5 μm).
o	Includes 16 land-cover classes with ground-truth labels.
2.	Salinas Scene
o	Acquired by AVIRIS sensor over Salinas Valley, California.
o	Contains 224 spectral bands and higher spatial resolution.
o	Ground-truth available for classes such as vegetation, soil, and crops.
3.	AVIRIS Urban Datasets
o	General AVIRIS hyperspectral cubes of urban environments.
o	Provide both hyperspectral data and ground-truth annotations for classification tasks.
Ground-Truth Availability
•	All datasets come with ground-truth maps, which were used for training, validation, and accuracy assessment of the models.
•	Ground-truth labels include categories such as vegetation, water, soil, crops, and man-made structures.

Project Structure
Folder & file layout explanation.
Installation & Setup
Prerequisites (Python version, libraries)
Steps for installation
Usage
How to run training
How to run classification/inference
Example commands
## Results & Evaluation

The performance of our proposed hyperspectral image classification pipeline was validated on benchmark datasets such as **Indian Pines** and **Salinas**. The evaluation focuses on multiple standard metrics to ensure both pixel-level accuracy and class-wise reliability.

---

### **1. Accuracy Metrics**

- **Overall Accuracy (OA):**  
  Proportion of correctly classified pixels across the entire dataset.  
  $$
  OA = \frac{\sum_{i=1}^{N} \mathbf{1}(y_i = \hat{y}_i)}{N}
  $$

- **Average Accuracy (AA):**  
  Mean of per-class accuracies, reducing bias due to class imbalance.  
  $$
  AA = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}
  $$

- **Kappa Coefficient (κ):**  
  Measures agreement between predicted and ground truth while correcting for chance.  
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e}
  $$

📌 **Images to Add:**  
- Bar chart comparing OA, AA, and κ across datasets.  
- Example classification maps (ground-truth vs predicted).  

---

### **2. Precision, Recall & F1-Score**

- **Precision (per class):**  
  Fraction of predicted samples that are correct.  
  $$
  Precision = \frac{TP}{TP + FP}
  $$

- **Recall (per class):**  
  Fraction of ground-truth samples correctly identified.  
  $$
  Recall = \frac{TP}{TP + FN}
  $$

- **F1-Score:**  
  Harmonic mean of Precision and Recall.  
  $$
  F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
  $$

📌 **Images to Add:**  
- Precision-Recall table or grouped bar chart.  
- Confusion matrix heatmap for class-level analysis.  

---

### **3. Sensitivity Analysis**

To test the robustness of the model, sensitivity analysis was conducted by varying:

- **Number of PCA components**  
  (e.g., 10, 20, 30) → evaluating trade-off between accuracy and computation.  

- **CNN depth and filter sizes**  
  Assessing impact on spectral-spatial feature extraction.  

- **Training sample size**  
  Evaluating classification accuracy under limited labeled data scenarios.  

This analysis highlights that the proposed method remains **stable and high-performing** even under reduced dimensionality or smaller training sets.  

📌 **Images to Add:**  
- Line plots showing OA vs. PCA components.  
- Accuracy trends for different CNN depths.  

---

### **4. Visual Results**

Sample classification outputs demonstrate the ability of the system to generate **smooth, spatially coherent land-cover maps** with minimal noise.  

📌 **Images to Add:**  
- Side-by-side visualization: (Ground Truth vs. Predicted Map).  
- Overlay of predicted classification on original hyperspectral image.  

---Poster & Presentation
Link/embed final project poster & PPT.
Future Work
Possible improvements and extensions.
Contributing
How others can contribute.
Authors / Team Members
Names, student IDs, contact info.
Acknowledgements
Mentors, resources.
License
MIT/GPL/etc.
References
Key papers cited in the project.

