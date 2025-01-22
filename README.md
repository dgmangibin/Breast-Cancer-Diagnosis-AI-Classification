# Breast-Cancer-Diagnosis-AI-Classification

Explanation: 
This AI Project implements a Feedforward Neural Networks to predict the diagnosis of breast cancer 
My approach starts by importing Python's sklearn.model_selection module 
so that we can split our dataset into training and testing sets into 80:20% 
and we can later cross-validate our model for accuracy. 
I also used the open-source tf from TensorFlow’s API which is good for numerical computations and ML

Data: 
The dataset contains features derived from digitized images of fine needle aspirates (FNA) of breast masses, focusing on the characteristics of cell nuclei. It includes 32 attributes, starting with an ID number, a diagnosis label (M = malignant, B = benign), and 30 computed features based on the cell nucleus properties.

The dataset contains no missing values, and all numerical values are recorded with four significant digits. The class distribution includes 357 benign and 212 malignant cases. It is available via the UCI Machine Learning Repository and the University of Wisconsin's FTP server.

The features are calculated as the mean, standard error, and worst values (largest mean of three values) for the following ten properties:

Radius: Mean distance from the center to perimeter points.
Texture: Standard deviation of grayscale values.
Perimeter and Area.
Smoothness: Local variation in radius lengths.
Compactness: (Perimeter² / Area) - 1.
Concavity and Concave Points: Severity and count of concave contour portions.
Symmetry.
Fractal Dimension: A measure of the "coastline approximation."
