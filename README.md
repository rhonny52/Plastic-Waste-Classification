##  Waste Management And Classification




## Overview  
This project focuses on building a Convolutional Neural Network (CNN) model to classify images of plastic waste into various categories. The primary goal is to enhance waste management systems by improving the segregation and recycling process using deep learning technologies.  

---

## Table of Contents  
- [Project Description](#project-description)  
- [Dataset](#dataset)  
- [Training](#training)  
- [Weekly Progress](#weekly-progress)  
- [How to Run](#how-to-run)  
- [Technologies Used](#technologies-used)  
- [Future Scope](#future-scope)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Description  
Plastic pollution is a growing concern globally, and effective waste segregation is critical to tackling this issue. This project employs a CNN model to classify plastic waste into distinct categories, facilitating automated waste management.  

## Dataset  
The dataset used for this project is the **Waste Classification Data** by Sashaank Sekar. It contains a total of 25,077 labeled images, divided into two categories: **Organic** and **Recyclable**. This dataset is designed to facilitate waste classification tasks using machine learning techniques.  


### Key Details:
- **Total Images**: 25,077  
  - **Training Data**: 22,564 images (85%)  
  - **Test Data**: 2,513 images (15%)  
- **Classes**: Organic and Recyclable  
- **Purpose**: To aid in automating waste management and reducing the environmental impact of improper waste disposal.
  
### Approach:  
- Studied waste management strategies and white papers.  
- Analyzed the composition of household waste.  
- Segregated waste into two categories (Organic and Recyclable).  
- Leveraged IoT and machine learning to automate waste classification.  

### Dataset Link:  
You can access the dataset here: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data).  

*Note: Ensure appropriate dataset licensing and usage guidelines are followed.*  



## Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** Configurable (default: 25)  
- **Batch Size:** Configurable (default: 32)  

Data augmentation techniques were utilized to enhance model performance and generalizability.  

## Weekly Progress  
This section will be updated weekly with progress details and corresponding Jupyter Notebooks.  

### **Week 1: Libraries, Data Import, and Setup**  
- **Date:** 21st January 2025 - 24th January 2025  
- **Activities:**  
  - Imported the required libraries and frameworks.  
  - Set up the project environment.  
  - Explored the dataset structure.  
  - Note: If the file takes too long to load, you can view the Kaggle notebook directly [Kaggle Notebook](https://www.kaggle.com/code/hardikksankhla/cnn-plastic-waste-classification).  

- **Notebooks:**  
  - [Week1-Libraries-Importing-Data-Setup.ipynb](https://github.com/rhonny52/Plastic-Waste-Classification/blob/main/week1-datasetup-and-visualization.ipynb)  
  - [Kaggle Notebook](https://www.kaggle.com/code/rajsaraf/week1-datasetup-and-visualization)

### **Week 2: Model Training, Evaluation, and Predictions**  
- **Date:** 28th January 2025 - 31st January 2025  
- **Activities:**  
  - Trained the CNN model on the dataset.  
  - Optimized hyperparameters to improve accuracy.  
  - Evaluated model performance using accuracy and loss metrics.  
  - Performed predictions on test images.  
  - Visualized classification results with a confusion matrix.  

- **Notebooks:**  
  - [Week2-Model-Training-Evaluation-Predictions.ipynb](https://github.com/rhonny52/Plastic-Waste-Classification/blob/main/wasteclassification.ipynb)
  -VS CODE 


## Week 3: Advanced Model Optimization
**Date:** 3rd February 2025 - 7th February 2025

### Activities:
- Fine-tuned the CNN model using transfer learning.
- Applied data augmentation techniques.
- Compared performance with different architectures (e.g., ResNet, VGG16).
- Deployed the model for real-world testing.

### Notebooks:
- [Week3-Advanced-Optimization.ipynb](https://github.com/rhonny52/Plastic-Waste-Classification/blob/main/Waste_Management_And_Classification.ipynb)
- [Google Colab](https://colab.research.google.com/drive/1S3Z0ZfkLIBeDiFQZ_ExA2IU4U1yDXvP0#scrollTo=pF7hZzu5JSJT)

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Hardik-Sankhla/CNN-Plastic-Waste-Classification  
   cd CNN-Plastic-Waste-Classification
   ```  
2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the training script:  *Details to be added after completion.*  
   ```bash  
   python train.py  
   ```  
4. For inference, use the following command:  *Details to be added after completion.*  
   ```bash  
   python predict.py --image_path /path/to/image.jpg  
   ```  

## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  

## Future Scope  
- Expanding the dataset to include more plastic waste categories.  
- Deploying the model as a web or mobile application for real-time use.  
- Integration with IoT-enabled waste management systems.  

## Contributing  
Contributions are welcome! If you would like to contribute, please open an issue or submit a pull request.  

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
