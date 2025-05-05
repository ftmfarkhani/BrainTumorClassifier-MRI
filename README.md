# **Brain Tumor Classification Platform**  
A platform for **brain tumor classification** using MRI images, enhanced with image modification techniques to improve detection accuracy in **low-quality scans**.

### **Purpose**  
- Classifies brain MRI images using **convolutional neural networks (CNNs)**.  
- Optimized for handling low-quality medical imaging.

### **Technologies & Frameworks**  
- **TensorFlow/Keras** – Deep learning model development  
- **Albumentations** – Advanced image augmentation  
- **Streamlit** – Interactive web-based deployment  

### **Model Architecture**  
- Built on **InceptionV3**, a pre-trained CNN model for feature extraction.  
- Incorporates **dropout layers** to prevent overfitting.  

### **Augmentation & Regularization**  
- Implements **noise-based augmentations** to enhance model robustness.  
- Leverages **class weight adjustments** to handle data imbalance.

### **Deployment**  
- **Streamlit integration** enables real-time tumor classification in a user-friendly web app.  
