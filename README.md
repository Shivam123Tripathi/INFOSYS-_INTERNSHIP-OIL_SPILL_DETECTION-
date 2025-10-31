🌊 Oil Spill Detection using Satellite Imagery  
---
## 🚀 Project Overview  

Oil spills cause catastrophic damage to marine ecosystems, impacting biodiversity, fisheries, and coastal economies.  
This project aims to *detect and segment oil spills automatically from satellite imagery* using *Deep Learning* — specifically a *U-Net based Convolutional Neural Network (CNN)* architecture.  

By leveraging AI and satellite data, this project demonstrates how modern technology can assist in *environmental surveillance and rapid disaster response*.  


## 🎯 Objectives  

- Develop an automated system for oil spill detection from satellite images.  
- Use deep learning-based *semantic segmentation* (U-Net) to highlight spill regions.  
- Build an interactive *Streamlit web app* for real-time oil spill detection.  
- Visualize and analyze detection results with adjustable sensitivity.  

---

## 🧩 Project Workflow  

📦 Data Collection → 🧹 Preprocessing → 🧠 Model Development → 🏋 Training & Evaluation → 💻 Deployment (Streamlit)

yaml
Copy code

Each phase of the workflow is organized into modules and milestones:  

| Milestone | Module | Description |
|------------|---------|-------------|
| *1* | Data Collection | Collected and structured satellite images (Kaggle Oil Spill Dataset, Sentinel-1 SAR, etc.) |
| *2* | Data Preprocessing | Resized, normalized, and augmented images; applied speckle noise reduction filters |
| *3* | Model Development | Implemented *U-Net* for segmentation and CNNs for classification |
| *4* | Training & Evaluation | Trained with Dice Loss, Binary Cross-Entropy; evaluated via Accuracy, IoU, Dice Coefficient |
| *5* | Visualization | Generated mask overlays and visual result summaries |
| *6* | Deployment | Built a Streamlit-based web app for real-time inference and visualization |

---

## 🧠 Model Details  

- *Architecture*: U-Net (Encoder–Decoder CNN for segmentation)  
- *Input Size*: 256×256 (RGB/HSI enhanced)  
- *Framework*: TensorFlow / Keras  
- *Loss Function*: Dice Loss + Binary Cross Entropy  
- *Metrics*: IoU, Dice Coefficient, Precision, Recall  
- *Optimizer*: Adam  

---

## 💻 Streamlit Application  

An intuitive *web-based interface* was built using Streamlit to make the system user-friendly and interactive.  

*Features:*  
- Upload satellite images (.jpg, .jpeg, .png)  
- Real-time spill detection and segmentation  
- Adjustable threshold (detection sensitivity) slider  
- Side-by-side visualization: Original Image | Predicted Mask | Overlay  

*Sample Workflow:*  
1. Upload satellite image  
2. Click “🔍 Detect Oil Spill”  
3. View prediction stats, binary mask, and overlay visualization  

---

## 📊 Sample Results  

| Original Image | Predicted Mask | Overlay |
|----------------|----------------|----------|
| 🛰 ![Input](https://via.placeholder.com/200x200?text=Satellite+Image) | 🩸 ![Mask](https://via.placeholder.com/200x200?text=Predicted+Mask) | 🧩 ![Overlay](https://via.placeholder.com/200x200?text=Overlay+Result) |

> The model highlights oil spill regions in green overlay for easy visualization.

---

## 🌍 Real-World Applications  

- Maritime and environmental monitoring agencies  
- Real-time oil spill surveillance and alert systems  
- Coastal ecosystem management  
- Integration with drones or live satellite feeds  
- Extension to detect *plastic waste* or *algal blooms* in oceans  

---

## 🔮 Future Scope  

- Integration with *hyperspectral* and *multispectral* satellite data for improved accuracy  
- Real-time cloud deployment (AWS / GCP / Azure)  
- Automated alert generation for detected spills  
- Expansion to *multi-class environmental hazard detection*  

---

## 🛠 Tech Stack  

| Category | Tools / Frameworks |
|-----------|--------------------|
| *Language* | Python 3.10+ |
| *Libraries* | TensorFlow, Keras, NumPy, OpenCV, Streamlit, Matplotlib |
| *Model* | U-Net (Deep Learning) |
| *Frontend* | Streamlit Web App |
| *Data Source* | Kaggle Oil Spill Detection Dataset, Sentinel-1 SAR |

---

## 🧩 Folder Structure  

📁 INFOSYS-_INTERNSHIP-OIL_SPILL_DETECTION-
│
├── 📄 app.py # Streamlit web app
├── 🧠 oil_spill_unet_best.h5 # Trained UNet model
├── 📁 dataset/ # Training and testing datasets
├── 📁 results/ # Predicted masks and overlays
├── 📁 notebooks/ # Jupyter notebooks for EDA & model training
├── 📄 requirements.txt # Dependencies
└── 📄 README.md # Project documentation

yaml
Copy code

---

## 👩‍💻 Author  

*Shivam Tripathi*  
Deep Learning & Computer Vision Enthusiast  
📧 s.shivamtripathi13@gmail.com  


---

> 🏁 "Protecting our oceans through the power of AI." 🌊

---
