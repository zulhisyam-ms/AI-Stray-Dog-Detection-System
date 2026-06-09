# Real-Time Stray Dog Detection & Control System using Deep Learning

## 📌 Project Overview
An automated urban animal monitoring solution developed to tackle public health and welfare challenges via edge computing. This production pipeline utilizes a custom fine-tuned **YOLOv8n Convolutional Neural Network (CNN)** to ingest raw surveillance streams, classify targets, log timestamped telemetry into relational formats, and execute immediate cloud-based push notifications.

---

##  Technical Architecture & Workflow
1. **Data Ingestion & Curation:** Collected a curated target sample array from Roboflow Universe, scaling the final asset space to **1,831 expanded images** via brightness, flipping, and blurring augmentations to ensure real-world durability.
2. **Deep Learning Core:** Implemented an anchor-free **YOLOv8n (Nano)** framework inside a GPU-accelerated persekitaran (Google Colab) to map out rapid bounding box feature fusion via C2f modules and GSConv computing blocks.
3. **Telemetry & API Communications:** Engineered an OpenCV pipeline evaluating frame arrays sequentially. Detections maintaining a confidence vector of **$\ge 75\%$** append spatial meta logs directly to localized databases and trigger instant push notifications through active webhooks.
4. **Cloud Analytics Dashboard:** Deployed a highly visual responsive analytics engine featuring explicit event timelines and geographic incident ratios.

---

##  Deep Learning Training Performance & Results

### Model Metric Parameters
- **Epoch Training Scope:** 90 Cycles 
- **Batch Processing Volume:** 32 Instances 
- **Input Dimension Map:** $640 \times 640$ Pixels 
- **Optimization Strategy:** AdamW 

### Evaluation Metrics (Test Dataset)
- **Model Precision ($B$):** 89.9% 
- **Model Recall ($B$):** 93.0% 
- **mAP @ 0.50 conventional threshold:** 92.26% 
- **System Classification Accuracy:** 85.0% *(Derived from 517 True Positives vs 42 False Negatives)* 

---

##  Dashboard & Performance Metrics Visualizations

### Advanced CNN Training Trajectory Loss Curves
![Model Analysis](images/model_analysis.png)
*Description: Explicit metrics map proving progressive tracking of standard bounding box loss (box_loss) and localized classification loss metrics decreasing across 90 training iterations without evidence of model overfitting.*

### Production Dashboard Visual Analytics
![Dashboard Preview](images/dashboard_preview.png)
*Description: Operational views of the active dashboard illustrating unified video streams alongside synchronized geographic donut configurations and target location telemetry.*
