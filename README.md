# Deep Learning Project - Deep Fake Detection and Generation

## 📌 Objective
The goal of this project is to develop both discriminative and generative deep learning models in the context of deep fakes:
- A **discriminative model** to classify images as "real" or "fake".
- A **generative model** to create new fake images.

## 📂 Dataset
The dataset used is **DeepFakeFace (DFF)**, accessible via Hugging Face:
[DeepFakeFace Dataset](https://huggingface.co/datasets/OpenRL/DeepFakeFace)

It consists of:
- **30,000 real images** of celebrities from the IMDB-WIKI dataset.
- **90,000 fake images** generated using:
  - Stable Diffusion v1.5
  - Stable Diffusion Inpainting
  - InsightFace

For further details, see: [Dataset Paper](https://arxiv.org/pdf/2309.02218)

## 🚀 Implementation
This project consists of implementing two models:

1. **Classifier** - A deep learning model to distinguish between real and fake images.
   - Possible approaches: MLP, CNN, Vision Transformers, etc.
   
2. **Generative Model** - A deep learning model to generate realistic fake images.
   - Possible approaches: GANs, VAEs, Diffusion Models, etc.

### ✅ Training Strategy & Evaluation
For both models, we will:
- Define appropriate **training strategies**.
- Choose suitable **performance metrics**.
- Continuously improve models based on intermediate results.

## 📌 Repository Structure
```
📂 project-root
│-- 📂 data/                # Dataset (not included in repo, use dataset link)
│-- 📂 models/              # Model implementations
│-- 📂 notebooks/           # Jupyter notebooks for experiments
│-- 📂 src/                 # Source code for training and evaluation
│-- 📂 results/             # Output results, logs, and model checkpoints
│-- README.md               # Project overview
│-- requirements.txt        # Dependencies
│-- train.py                # Training script
│-- generate.py             # Image generation script
```

## 📦 Installation & Setup
Clone this repository:
```bash
git clone https://github.com/your-repo/deepfake-detection.git
cd deepfake-detection
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## 🏃 Running the Project
### Train the Classifier:
```bash
python train.py --model classifier --epochs 50
```

### Train the Generative Model:
```bash
python train.py --model generator --epochs 50
```

### Generate Fake Images:
```bash
python generate.py --num_images 10
```
