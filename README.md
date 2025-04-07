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
   - Possible approaches: CNN, Vision Transformers.
   
2. **Generative Model** - A deep learning model to generate realistic fake images.
   - Approaches: DCGAN.
