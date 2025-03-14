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

## 📊 Evaluation Criteria
The project will be graded based on:
- Suitability of the model for the task.
- Model correctness and complexity.
- Iterative improvements and adjustments.

## 🎖️ Bonus (Up to +1.5 Points)
Extra points can be earned by implementing:
- **Self-supervised learning**
- **Uncertainty quantification**
- **Federated learning**

## 📌 Submission Details
**Deadline**: March 30th, 2025 (23:59:59) on Moodle.

### Required Deliverables:
- **Code** (Notebook or script files) allowing full model training and result reproduction.
- **Report** including:
  - Overview of the deep learning approaches used.
  - Step-by-step improvements and justifications.
  - Classification and generation results with training details.
  - Discussion and conclusions.
- **Auto-evaluation file** detailing team contributions.

## 🎤 Presentation
- **Dates**: April 4th and 11th, 2025.
- **All team members must be present** and explain their contributions.
- Lack of preparation may result in penalization.

## ⚠️ Important Notes
- The dataset files **must not** be included in the submission.
- Groups must have **3 members** (exceptions need approval).
- Late submissions will be penalized.

---

Made with ❤️ by [Your Team Name]
