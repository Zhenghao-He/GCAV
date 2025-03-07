# GCAV: A Global Concept Activation Vector Framework for Cross-Layer Consistency in Interpretability

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)

## 🧩 Overview

**Global Concept Activation Vector (GCAV)** is a novel framework designed to improve the interpretability of deep neural networks by unifying Concept Activation Vectors (CAVs) into a globally consistent representation. Unlike traditional **TCAV**, which independently computes CAVs at each layer, GCAV **aligns and integrates CAVs** across layers using:
- **Contrastive learning** to ensure concept representations are semantically aligned.
- **Attention-based fusion** to construct a unified global concept representation.
- **Decoder-based projection** to reconstruct per-layer concept vectors, allowing TCAV to be applied to GCAV representations (**TGCAV**).

## 🚀 Features
- **Cross-Layer Concept Consistency**: Mitigates inconsistencies in TCAV scores across layers.
- **Improved Concept Localization**: Enhances interpretability by refining concept attributions.
- **Adversarial Robustness**: Provides more stable explanations under adversarial perturbations.
- **Flexible & Scalable**: Works with multiple deep architectures (e.g., ResNet, GoogleNet, MobileNet).

## 🛠 Installation

### **1. Clone the Repository**
```bash
