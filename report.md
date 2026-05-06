# 100-Species Animal Classifier: Fine-Tuned CNN for Indian Wildlife & Beyond

**Course:** Statistical Methods in AI (SMAI) — Assignment 3, Tier 2
**Theme:** T7.7 — 100-Species Fine-Tuned Identifier
**Dataset:** [ViratGarg/animal_species_SMAI](https://huggingface.co/datasets/ViratGarg/animal_species_SMAI)

---

**Team Name:** Kuch bhi daal do

| Name | Roll No. | Email |
|------|----------|-------|
| Krrish Goenka | 2023112023 | krrish.goenka@research.iiit.ac.in |
| Krish Agarwal | 2023113017 | krish.agarwal@research.iiit.ac.in |
| Pranav Shankar | 2023112011 | pranav.shankar@research.iiit.ac.in |
| Virat Garg | 2023101081 | virat.garg@students.iiit.ac.in |
| Manas Agrawal | 2023113023 | manas.agrawal@research.iiit.ac.in |

---

## Abstract

We present a multi-approach animal species classification system unifying three wildlife datasets — mammals, birds, and butterflies — into a single 100-class benchmark. Three strategies are evaluated: (1) a ResNet50 fine-tuned end-to-end on all 100 classes, (2) a two-stage hierarchical ResNet50 pipeline that routes images by animal group before species identification, and (3) a CLIP-based approach using KNN and SVM over frozen visual embeddings. All approaches exceed 95% test accuracy, with fine-tuned ResNet pipelines reaching up to **97.22%**.

---

## 1. Dataset

The dataset merges three animal image sources into a unified 100-class label space, hosted on HuggingFace:

| Subset | Species | Images |
|--------|---------|--------|
| Mammals | 45 | 13,751 |
| Birds | 25 | 8,750 |
| Butterflies | 30 | 4,158 |
| **Total** | **100** | **26,659** |

**Splits:** Train: 21,327 | Val: 2,666 | Test: 2,666

---

## 2. Methods

### 2.1 Model 1: Single ResNet50 (Direct 100-Class Fine-Tuning)

A ResNet50 pretrained on ImageNet1K is used as the backbone. The final FC layer is replaced with a sequential head consisting of a Dropout layer (p=0.4) followed by a linear layer mapping to 100 output classes. The entire network is fine-tuned end-to-end.

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Epochs | 20 |
| LR Schedule | Cosine Annealing (T_max=20) |
| Loss | Cross-Entropy (label smoothing ε=0.1) |
| Dropout | 0.4 |
| Batch Size | 32 |
| Input Size | 224×224 |

---

### 2.2 Model 2: Two-Stage Hierarchical ResNet50 Pipeline

Four ResNet50 models are used in a cascaded pipeline:

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: Group Classifier (3-way)  │
│       Bird | Butterfly | Mammal     │
└─────────────────────────────────────┘
        │           │            │
        ▼           ▼            ▼
    Birds       Butterfly    Mammals
    ResNet50    ResNet50     ResNet50
    (25-class)  (30-class)   (45-class)
```

Stage 1 classifies images into one of three animal groups; the predicted group routes each image to the corresponding Stage 2 species model. Each Stage 2 model tackles at most a 45-way classification, rather than the full 100-way problem.

**Hyperparameters (same across all 4 models):**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Stage 1 Epochs | 15 |
| Stage 2 Epochs | 15 |
| LR Schedule | Cosine Annealing |
| Loss | Cross-Entropy (label smoothing ε=0.1) |
| Dropout | 0.4 |
| Batch Size | 32 |

Checkpointing is done on best validation accuracy for each model.

---

### 2.3 Model 3: CLIP Embeddings + KNN / SVM

No fine-tuning is required. Frozen image encoders extract visual embeddings which are classified using classical ML methods.

| Encoder | Description |
|---------|-------------|
| OpenAI CLIP ViT-B/16 | General-purpose, trained on 400M image-text pairs |
| OpenAI CLIP ViT-B/32 | Lighter general-purpose variant |
| BioCLIP | Domain-specialized CLIP for biological/taxonomic data |

**Classification heads:** Zero-Shot (cosine similarity to text prompts), KNN (k=1 to 20), Linear SVM.

---

## 3. Feature Space Analysis

PCA projections of CLIP (ViT-B/16) embeddings reveal macro-level separation across the three animal groups — butterflies form a tight, well-separated cluster while mammals and birds overlap moderately at boundaries. This confirms CLIP encodes high-level biology without any task-specific training.

**Silhouette Scores:**

| Scope | Score |
|-------|-------|
| 3 broad categories | 0.1449 |
| Mammals (45 species) | 0.1165 |
| Birds (25 species) | 0.1192 |
| Butterflies (30 species) | 0.0600 |

The low butterfly silhouette score (0.06) foreshadows weaker KNN/SVM performance on that subset — butterfly species share considerable visual overlap in the embedding space.

![Broad PCA of All Categories](all_categories_pca.png)

---

## 4. Results

### 4.1 Model 1: Single ResNet50

| Epoch | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| 1 | 78.09% | 94.37% | 94.11% |
| 5 | 98.18% | 96.06% | 95.87% |
| 10 | 99.34% | 96.70% | 96.51% |
| 15 | 99.84% | 97.00% | 97.11% |
| 20 | 99.96% | 97.41% | **97.22%** |

**Final Test Accuracy: 97.22% | Macro Specificity: 99.97%**

The model converges rapidly — validation accuracy exceeds 94% by epoch 1. A small but stable generalization gap persists (~2.5%), controlled by label smoothing and cosine annealing.

---

### 4.2 Model 2: Two-Stage Hierarchical Pipeline

**Stage 1 — Group Classifier:**

| Metric | Score |
|--------|-------|
| Accuracy | 99.89% |
| Macro F1 | 99.86% |
| Macro Specificity | 99.93% |

Only 3 out of 2,666 test samples were misrouted (0.11%). Misrouted samples cannot be recovered regardless of Stage 2 performance.

**Stage 2 — Species Classifiers:**

| Group | Test Accuracy | Macro F1 |
|-------|---------------|----------|
| Birds (25 classes) | 97.14% | 97.13% |
| Butterflies (30 classes) | 97.59% | 97.54% |
| Mammals (45 classes) | 96.66% | 96.59% |

**End-to-End Pipeline:**

| Metric | Score |
|--------|-------|
| End-to-end Species Accuracy | **96.89%** |
| End-to-end Macro F1 | 96.91% |
| End-to-end Macro Specificity | 99.97% |

---

### 4.3 Model 3: CLIP Embeddings

**SVM Results:**

| Embeddings | Mammals | Birds | Butterflies | Combined |
|------------|---------|-------|-------------|----------|
| CLIP ViT-B/16 | 95.86% | 97.54% | 91.71% | 95.76% |
| BioCLIP | 94.62% | **99.03%** | **98.08%** | **96.61%** |

BioCLIP dramatically improves butterfly accuracy (91.71% → 98.08%), as expected given its biological taxonomy training data.

**Zero-Shot Results (no training data used):**

| Model | Mammals | Birds | Butterflies | Combined |
|-------|---------|-------|-------------|----------|
| CLIP ViT-B/32 | 92.26% | 79.09% | 37.86% | 78.72% |
| CLIP ViT-B/16 | 93.82% | 83.49% | 39.42% | 81.14% |
| BioCLIP | 81.46% | 89.71% | 55.17% | 80.01% |

Zero-shot butterfly accuracy is low for standard CLIP models, confirming that the bottleneck is text-prompt matching rather than the image encoder quality.

---

### 4.4 Summary Comparison

| Approach | Test Accuracy |
|----------|---------------|
| Model 1: Single ResNet50 | **97.22%** |
| Model 2: Hierarchical ResNet50 | 96.89% |
| Model 3: BioCLIP + SVM | 96.61% |
| Model 3: CLIP ViT-B/16 + SVM | 95.76% |
| Model 3: CLIP + KNN (best k) | ~94–95% |
| Zero-Shot CLIP ViT-B/16 | 81.14% |
| Zero-Shot BioCLIP | 80.01% |

---

## 5. Demo Application

We built a Streamlit web application that allows users to upload an image and get species predictions from all three models. The app returns the top-3 predicted species along with confidence scores, and displays the model's routing decision in the hierarchical pipeline.

> **Screenshot placeholder — insert app screenshot here:**

![Streamlit App Screenshot](streamlit_app_screenshot.png)

*The app takes an image as input, runs inference through the selected model, and returns the top predicted species with confidence scores.*

---

## 6. Discussion

**Why are accuracies so high?** ResNet50 pretrained on ImageNet1K and CLIP trained on 400M image-text pairs are both significantly overqualified for 100-class animal classification. These models already encode deep features for fur texture, wing patterns, and body shape — fine-tuning only needs to align the final classification head to the new label space. Additionally, most species in our dataset are visually distinctive — a giraffe looks nothing like a warthog — unlike fine-grained benchmarks such as CUB-200 where inter-class differences can be subtle.

**Why does CLIP underperform fine-tuned ResNets?** KNN and SVM find proximity/linear boundaries in a 512-dimensional space not explicitly trained for species-level separability. Fine-tuned ResNets adapt the entire representation to maximize species-level discrimination. That said, the use of KNN and SVM here was a deliberate pedagogical choice — integrating classical ML algorithms from the course curriculum with modern CLIP embeddings demonstrates that strong representations can unlock solid performance even from simple classifiers.

**Butterflies are the hardest class.** The lowest silhouette score (0.06), lowest KNN accuracy (~88%), and worst zero-shot scores all point to butterflies as the most challenging group. BioCLIP's dramatic improvement on butterflies confirms this is a representation issue rather than an inherent difficulty — the right pretraining data matters.

---

## 7. Limitations

- **Butterfly discrimination** remains the weakest link across all methods, particularly for zero-shot CLIP.
- **Seal and blue whale** show the lowest per-class F1 scores (seal: 0.828, blue whale: 0.840) among mammals, likely due to visual similarity with sea lions and lack of close-up distinguishing features respectively.
- **Small test set per class** (~26 samples average) means per-class metrics have high variance.

---

## 8. Conclusion

Three approaches to 100-class animal species classification all achieve >95% test accuracy on a combined mammals/birds/butterflies benchmark. The single ResNet50 direct fine-tune achieves the best overall accuracy (97.22%), the hierarchical pipeline (96.89%) offers better modularity and interpretability, and the CLIP+SVM approach (up to 96.61% with BioCLIP) requires no gradient-based training on the target dataset. High accuracies across the board reflect the strength of modern pretrained representations applied to a visually distinctive, well-curated dataset.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
2. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*.
3. Stevens, S., et al. (2024). BioCLIP: A Vision Foundation Model for the Tree of Life. *CVPR*.
4. Cover, T. M., & Hart, P. E. (1967). Nearest Neighbor Pattern Classification. *IEEE Transactions on Information Theory*.
5. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*.
6. Dataset: [ViratGarg/animal_species_SMAI](https://huggingface.co/datasets/ViratGarg/animal_species_SMAI) on HuggingFace.

---

## Acknowledgements

LLM tools were used for code scaffolding assistance. All model training, evaluation, analysis, and report writing were carried out by the project team.

---

*SMAI Assignment 3 — Tier 2 | T7.7: 100-Species Fine-Tuned Identifier*