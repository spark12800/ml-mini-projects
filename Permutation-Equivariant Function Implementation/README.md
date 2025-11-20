# Permutation-Equivariant Function Implementation 

This project implements a **permutation-equivariant function** using a combination of **Conv1D layers** and **Global Average Pooling**, reproducing the two-term formulation:

```math
\sum_{u=1}^{d} a_{u,v} x_{j,u} + c_v
\quad+\quad
\sum_{u=1}^{d} b_{u,v} \left( \frac{1}{m} \sum_{l=1}^{m} x_{l,u} \right)
```

The goal is to study how architectural depth and kernel width influence equivariance, optimisation behaviour, and generalisation.

---

## Method

### **1. Row-wise Equivariant Term**
```math
\sum_{u=1}^{d} a_{u,v} x_{j,u} + c_v
```
- Implemented using **Conv1D(kernel_size=1, bias=True)**  
- Each row is processed independently → preserves permutation equivariance  
- Conv1D naturally computes a learnable weighted sum over feature dimension \( d \)  
- Convolution’s locality ensures that **reordering rows does not mix their information**

### **2. Permutation-Invariant Global Term**
```math
\sum_{u=1}^{d} b_{u,v} \left( \frac{1}{m} \sum_{l=1}^{m} x_{l,u} \right)
```
- Implemented via **GlobalAveragePooling1D**  
- The mean across rows is invariant to row order  
- Followed by **Conv1D(kernel_size=1, bias=False)**  
- Bias removed because global features are already invariant  

### **Why Conv1D Instead of Dense?**

A fully-connected layer mixes across rows and **breaks equivariance** unless specially constrained.  
Conv1D(kernel=1) applies the same transformation to each row independently → preserves structure.

---

## Experiments: Depth × Width Ablation

Architectures are denoted as **L:x (layers) / w:y (kernels)**.

### Key empirical findings:

- **Deeper and wider models reduce loss more effectively.**  
  L:3 w:200 consistently gives the strongest optimisation.

- **L:5 w:2 underperforms**, showing that depth alone cannot compensate for insufficient kernel capacity.

- **Two-layer models (L:2 w:10 → 100 → 200)** improve gradually with width,  
  but plateau earlier than deeper models.

- **L:3 w:5 performs surprisingly well**, indicating diminishing returns for very wide expansions once depth is adequate.

- **Over-parameterised networks** (e.g., L:3 w:200) generalise better but require significantly more compute.

### Interpretation

- There is a clear trade-off between **network expressiveness** and **computational efficiency**.  
- Beyond a depth threshold (≈ L:3), width becomes less critical.  
- Sparse, deeper architectures (e.g., L:3 w:5) can achieve strong performance with far fewer parameters.

---
## Tools

- TensorFlow / Keras  
- Conv1D, GlobalAveragePooling1D  
- Python (NumPy, Matplotlib)
