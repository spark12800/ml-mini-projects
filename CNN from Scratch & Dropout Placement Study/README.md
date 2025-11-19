# CNN from Scratch & Dropout Placement Study

**Goal:**  
Build a convolutional neural network from scratch on **CIFAR-10** and systematically evaluate:
- different optimisation algorithms  
- the effect of dropout placement and dropout rates  
- architectural variations (baseline CNN → deeper CNN → Inception-style CNN → ResNet-style CNN)

---

## 1. Baseline CNN Implementation

The first model follows a simple, classical CNN architecture:

**3×32×32 → 32C5 → P2 → 64C5 → P2 → 128C5 → P2 → 1000N → 10N**

**Hyperparameters:**
- Conv filters: **5×5**
- Stride: **1**
- Pooling: **MaxPool 2×2**
- Activation: **ReLU**

This serves as the reference model for comparing optimisers and dropout placement.

---

## 2. Optimiser Comparison

The following mini-batch optimisers were evaluated:

- SGD  
- SGD + Nesterov momentum  
- AdaGrad  
- RMSProp  
- **Adam (best)**  

To ensure fair comparison:
- **Dropout 0.3** was applied **only at the fully connected layer**.  
- `steps_per_epoch = 50` and `shuffle = True` ensured consistent epoch-level loss curves.

**Key finding:**  
**Adam achieved the most stable convergence** on this architecture, followed by RMSProp and AdaGrad.  
Despite this, optimiser performance can change under hyperparameter tuning, as shown in prior work (Choi et al., 2019).

---

## 3. Dropout Placement Study

Using Adam, dropout was applied at **different stages**:

- After convolutional layers  
- After max-pooling layers  
- After the fully connected layer  

**Rates tested:** 0.1, 0.3, 0.5, 0.7, 0.9

**Results:**
- Best overall setting: **0.1 (conv + pool) and 0.3 (FC)**
- **Pooling-layer dropout** produced the highest validation accuracy  

This suggests that **stochastic pooling** helps prevent dominance of strong local features and reduces overfitting.

---

## 4. Improved Architecture (Deep CNN)

To address the limitations of the baseline model, a VGG-inspired deeper architecture was built:

- Eight Conv layers (3×3)
- Three MaxPool layers
- Two Dense layers
- Batch Normalisation after every Conv
- ReLU activations
- Learning Rate Scheduler (factor 0.05, patience 2)
- Pooling dropout = 0.1

Using **multiple small kernels (3×3)** increased effective receptive fields while controlling parameter counts.  
This architecture showed significantly better generalisation.

---

## 5. Inception-Style Architecture

A second improved model used **Inception blocks** for multi-scale feature extraction:

- Parallel 3×3, 5×5, and 7×7 filters  
- 1×1 bottlenecks for channel reduction  
- Global Average Pooling instead of Dense layers  
- **Trainable parameters reduced from 6.4M → 3.0M**

**Optimisation:**
- **AdamW**
- LeakyReLU (α = 0.3)
- L2 regularisation (0.0001)
- Learning rate exponential decay

This architecture captured both fine- and coarse-scale patterns and reduced overfitting.

---

## 6. ResNet-Style Architecture

A third model explored a simplified **ResNet-18–like** design:

Differences from original ResNet:
1. CIFAR-10-appropriate 3×3 initial convolution (no 7×7 + MaxPool)  
2. Downsampling performed via **MaxPool between residual blocks**, not stride-2 convolutions  
3. Pooling dropout used  
4. Adam instead of SGD (0.9 momentum)  
5. Custom learning-rate schedule:  
   - Epoch 1: lr = 0.1  
   - Epoch 2–5: lr = 0.001  
   - Later epochs: manually reduced based on error plateau
     
---

## Summary of Techniques Applied

**Regularisation**
- Dropout (various placements)
- Batch Normalisation
- Layer Normalisation
- L2 Regularisation
- Early Stopping

**Optimisation**
- Adam / AdamW
- Learning-rate warmup (linear)
- LR scheduling (ReduceLROnPlateau or manual scheduling)
- He Normal initialisation

**Architectural Variants**
- Basic CNN  
- Deep VGG-style CNN  
- Inception-style multi-branch CNN  
- ResNet-style CNN with modified downsampling  

---

**Framework:** TensorFlow, Keras
**File:** `CNNs_From_Scratch.ipynb`  
