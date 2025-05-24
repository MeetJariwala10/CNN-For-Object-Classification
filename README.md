# CNN-For-Object-Classification

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for object classification on the CIFAR-10 dataset. All code lives in the Jupyter notebook `cnn-for-object-classification.ipynb`.

---

## 📝 Notebook Overview

1. **Environment setup**  
   - Imports: `torch`, `torchvision.transforms`, `torch.nn`, `torch.optim`, `tqdm`, `numpy`, `pandas`, `pickle`  
   - Device selection: GPU if available, otherwise CPU

2. **Data loading**
   - Download the dataset from: `https://www.cs.toronto.edu/~kriz/cifar.html`
   - Raw CIFAR-10 binary files read via a custom `unpickle` function  
   - `CIFAR10Raw(Dataset)` wraps the unpickled batches  
   - `DataLoader` for training and test sets (batch size = 128)

4. **Transforms**  
   - **Training**: random horizontal flip, tensor conversion, normalization (mean = `(0.4914, 0.4822, 0.4465)`, std = `(0.2470, 0.2435, 0.2616)`)  
   - **Test**: tensor conversion, same normalization

5. **Model architecture**  
   - Defined as `SimpleCNN(nn.Module)`  
   - Two convolutional blocks (32→64 channels), each with Conv2d→ReLU→MaxPool→Dropout  
   - Classifier: flatten → 512-unit Linear → ReLU → Dropout → 10-unit Linear  

6. **Training loop**  
   - Loss: `CrossEntropyLoss`  
   - Optimizer: Adam (learning rate = 1e-3)  
   - Epochs: 20  
   - Progress bar via `tqdm`

7. **Evaluation**  
   - Switch to eval mode, disable gradients  
   - Compute overall test accuracy:  
     ```python
     test_acc = 100.0 * correct / total
     print(f"Test accuracy: {test_acc:.2f}%")
     ```
---

## 🚀 Getting Started

1. **Clone this repo**  
   ```bash
   git clone https://github.com/<your-username>/cnn-for-object-classification.git
   cd cnn-for-object-classification
   ```


2. **Install dependencies**

Import all the necessary libraries from `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

3. **Run the code cell by cell with proper understanding and yaa thats all!!** 

