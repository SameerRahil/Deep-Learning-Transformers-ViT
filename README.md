# Transformers and Vision Transformers from Scratch

Project 4 of CS 437 Deep Learning, LUMS, Spring 2026. The first notebook builds a Transformer language model and a Vision Transformer from their parts; the second changes how attention works and measures what it does to a ViT.

## Parts

| Part | Notebook | What it covers |
| --- | --- | --- |
| 1. Transformer LM and ViT | [`1_transformer_lm_and_vit.ipynb`](notebooks/1_transformer_lm_and_vit.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Transformers-ViT/blob/main/notebooks/1_transformer_lm_and_vit.ipynb), [Colab](https://colab.research.google.com/github/SameerRahil/Deep-Learning-Transformers-ViT/blob/main/notebooks/1_transformer_lm_and_vit.ipynb)) | A tokenized movie-script dataset, multi-head attention, feedforward and encoder blocks, a trainer and text generation; then a Vision Transformer (patchifier, positional embeddings, attention, encoder) trained on MNIST with attention visualizations. |
| 2. Attention variants and a hybrid ViT | [`2_attention_variants_and_hybrid_vit.ipynb`](notebooks/2_attention_variants_and_hybrid_vit.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Transformers-ViT/blob/main/notebooks/2_attention_variants_and_hybrid_vit.ipynb), [Colab](https://colab.research.google.com/github/SameerRahil/Deep-Learning-Transformers-ViT/blob/main/notebooks/2_attention_variants_and_hybrid_vit.ipynb)) | Top-k sparse attention, temperature-scaled attention (fixed and learned), CLS-token attention maps and entropy, a vanilla ViT against a hybrid with a CNN patch-embedding stem, and a patch-size sweep, all on CIFAR-10. |

GitHub's notebook preview can be slow or fail to load. If it does, the nbviewer link next to each notebook shows the same file, and the Colab link opens it ready to run.

## Results

- **ViT on MNIST:** 95.78% test accuracy after 5 epochs.
- **Sparse attention (CIFAR-10, 20k-image subset, 15 epochs):** keeping only the top 32 keys matched full attention (51.6% vs 51.5% validation accuracy) with lower attention entropy (3.43 vs 4.08 nats).
- **Temperature:** a learned temperature reached 51.6%; a fixed temperature of 4 reached 47.9%.

| Model (CIFAR-10) | Parameters | Train accuracy | Validation accuracy |
| --- | --- | --- | --- |
| Vanilla ViT | 544,138 | 76.60% | 74.84% |
| Hybrid CNN + ViT | 670,858 | 78.58% | 76.27% |

| Patch size | Tokens | Validation accuracy | Seconds per epoch |
| --- | --- | --- | --- |
| 2 | 256 | 70.44% | 19.10 |
| 4 | 64 | 70.10% | 19.09 |
| 8 | 16 | 66.76% | 19.11 |

## Stack

Python, PyTorch, tiktoken, NLTK, Matplotlib.

## Running the notebooks

The notebooks were run on Google Colab or Kaggle with a GPU. Open one in Colab, set the runtime to GPU, and run the cells in order. MNIST and CIFAR-10 download through torchvision. The language model trains on a movie script loaded in the notebook.

## About

Coursework for CS 437 Deep Learning, LUMS, Spring 2026, shared as part of my portfolio (course grade: A+). The notebooks keep the original handout text and my implementations, answers and outputs.

If you are taking this course, please do your own work.

**Sameer Rahil**, BS Computer Science, LUMS. More projects at [github.com/SameerRahil](https://github.com/SameerRahil).
