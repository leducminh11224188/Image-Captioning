# 🖼️ Image Captioning with Transformer

An image captioning system that generates natural language descriptions for images using **EfficientNet-B0** as the visual encoder and a **Transformer Decoder** for caption generation, trained on the **Flickr30k** dataset.

🔗 **Live Demo:** [huggingface.co/spaces/leducminh/Image-Captioning](https://huggingface.co/spaces/leducminh/Image-Captioning)

---

## Highlights

- **23.5M parameter** Transformer-based decoder with multi-head attention
- Pretrained **EfficientNet-B0** encoder (frozen) for efficient feature extraction
- **Beam search** decoding with configurable width for high-quality captions
- Trained on **Flickr30k** (31,783 images, 158,915 captions)
- Deployed as a **Gradio** web app on Hugging Face Spaces

---

## Architecture

```
Input Image (224×224)
       │
       ▼
┌──────────────┐
│ EfficientNet  │  Encoder (frozen, pretrained)
│    B0         │  Output: (B, 49, 1280)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Linear Proj  │  1280 → 512
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Transformer  │  3 layers, 8 heads, 512 dim
│   Decoder     │  FFN: 2048, Dropout: 0.3
└──────┬───────┘
       │
       ▼
  Generated Caption
```

| Component | Detail |
|---|---|
| Encoder | EfficientNet-B0 (frozen, ImageNet pretrained) |
| Decoder | Transformer Decoder (3 layers, 8 heads) |
| Embedding dim | 512 |
| FFN dim | 2,048 |
| Dropout | 0.3 |
| Vocab size | 9,964 tokens |
| Total params | 23,481,068 |

---

## Results

### Training (15 epochs, ~38 min on Kaggle GPU)

| Metric | Value |
|---|---|
| Best Train Loss | 3.059 |
| Best Val Loss | 3.387 (epoch 15) |
| Optimizer | AdamW (lr=1e-4, warmup 4000 steps) |
| Batch size | 32 |

### Evaluation Metrics (beam search, beam_width=3, 1000 test samples)

| Metric | Score |
|---|---|
| **BLEU-1** | **0.681** |
| **BLEU-2** | **0.516** |
| **BLEU-3** | **0.387** |
| **BLEU-4** | **0.282** |
| **ROUGE-L** | **0.490** |

### Sample Outputs

| Image | Generated Caption |
|---|---|
| Outdoor scene | *a man in a green shirt and jeans is standing in front of a tree* |
| Workers | *two men are working on a machine* |
| Child playing | *a little girl in a pink dress is sitting on a wooden bench* |
| Building | *a man on a ladder painting a building* |
| Kitchen | *two men cooking in a kitchen* |

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Image-Captioning.git
cd Image-Captioning
pip install -r requirements.txt
```

### Run the Web App

```bash
python app.py
```

The Gradio interface will open at `http://localhost:7860`

### Inference in Code

```python
from model import EncoderCNN, TransformerDecoder, load_vocab, generate_caption
import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab & model
vocab = load_vocab("vocab.pkl")
encoder = EncoderCNN().to(device)
encoder.eval()

model = TransformerDecoder(
    vocab_size=len(vocab), embed_size=512,
    hidden_size=512, num_layers=3, num_heads=8, dropout=0.3
).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = transform(Image.open("your_image.jpg").convert("RGB"))

# Generate caption
caption = generate_caption(model, encoder, image, vocab, device, max_len=20, beam_width=3)
print(caption)
```

---

## Project Structure

```
├── app.py                              # Gradio web interface
├── model.py                            # Model architecture & inference
├── best_model.pth                      # Trained decoder weights (~90MB)
├── vocab.pkl                           # Vocabulary (9,964 tokens)
├── requirements.txt                    # Python dependencies
├── image-captioning_experiment.ipynb   # Training notebook (Kaggle)
├── test_load.py                        # Model loading test
└── README.md
```

---

## Tech Stack

- **PyTorch** — Model training & inference
- **Torchvision** — EfficientNet-B0 backbone & image transforms
- **Gradio** — Web UI
- **Weights & Biases** — Experiment tracking
- **Kaggle** — GPU training environment

---

## Dataset

**Flickr30k** — 31,783 images with 5 human-annotated captions each (158,915 total captions).

| Split | Captions | Ratio |
|---|---|---|
| Train | 143,023 | 90% |
| Validation | 15,892 | 10% |

---

## License

This project is for educational and research purposes.
