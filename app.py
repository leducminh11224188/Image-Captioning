import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import (
    EncoderCNN,
    TransformerDecoder,
    load_vocab,
    generate_caption,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
VOCAB_PATH = "vocab.pkl"

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Load model & vocab once at startup
# ---------------------------------------------------------------------------
print("Loading vocabulary...")
vocab = load_vocab(VOCAB_PATH)
print(f"Vocabulary size: {len(vocab)}")

print("Loading encoder...")
encoder = EncoderCNN().to(DEVICE)
encoder.eval()

print("Loading decoder...")
model = TransformerDecoder(
    vocab_size=len(vocab),
    embed_size=512,
    hidden_size=512,
    num_layers=3,
    num_heads=8,
    dropout=0.3,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def predict(image: Image.Image, beam_width: int = 3, max_length: int = 20):
    """Generate a caption for the uploaded image."""
    if image is None:
        return "Please upload an image."

    image = image.convert("RGB")
    img_tensor = transform(image)

    caption = generate_caption(
        model=model,
        encoder=encoder,
        image_tensor=img_tensor,
        vocab=vocab,
        device=DEVICE,
        max_len=max_length,
        beam_width=beam_width,
    )

    return caption


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
examples = []  # Add local image paths here if you want example images

with gr.Blocks(title="Image Captioning") as demo:
    gr.Markdown(
        """
        # 🖼️ Image Captioning
        Upload an image and the model will generate a caption describing it.

        **Model:** EfficientNet-B0 (encoder) + Transformer Decoder  
        **Dataset:** Flickr30k
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            beam_width = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="Beam Width",
                info="Higher = better quality but slower",
            )
            max_length = gr.Slider(
                minimum=5, maximum=50, value=20, step=1,
                label="Max Caption Length",
            )
            generate_btn = gr.Button("Generate Caption", variant="primary")

        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Generated Caption",
                lines=3,
            )

    generate_btn.click(
        fn=predict,
        inputs=[image_input, beam_width, max_length],
        outputs=caption_output,
    )

    image_input.change(
        fn=predict,
        inputs=[image_input, beam_width, max_length],
        outputs=caption_output,
    )

    if examples:
        gr.Examples(examples=examples, inputs=image_input)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
