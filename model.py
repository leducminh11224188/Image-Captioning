import torch
import torch.nn as nn
import torchvision.models as models
import math
import re
import pickle


class Vocabulary:
    """Vocabulary class for tokenizing and numericalizing captions."""

    def __init__(self, freq_threshold):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z ]', '', text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]


class EncoderCNN(nn.Module):
    """EfficientNet-B0 encoder to extract visual features.
    Output: (Batch, 49, 1280)
    """

    def __init__(self):
        super(EncoderCNN, self).__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

    def forward(self, images):
        with torch.no_grad():
            features = self.features(images)  # (B, 1280, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (B, 7, 7, 1280)
        features = features.view(features.size(0), -1, features.size(3))  # (B, 49, 1280)
        return features


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerDecoder(nn.Module):
    """Transformer Decoder for Image Captioning."""

    def __init__(self, vocab_size, embed_size=512, hidden_size=512,
                 num_layers=3, num_heads=8, dropout=0.3):
        super(TransformerDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.feature_proj = nn.Linear(1280, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, features, captions):
        memory = self.feature_proj(features)

        tgt = self.embed(captions) * math.sqrt(self.embed_size)
        tgt = self.pos_encoder(tgt)
        tgt = self.embed_dropout(tgt)

        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(
            captions.device
        )

        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
        )

        output = self.dropout(output)
        logits = self.fc_out(output)
        return logits


class _VocabUnpickler(pickle.Unpickler):
    """Custom unpickler that redirects Vocabulary lookups to this module."""

    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab(vocab_path="vocab.pkl"):
    """Load vocabulary from pickle file."""
    with open(vocab_path, 'rb') as f:
        vocab = _VocabUnpickler(f).load()
    return vocab


def load_model(model_path="best_model.pth", vocab_size=None, device="cpu"):
    """Load trained model weights."""
    model = TransformerDecoder(
        vocab_size=vocab_size,
        embed_size=512,
        hidden_size=512,
        num_layers=3,
        num_heads=8,
        dropout=0.3,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_caption(model, encoder, image_tensor, vocab, device,
                     max_len=20, beam_width=3):
    """Generate caption using beam search."""
    model.eval()
    encoder.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)

    sequences = [[vocab.stoi["<start>"]]]
    scores = [0.0]

    for step in range(max_len):
        all_candidates = []

        for i, seq in enumerate(sequences):
            if seq[-1] == vocab.stoi["<end>"]:
                all_candidates.append((seq, scores[i]))
                continue

            tgt = torch.LongTensor([seq]).to(device)

            with torch.no_grad():
                logits = model(features, tgt)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for j in range(beam_width):
                candidate_seq = seq + [topk_indices[0][j].item()]
                candidate_score = scores[i] + topk_log_probs[0][j].item()
                all_candidates.append((candidate_seq, candidate_score))

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = [seq for seq, _ in ordered[:beam_width]]
        scores = [score for _, score in ordered[:beam_width]]

        if all(seq[-1] == vocab.stoi["<end>"] for seq in sequences):
            break

    best_seq = sequences[0]
    caption = [
        vocab.itos[idx]
        for idx in best_seq[1:]
        if idx not in [vocab.stoi["<end>"], vocab.stoi["<pad>"]]
    ]

    return ' '.join(caption)
