import pickle
import torch
from model import Vocabulary

# Test vocab
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(f'Vocab loaded: {len(vocab)} tokens')
print(f'pad={vocab.stoi["<pad>"]}, start={vocab.stoi["<start>"]}, end={vocab.stoi["<end>"]}')

# Test tokenizer
tokens = vocab.tokenizer_eng("A dog is running in the park")
indices = vocab.numericalize("A dog is running in the park")
print(f'Tokenize test: {tokens}')
print(f'Numericalize test: {indices}')

# Test model weights
state_dict = torch.load('best_model.pth', map_location='cpu', weights_only=True)
print(f'\nModel loaded: {len(state_dict)} parameter tensors')
for k in list(state_dict.keys())[:5]:
    print(f'  {k}: {state_dict[k].shape}')

# Check vocab_size matches embedding layer
embed_shape = state_dict['embed.weight'].shape
print(f'\nModel embed vocab_size = {embed_shape[0]}, embed_dim = {embed_shape[1]}')
print(f'Vocab size = {len(vocab)}')
if embed_shape[0] == len(vocab):
    print('MATCH - vocab and model are compatible!')
else:
    print('MISMATCH - vocab size does not match model!')
