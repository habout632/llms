

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from transformers import GPT2Tokenizer
#
# # Define the GPT-2 model architecture
# class GPT2(nn.Module):
#     def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len):
#         super(GPT2, self).__init__()
#         self.token_embedding = nn.Embedding(vocab_size, d_model)
#         self.position_embedding = nn.Embedding(max_seq_len, d_model)
#         self.layers = nn.ModuleList([TransformerBlock(d_model, n_head) for _ in range(n_layer)])
#         self.ln_f = nn.LayerNorm(d_model)
#         self.head = nn.Linear(d_model, vocab_size, bias=False)
#
#     def forward(self, x):
#         b, t = x.size()
#         pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
#         tok_emb = self.token_embedding(x)
#         pos_emb = self.position_embedding(pos)
#         x = tok_emb + pos_emb
#         for layer in self.layers:
#             x = layer(x)
#         x = self.ln_f(x)
#         logits = self.head(x)
#         return logits
#
# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, n_head):
#         super(TransformerBlock, self).__init__()
#         self.ln_1 = nn.LayerNorm(d_model)
#         self.attn = MultiHeadAttention(d_model, n_head)
#         self.ln_2 = nn.LayerNorm(d_model)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, 4 * d_model),
#             nn.GELU(),
#             nn.Linear(4 * d_model, d_model)
#         )
#
#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_head):
#         super(MultiHeadAttention, self).__init__()
#         self.n_head = n_head
#         self.d_k = d_model // n_head
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_o = nn.Linear(d_model, d_model)
#
#     def forward(self, x):
#         b, t, c = x.size()
#         q = self.w_q(x).view(b, t, self.n_head, self.d_k).transpose(1, 2)
#         k = self.w_k(x).view(b, t, self.n_head, self.d_k).transpose(1, 2)
#         v = self.w_v(x).view(b, t, self.n_head, self.d_k).transpose(1, 2)
#
#         att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.d_k)))
#         att = torch.softmax(att, dim=-1)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(b, t, c)
#         return self.w_o(y)
#
# # Prepare the dataset
# class WikiTextDataset(Dataset):
#     def __init__(self, ds, tokenizer, max_length):
#         self.ds = ds
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.ds)
#
#     def __getitem__(self, idx):
#         text = self.ds[idx]['text']
#         encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
#         return encodings['input_ids'].squeeze()
#
# # Initialize tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# vocab_size = len(tokenizer)
# d_model = 768
# n_head = 12
# n_layer = 12
# max_seq_len = 1024
#
# model = GPT2(vocab_size, d_model, n_head, n_layer, max_seq_len)
#
# # Prepare data
# dataset = WikiTextDataset(ds, tokenizer, max_seq_len)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
# # Training loop
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=3e-5)
# criterion = nn.CrossEntropyLoss()
#
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     for batch in dataloader:
#         inputs = batch[:, :-1].to(device)
#         targets = batch[:, 1:].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
#
# print("Training completed!")
