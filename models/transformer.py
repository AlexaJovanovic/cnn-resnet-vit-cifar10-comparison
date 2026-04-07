import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, d_emb, n_blocks, n_heads, n_classes, p_dropout = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbeddings(3, 32, 32, patch_size, d_emb)
        self.num_patches = ((32 - patch_size) // patch_size + 1)**2
        
        self.pos_embed = nn.Parameter(torch.randn((1, self.num_patches+1, d_emb)))
        self.pos_drop = nn.Dropout(p=p_dropout)

        self.cls_token = nn.Parameter(torch.randn((1, 1, d_emb)))

        self.encoder = nn.ModuleList([
            TransformerBlock(d_emb, n_heads)
            for _ in range(n_blocks)
        ])

        self.cls_head = nn.Linear(in_features=d_emb, out_features=n_classes)
    
    def forward(self, x: torch.Tensor):
        # Embed image patches
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Attach classification token
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Go through encoder
        for block in self.encoder:
            x = block(x)

        # Extract the embedding of the classification token and colapse the sequence length dimension        
        x = x[:, 0, :]
        
        # Calculate class scores
        out = self.cls_head(x)

        return out


class PatchEmbeddings(nn.Module):
    def __init__(self, C, H, W, patch_size, d_emb):
        super().__init__()
        self.d_emb = d_emb

        self.H_out = (H - patch_size)// patch_size + 1 # H_out = (H-K)/S + 1
        self.W_out = (W - patch_size)// patch_size + 1 # W_out = (W-K)/S + 1
        
        self.conv = nn.Conv2d(in_channels=C, out_channels=d_emb, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H_out, W_out)
        x = self.conv(x)
        
        # (B, D, H_out, W_out) -> (B, D, H_out * W_out)
        x = x.view(-1, self.d_emb, self.H_out * self.W_out)
        
        # (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2).contiguous()
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_emb, n_head):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_emb)
        self.msa = MultiheadSelfAttention(n_head=n_head, d_emb=d_emb)
        
        self.layer_norm2 = nn.LayerNorm(d_emb)
        self.mlp = MultiLayerPerceptron(d_emb=d_emb)

    def forward(self, x):

        x = x + self.msa(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))

        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, n_head, d_emb, p_dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_emb = d_emb
        self.d_qkv = d_emb // n_head
        self.score_scale : float = self.d_qkv ** -0.5
        
        self.qkv_expansion = nn.Linear(in_features=d_emb, out_features=3*d_emb)
        self.dropout = nn.Dropout(p=p_dropout)
        self.out_projection = nn.Linear(in_features=d_emb, out_features=d_emb)

    def forward(self, x : torch.Tensor):
        # x is shaped at the begining (Batch, Num_Patches, Embed_Dim)
        B, N, _ = x.shape
        # Expand batch of original embeddings to batch of keys, queries and values for each embedding.
        # (Batch, Num_Patches, Embed_Dim) -> (Batch, Num_Patches, 3*Embed_Dim)
        x = self.qkv_expansion(x)
        
        # I want Q, K, V matrices of size (Num_Patches, Embed_Dim/Num_Heads) for each head.
        # So i need 3 tensors sized (Num_Heads, Num_Patches, Embed_Dim/Num_Heads) expanded accros the whole batch
        # (Batch, Num_Patches, 3*Embed_Dim) -> (Batch, Num_Patches, 3, Embed_Dim) -> (Batch, Num_Patches, 3, Num_Heads, Embed_Dim / Num_Heads)
        x = x.reshape(B, N, 3, self.n_head, -1)

        # (Batch, Num_Patches, 3, Num_Heads, Embed_Dim / Num_Heads) -> (Batch, Num_Heads, 3, Num_Patches, Embed_Dim / Num_Heads)
        x = x.transpose(1, 3)

        # Create K, Q, V matrices of shape (Batch, Num_Heads, Num_Patches, Embed_Dim / Num_Heads)
        q, k, v = x.unbind(2)

        # Compute attention softmax(Q*K.T/sqrt(d_query))*V
        # q @ k -> (B, H, N, N)
        e = q @ k.transpose(-2, -1) 
        e = e * self.score_scale
        attn = F.softmax(e, dim=-1)
        # Regularization with dropout
        attn = self.dropout(attn)
        
        # attn @ v: (B, H, N, N) @ (B, H, N, d_qkv) -> (B, H, N, d_qkv)
        out = attn @ v
        
        # (B, H, N, d_qkv) -> (B, N, H, d_qkv) -> (B, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        out = self.out_projection(out)

        return out
    

class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_emb: int, p_dropout = 0.1):
        super().__init__()

        self.lin1 = nn.Linear(in_features=d_emb, out_features=4*d_emb)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(in_features=4*d_emb, out_features=d_emb)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        # Expanison of embeddings (B, N, D) -> (B, N, 4D)
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        # Projecting back to embedding space (B, N, 4D) -> (B, N, D)
        x = self.lin2(x)
        x = self.dropout(x)
        
        return x
    

if __name__ == "__main__":
    vit = VisionTransformer(patch_size=4, d_emb=256, n_blocks=6, n_heads=8, n_classes=10)

    total_params = sum(p.numel() for p in vit.parameters() if p.requires_grad)

    print(f"Total Trainable Parameters: {total_params:,}")

