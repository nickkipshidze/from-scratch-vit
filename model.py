import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, target_resolution=(256, 256), num_patches=(16, 16), embedding_dim=768, in_channels=3):
        super().__init__()

        assert isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 2
        assert isinstance(num_patches, (list, tuple)) and len(num_patches) == 2

        assert target_resolution[0] % num_patches[0] == 0
        assert target_resolution[1] % num_patches[1] == 0

        self.target_resolution = torch.Size(target_resolution)
        self.num_patches = torch.Size(num_patches)
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.patch_size = (
            self.target_resolution[0] // self.num_patches[0],
            self.target_resolution[1] // self.num_patches[1]
        )

        self.conv_patch = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
    
    def forward(self, input):
        assert len(input.shape) >= 3
        if len(input.shape) == 3: input = input.unsqueeze(dim=0)
        
        if input.shape[-2:] != self.target_resolution:
            input = F.interpolate(input, size=self.target_resolution, mode="bilinear", antialias=True)
        
        X = self.conv_patch(input)
        X = X.flatten(start_dim=2).transpose(1, 2)

        return X

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_p=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.head_dim = embedding_dim // num_heads

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.embedding_dim)

        self.proj_qkv = nn.Linear(in_features=self.embedding_dim, out_features=3*self.embedding_dim)
        self.proj_out = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim*4),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=self.embedding_dim*4, out_features=self.embedding_dim),
        )

    def forward(self, X):
        B, T, _ = X.shape

        qkv = self.proj_qkv(X)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        attn_scores = F.softmax((Q @ K.mT) / (self.head_dim ** 0.5), dim=-1)
        attn_scores = self.dropout(attn_scores)

        X_attn = attn_scores @ V
        X_attn = X_attn.transpose(1, 2).flatten(start_dim=2)
        X_attn = self.proj_out(X_attn)

        X = self.layer_norm_1(X_attn + X)
        X_ffn = self.ffn(X)
        X = self.layer_norm_2(X_ffn + X)

        return X

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, dropout_p=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.layers = nn.Sequential(*[
            TransformerEncoderLayer(embedding_dim, num_heads, dropout_p)
            for layer in range(self.num_layers)
        ])
    
    def forward(self, X):
        X = self.layers(X)
        return X

class LightViTBase(nn.Module):
    def __init__(self, target_resolution=(256, 256), num_patches=(16, 16), embedding_dim=768, num_layers=4, num_heads=8, dropout_p=0.1, in_channels=3):
        super().__init__()

        self.target_resolution = torch.Size(target_resolution)
        self.num_patches = torch.Size(num_patches)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.in_channels = in_channels

        self.embedding = PatchEmbedding(
            target_resolution=self.target_resolution,
            num_patches=self.num_patches,
            embedding_dim=self.embedding_dim,
            in_channels=self.in_channels
        )

        self.positional_embeddings = nn.Parameter(torch.randn(
            size=(1, self.num_patches[0] * self.num_patches[1] + 1, self.embedding_dim)
        ))

        self.cls_token = nn.Parameter(torch.randn(
            size=(1, 1, self.embedding_dim)
        ))

        self.encoder = TransformerEncoder(
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p
        )

    def forward(self, X):
        X = self.embedding(X)
        X = torch.cat([self.cls_token.repeat(X.shape[0], 1, 1), X], dim=1)
        X = X + self.positional_embeddings
        X = self.encoder(X)
        return X

class LightViTClassifier(nn.Module):
    def __init__(self, target_resolution=(256, 256), num_classes=2, num_patches=(16, 16), embedding_dim=768, num_layers=4, num_heads=8, dropout_p=0.1, in_channels=3):
        super().__init__()

        self.backbone = LightViTBase(
            target_resolution,
            num_patches,
            embedding_dim,
            num_layers,
            num_heads,
            dropout_p,
            in_channels
        )

        self.cls_head = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim*4),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=embedding_dim*4, out_features=num_classes)
        )
    
    def forward(self, X):
        X = self.backbone(X)
        X = self.cls_head(X[:, 0])
        return X

if __name__ == "__main__":
    dummy = LightViTClassifier()
    print(dummy)
