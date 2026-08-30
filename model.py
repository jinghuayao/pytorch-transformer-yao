import torch 
import torch.nn as nn
import math

print(f"{torch.__version__}")

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable param
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable param


    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True) # (B, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (B, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # (B, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


    

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)


    def forward(self, x):
        # (B, seq_len) -> (B, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # make a matrix of size (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        # make a vector of shape (seq_len, ) then unsequeeze to make it shape (seq_len, 1) for broadcasting
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # make a vector of shape (d_model // 2, )
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        # note the broadcasting of position * div_term yields a shape (seq_len, d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (-2 * i / d_model)))
        pe[:, 1::2] = torch.cos(position * div_term[:self.d_model // 2]) # cos(position * (10000 ** (-2 * i / d_model))); [:d_model//2] deals with d_model being odd case
        # add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (seq_len, d_model) -> (1, seq_len, d_model)
        # register the positional encoding as a buffer
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (B, seq_len, d_model)
        return self.dropout(x)

        

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        # sublayer shall preverse the input shape
        return x + self.dropout(sublayer(self.norm(x)))
        


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # embedding dimension, i.e., model dimension
        self.h = h # number of heads
        assert d_model % h == 0, "d_model shall be divisible by h"
        self.d_k = self.d_model // self.h 

        # weight matrices and linear transform
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # we prepare query, key, value such that their shape is (B, h, seq_len, d_k)
        # you can use einsum, matmul or @
        # attention_scores = torch.einsum('bhqd, bhkd -> bhqk', query, key) / math.sqrt(d_k)
        # attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k) # (B, h, seq_len, seq_len)

        # the mask shall be of shape (..., seq_len, seq_len) or broadcasted to the shape
        if mask is not None:
            # make elements above diagonal to be -float("inf")
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # (B, h, seq_len, seq_len)

        # shape: attention_scores @ value: (B, h, seq_len, seq_len) @ (B, h, seq_len, d_k) -> (B, h, seq_len, d_k)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        # q, k, v shall be of shape: (B, seq_len, d_model), mask shall be of shape (..., seq_len, seq_len)
        query = self.w_q(q) # (B, seq_len, d_model) -> (B, seq_len, d_model) 
        key = self.w_k(k) # (B, seq_len, d_model) -> (B, seq_len, d_model) 
        value = self.w_v(v) # (B, seq_len, d_model) -> (B, seq_len, d_model) 

        # split the d_model dim to be h * d_k to get shape (B, seq_len, h, d_k), then transpose to
        # a shape (B, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # calculate attention
        # x shape: (B, h, seq_len, d_k), attention_scores shape: (B, h, seq_len, seq_len)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # combine all heads:  (B, h, seq_len, d_k) -> (B, seq_len, h, d_k) -> (B, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # linear transform with w_0
        return self.w_o(x)
    

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    
        


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList[EncoderBlock]) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)



class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList[DecoderBlock]) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # x: (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        # (B, seq_len, d_model) shape remains
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (B, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        # need: tgt input, encoder_output, src_mask, tgt_mask
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    

    def project(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, tgt_vocab_size)
        return self.projection_layer(x)



def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6,
                      h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> None:

    # 1. create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 2. create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 3. N encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    # 4. N decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)



    # 3-continued: N encoder blocks form the encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))


    # 4-continued: N decoder blocks form the decoder
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))


    # 5. create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)


    # 6. create the whole transform
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    

    # lastly, initialize the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)


    return transformer

if __name__ == "__main__":
    print("Run Through Without Issue.")



