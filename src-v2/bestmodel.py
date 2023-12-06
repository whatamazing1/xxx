import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
conv_filters = [[1,32],[3,32],[5,64],[7,128]]
embedding_size = output_dim = 256
d_ff = 256
n_heads = 8
d_k = 16
n_layer = 1

smi_vocab_size = 53
seq_vocab_size = 21

seed = 990721

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()
def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]



class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)

        elif type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        # The dimension of concatenated vectors obtained from multiple one-dimensional convolutions
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)


    def forward(self, inputs):
        embeds = self.embed(inputs).transpose(-1,-2) # (batch_size, embedding_size, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2) # (batch_size, seq_len, num_filters)
        res_embed = self.projection(res_embed)
        return res_embed

# A highway neural network, where the dimensions of input and output are the same, similar to the ResNet principle
class Highway(nn.Module):
    def __init__(self, input_dim, num_layers, acticvation = F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([nn.Linear(input_dim, input_dim*2) for _ in range(num_layers)])
        self.acticvation = acticvation
        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)
            # The output dimension is 2 * input_ Dim, the first half is used for hidden layer output, and the second half is used for gate output
            hidden = self.acticvation(projected_inputs[:,:self.input_dim])
            gate = torch.sigmoid(projected_inputs[:,self.input_dim:])
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        return curr_inputs

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim = - 1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(embedding_size, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_k, embedding_size, bias=False)
        self.ln = nn.LayerNorm(embedding_size)
        self.attn = SelfAttention()
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size = input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = self.attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_k) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return self.ln(output)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, embedding_size, bias=False)
        )
        self.ln = nn.LayerNorm(embedding_size)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output) # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi = MultiHeadAttention()
        self.feed = FeedForward()

    def forward(self, en_input, attn_mask):
        context = self.multi(en_input, en_input, en_input, attn_mask)
        output = self.feed(context+en_input)
        return output


class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb = ConvEmbedding(seq_vocab_size, embedding_size, conv_filters, output_dim, 'seq')
        self.poc_emb = ConvEmbedding(seq_vocab_size, embedding_size, conv_filters, output_dim, 'poc')
        self.highway = Highway(embedding_size, n_layer)
        self.layers  = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])
    def forward(self, seq_input):
        output_emb = self.seq_emb(seq_input)
        enc_self_attn_mask = get_attn_pad_mask(seq_input, seq_input)
        for layer in self.layers:
            output_emb = layer(output_emb,enc_self_attn_mask)
        return output_emb


class Smi_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = self.emb = ConvEmbedding(smi_vocab_size, embedding_size, conv_filters, output_dim, 'seq')
        self.highway = Highway(embedding_size, n_layer)
        self.layers  = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])
    def forward(self,smi_input):
        output_emb = self.emb(smi_input)
        enc_self_attn_mask = get_attn_pad_mask(smi_input, smi_input)
        for layer in self.layers:
            output_emb = layer(output_emb,enc_self_attn_mask)
        return output_emb


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = Seq_Encoder()
        self.smi_encoder = Smi_Encoder()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Squeeze(),
            nn.Linear(1024,256),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            Squeeze())
    def forward(self,seq_encode, smi_encode):
        seq_outputs = self.seq_encoder(seq_encode)
        smi_outputs = self.smi_encoder(smi_encode)
        score = torch.matmul(seq_outputs, smi_outputs.transpose(-1, -2))/np.sqrt(embedding_size)
        final_outputs = self.fc(score)
        return final_outputs


