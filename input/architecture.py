# Architecture extracted verbatim from the supplied research code.


import torch
import torch.nn as nn
import torch.nn.functional as F


class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T):
        super().__init__()
        self.N = N
        self.M = M
        self.T = T

        self.encoder_lstm = nn.LSTMCell(input_size=N, hidden_size=M)
        self.W_e = nn.Linear(2 * M, T)
        self.U_e = nn.Linear(T, T, bias=False)
        self.v_e = nn.Linear(T, 1, bias=False)

    def forward(self, inputs):
        B = inputs.size(0)
        device_ = inputs.device

        encoded_inputs = torch.zeros((B, self.T, self.M), device=device_)
        h_tm1 = torch.zeros((B, self.M), device=device_)
        s_tm1 = torch.zeros((B, self.M), device=device_)

        x_perm = inputs.permute(0, 2, 1)

        for t in range(self.T):
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)
            x = self.W_e(h_c_concat).unsqueeze(1).repeat(1, self.N, 1)
            y = self.U_e(x_perm)
            z = torch.tanh(x + y)
            e_k_t = self.v_e(z).squeeze(-1)
            alpha_k_t = F.softmax(e_k_t, dim=1)

            weighted_inputs = alpha_k_t * inputs[:, t, :]
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            encoded_inputs[:, t, :] = h_tm1

        return encoded_inputs

class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T_enc, horizon):
        super().__init__()
        self.M = M
        self.P = P
        self.T_enc = T_enc
        self.horizon = horizon

        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=P)
        self.W_d = nn.Linear(2 * P, M)
        self.U_d = nn.Linear(M, M, bias=False)
        self.v_d = nn.Linear(M, 1, bias=False)
        self.w_tilda = nn.Linear(M + 1, 1)
        self.W_y = nn.Linear(P + M, P)
        self.v_y = nn.Linear(P, 1)

    def forward(self, encoded_inputs, y=None):
        B = encoded_inputs.size(0)
        device_ = encoded_inputs.device

        d_tm1 = torch.zeros((B, self.P), device=device_)
        s_prime_tm1 = torch.zeros((B, self.P), device=device_)
        y_prev = torch.zeros((B, 1), device=device_)

        outputs = []

        for t in range(self.horizon):
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            x1 = self.W_d(d_s_prime_concat).unsqueeze(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            beta_i_t = F.softmax(l_i_t, dim=1)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)

            y_c_concat = torch.cat((c_t, y_prev), dim=1)
            y_tilda_t = self.w_tilda(y_c_concat)

            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
            d_c_concat = torch.cat((d_tm1, c_t), dim=1)
            y_hat_t = self.v_y(self.W_y(d_c_concat))
            outputs.append(y_hat_t)

            y_prev = y_hat_t if y is None else y[:, t, :1]

        return torch.stack(outputs, dim=1)

class cLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, out_day):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_day = out_day

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, out_day)

    def init_hidden(self, batch_size, device_):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device_)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device_)
        return h_0, c_0

    def forward(self, x):
        batch_size = x.size(0)
        device_ = x.device
        h_0, c_0 = self.init_hidden(batch_size, device_)
        _, (h_n, _) = self.lstm(x, (h_0, c_0))
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output

class MeteorologicalVariableAttention(nn.Module):
    def __init__(self, tw, ftw, num_meteo_vars, d_query, d_model, dropout=0.0):
        super().__init__()
        self.total_length = tw + ftw
        self.num_meteo_vars = num_meteo_vars
        self.d_model = d_model

        self.var_embed = nn.Linear(self.total_length, d_model)
        self.query_proj = nn.Linear(d_query, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("scale", torch.sqrt(torch.tensor(float(d_model))))

    def forward(self, x_met_hist, x_met_fut, query_vec):
        x_met_total = torch.cat([x_met_hist, x_met_fut], dim=1)
        x_met_t = x_met_total.permute(0, 2, 1)

        var_repr = F.relu(self.var_embed(x_met_t))

        Q = self.query_proj(query_vec).unsqueeze(1)
        K = self.key_proj(var_repr)
        V = self.value_proj(var_repr)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V).squeeze(1)

        return context, attn_weights.squeeze(1)

class AMBER(nn.Module):
    def __init__(
        self,
        N,
        M,
        P,
        T,
        ftw,
        hidden_dim_lstm,
        num_layers,
        meteo_indices,
        meteo_d_model=24,
        fusion_hidden=32,
        meteo_dropout=0.0
    ):
        super().__init__()

        self.meteo_indices = meteo_indices

        self.encoder = InputAttentionEncoder(N, M, T)
        self.decoder = TemporalAttentionDecoder(M, P, T, ftw)
        self.lstm = cLSTM(
            input_size=1,
            hidden_dim=hidden_dim_lstm,
            num_layers=num_layers,
            out_day=ftw
        )

        self.meteo_attention = MeteorologicalVariableAttention(
            tw=T,
            ftw=ftw,
            num_meteo_vars=len(meteo_indices),
            d_query=M,
            d_model=meteo_d_model,
            dropout=meteo_dropout
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(ftw + meteo_d_model, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, ftw)
        )

    def forward(self, x, x_met_fut, return_attention=False):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        daleed_out = self.lstm(dec_out)

        x_met_hist = x[:, :, self.meteo_indices]
        g_demand = enc_out[:, -1, :]

        c_met, attn_weights = self.meteo_attention(x_met_hist, x_met_fut, g_demand)

        combined = torch.cat([daleed_out, c_met], dim=1)
        final_out = self.fusion_head(combined)

        if return_attention:
            return final_out, attn_weights
        return final_out
