import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """Encodes static or dynamic features with 1D convolution."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Attention(nn.Module):
    """Calculates attention over nodes."""

    def __init__(self, hidden_size):
        super().__init__()
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attn = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        return F.softmax(attn, dim=2)


class Pointer(nn.Module):
    """Predicts next action logits."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), device=device, requires_grad=True))

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = self.drop_rnn(rnn_out.squeeze(1))

        if self.num_layers == 1:
            last_hh = self.drop_hh(last_hh)

        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))

        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        logits = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        return logits, last_hh


class DRL4SoSEnvActor(nn.Module):
    """Step-based actor for Gym-style rollout while keeping original pointer architecture."""

    def __init__(self, static_size, dynamic_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()

        if dynamic_size < 1:
            raise ValueError('dynamic_size must be > 0')

        self.static_core_size = static_size - 4

        self.static_encoder = Encoder(self.static_core_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(self.static_core_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self.x0 = nn.Parameter(torch.zeros((1, self.static_core_size, 1), device=device), requires_grad=True)

    def init_episode(self, static, dynamic, decoder_input=None, last_hh=None):
        """Prepare per-episode cached tensors."""
        batch_size = static.size(0)

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        static_core = static[:, : self.static_core_size, :]
        state = {
            'static_core': static_core,
            'static_hidden': self.static_encoder(static_core),
            'dynamic_hidden': self.dynamic_encoder(dynamic),
            'decoder_input': decoder_input,
            'last_hh': last_hh,
        }
        return state

    def forward_step(self, state, action_mask, done_mask=None):
        """One decoding step: masked action sampling + decoder state update."""
        decoder_hidden = self.decoder(state['decoder_input'])
        logits, last_hh = self.pointer(
            state['static_hidden'],
            state['dynamic_hidden'],
            decoder_hidden,
            state['last_hh'],
        )

        masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
        probs = F.softmax(masked_logits, dim=1)

        if self.training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
        else:
            action = torch.argmax(probs, dim=1)
            chosen_prob = torch.gather(probs, 1, action.unsqueeze(1)).squeeze(1)
            logp = torch.log(chosen_prob + 1e-12)

        gather_idx = action.view(-1, 1, 1).expand(-1, self.static_core_size, 1)
        next_decoder_input = torch.gather(state['static_core'], 2, gather_idx).detach()

        if done_mask is not None:
            logp = logp * (~done_mask).float()

        next_state = {
            'static_core': state['static_core'],
            'static_hidden': state['static_hidden'],
            'dynamic_hidden': state['dynamic_hidden'],
            'decoder_input': next_decoder_input,
            'last_hh': last_hh,
        }

        return action, logp, next_state, probs


class StateCritic(nn.Module):
    """State-value critic consistent with original baseline design."""

    def __init__(self, static_size, dynamic_size, hidden_size):
        super().__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        out = F.relu(self.fc1(hidden))
        out = F.relu(self.fc2(out))
        out = self.fc3(out).sum(dim=2)
        return out
