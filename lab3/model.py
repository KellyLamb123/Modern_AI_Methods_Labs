import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from settings import (
    ARCHITECTURE,
    ARTIFACTS,
    CONTEXT,
    SEED,
)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context, dropout):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model должен делиться на n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        future_mask = torch.ones(
            context,
            context,
            dtype=torch.bool,
        ).triu(diagonal=1)

        self.register_buffer(
            "future_mask",
            future_mask,
            persistent=False,
        )

    def forward(self, x):
        batch, length, width = x.shape

        qkv = self.qkv(x).reshape(
            batch,
            length,
            3,
            self.n_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.future_mask[:length, :length],
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        output = weights @ v
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch, length, width)

        return self.output_dropout(self.output(output))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context, dropout):
        super().__init__()

        self.ln_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            n_heads,
            context,
            dropout,
        )

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln_attention(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context=CONTEXT,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.context = context

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model,
                    n_heads,
                    context,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._initialize)

        # Общая матрица входных эмбеддингов и выходной проекции.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        _, length = token_ids.shape

        if length > self.context:
            raise ValueError("Вход превышает окно контекста")

        positions = torch.arange(
            length,
            device=token_ids.device,
        )

        hidden = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        hidden = self.embedding_dropout(hidden)
        hidden = self.blocks(hidden)
        hidden = self.final_norm(hidden)

        return self.lm_head(hidden)


def draw_architecture(vocab_size, parameters):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Token IDs: [batch, 128]",
        f"Token embedding ({vocab_size} x 128)\n"
        "+ learned position embedding (128 x 128)",
        "Transformer block x 2\n"
        "LN -> causal attention (4 heads) -> residual\n"
        "LN -> FFN (128 -> 512 -> 128, GELU) -> residual",
        "Final LayerNorm",
        "Linear vocabulary projection\n"
        "(weight tied to token embedding)",
        f"Logits: [batch, 128, {vocab_size}]",
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.axis("off")

    for index, label in enumerate(labels):
        y = len(labels) - 1 - index
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": "#edf3fa",
                "edgecolor": "#385778",
            },
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.67),
                xytext=(0.5, y - 0.32),
                arrowprops={"arrowstyle": "->"},
            )

    ax.set_title(
        f"Decoder-only Transformer: {parameters:,} parameters",
        pad=20,
    )

    fig.tight_layout()
    fig.savefig(ARTIFACTS / "architecture.png", dpi=180)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)

    metadata = json.loads(
        (ARTIFACTS / "data_summary.json").read_text(encoding="utf-8")
    )

    vocab_size = metadata["vocab_size"]
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        **ARCHITECTURE,
    ).to(device)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameters <= 20_000_000

    x = torch.randint(
        0,
        vocab_size,
        (2, 32),
        device=device,
    )

    model.eval()
    with torch.no_grad():
        original = model(x)

        changed = x.clone()
        changed[:, 16:] = (changed[:, 16:] + 1) % vocab_size
        modified = model(changed)

        torch.testing.assert_close(
            original[:, :16],
            modified[:, :16],
            atol=1e-5,
            rtol=1e-5,
        )

    print(model)
    print("Устройство:", device)
    print("Обучаемых параметров:", parameters)
    print("Форма выхода:", tuple(original.shape))
    print("CAUSAL_MASK_OK")

    model.train()
    logits = model(x[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        x[:, 1:].reshape(-1),
    )

    loss.backward()

    assert torch.isfinite(loss).item()
    assert all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    print("Проверочная cross-entropy:", loss.item())
    print("BACKWARD_OK")

    draw_architecture(vocab_size, parameters)

    (ARTIFACTS / "architecture.json").write_text(
        json.dumps(
            {
                **ARCHITECTURE,
                "context": CONTEXT,
                "vocab_size": vocab_size,
                "trainable_parameters": parameters,
                "weight_tying": True,
                "device_used_for_check": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()