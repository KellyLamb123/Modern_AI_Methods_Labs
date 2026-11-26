import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import TinyLanguageModel
from settings import (
    ARCHITECTURE,
    ARTIFACTS,
    CHECKPOINT_DIR,
    CONTEXT,
    DATA_DIR,
    MAX_NEW_TOKENS,
    MODEL_ID,
    STRATEGIES,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def mps_driver_gib(device):
    if device.type != "mps":
        return 0.0
    return torch.mps.driver_allocated_memory() / (1024 ** 3)


def get_metadata():
    return read_json(ARTIFACTS / "data_summary.json")


def load_tokens(split):
    return torch.load(
        DATA_DIR / f"{split}.pt",
        map_location="cpu",
        weights_only=True,
    )


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        DATA_DIR / "tokenizer",
        local_files_only=True,
    )


class TokenWindows(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        self.length = (len(tokens) - 1) // CONTEXT

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start = index * CONTEXT
        x = self.tokens[start:start + CONTEXT]
        y = self.tokens[start + 1:start + CONTEXT + 1]
        return x, y


class PretrainedLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            use_safetensors=True,
        )

    def forward(self, token_ids):
        return self.hf_model(
            input_ids=token_ids,
            use_cache=False,
        ).logits


def build_model(kind):
    metadata = get_metadata()

    if kind == "scratch":
        return TinyLanguageModel(
            vocab_size=metadata["vocab_size"],
            **ARCHITECTURE,
        )

    if kind == "transfer":
        model = PretrainedLanguageModel()
        assert (
            model.hf_model.config.vocab_size
            == metadata["vocab_size"]
        )
        return model

    raise ValueError(kind)


def cross_entropy(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        loss = cross_entropy(model(x), y)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Неконечная валидационная ошибка")

        count = y.numel()
        total_loss += loss.item() * count
        total_tokens += count

    mean_loss = total_loss / total_tokens

    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "target_tokens": total_tokens,
    }


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    strategy,
    seed,
    device,
    max_new_tokens=MAX_NEW_TOKENS,
):
    options = STRATEGIES[strategy]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("Промпт не должен быть пустым")

    generated = []
    previous_mode = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            context = torch.tensor(
                [ids[-CONTEXT:]],
                dtype=torch.long,
                device=device,
            )

            logits = model(context)[0, -1].float().cpu()

            if strategy == "greedy":
                next_id = int(logits.argmax().item())
            else:
                temperature = options.get("temperature", 1.0)
                logits = logits / temperature

                if strategy == "top_k":
                    k = min(options["top_k"], logits.numel())
                    threshold = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf"),
                    )

                if strategy == "top_p":
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                    )
                    cumulative = torch.softmax(
                        sorted_logits,
                        dim=-1,
                    ).cumsum(dim=-1)

                    remove = cumulative > options["top_p"]
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False

                    logits[sorted_indices[remove]] = float("-inf")

                probabilities = torch.softmax(logits, dim=-1)
                next_id = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    ).item()
                )

            ids.append(next_id)
            generated.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break
    finally:
        model.train(previous_mode)

    return {
        "prompt": prompt,
        "strategy": strategy,
        "parameters": options,
        "seed": seed,
        "new_tokens": len(generated),
        "continuation": tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ),
    }


def load_best_model(kind, device):
    checkpoint = torch.load(
        CHECKPOINT_DIR / f"{kind}_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    metadata = get_metadata()
    actual_hashes = {
        split: metadata[split]["tokens_sha256"]
        for split in ("train", "validation")
    }

    if checkpoint["config"]["data_hashes"] != actual_hashes:
        raise RuntimeError("Данные отличаются от использованных при обучении")

    model = build_model(kind)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint