import argparse
import math
import platform
import resource
import statistics
import time

import torch
from torch.utils.data import DataLoader

from common import (
    TokenWindows,
    build_model,
    cross_entropy,
    evaluate_loss,
    generate,
    get_device,
    get_metadata,
    load_tokenizer,
    load_tokens,
    mps_driver_gib,
    read_json,
    set_seed,
    synchronize,
    write_json,
)
from settings import (
    ARCHITECTURE,
    ARTIFACTS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CONTEXT,
    HARDWARE,
    PROMPTS,
    RUNS,
    SEED,
)


def make_loader(tokens, shuffle, seed):
    return DataLoader(
        TokenWindows(tokens),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


def peak_process_rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2
    return value / divisor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["scratch", "transfer"])

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = get_device()
    metadata = get_metadata()

    config = {
        "kind": args.kind,
        **RUNS[args.kind],
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "context": CONTEXT,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "warmup_ratio": 0.05,
        "architecture": (
            ARCHITECTURE if args.kind == "scratch" else None
        ),
        "data_hashes": {
            split: metadata[split]["tokens_sha256"]
            for split in ("train", "validation")
        },
    }

    last_path = CHECKPOINT_DIR / f"{args.kind}_last.pt"
    best_path = CHECKPOINT_DIR / f"{args.kind}_best.pt"
    benchmark_path = ARTIFACTS / f"{args.kind}_benchmark.json"

    if not args.benchmark:
        if args.resume and not last_path.exists():
            raise FileNotFoundError("Checkpoint для продолжения не найден")
        if not args.resume and last_path.exists():
            raise RuntimeError(
                "Обучение уже запускалось. Используйте --resume."
            )

        if not benchmark_path.exists():
            raise RuntimeError("Сначала выполните запуск с --benchmark")

        benchmark = read_json(benchmark_path)

        if benchmark["config"] != config:
            raise RuntimeError(
                "Настройки изменились. Повторите --benchmark."
            )

        if benchmark["estimated_train_and_validation_seconds"] > 7200:
            raise RuntimeError(
                "Оценка времени превышает два часа. "
                "Перед запуском необходимо скорректировать настройки."
            )

    train_tokens = load_tokens("train")
    validation_tokens = load_tokens("validation")

    validation_loader = make_loader(
        validation_tokens,
        shuffle=False,
        seed=SEED,
    )
    train_loader = make_loader(
        train_tokens,
        shuffle=True,
        seed=SEED + 1,
    )

    tokenizer = load_tokenizer()
    model = build_model(args.kind).to(device)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = max(1, int(total_steps * config["warmup_ratio"]))

    def learning_rate_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )

    def update_batch(x, y):
        optimizer.zero_grad(set_to_none=True)

        x = x.to(device)
        y = y.to(device)

        loss = cross_entropy(model(x), y)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Обнаружена неконечная обучающая ошибка")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config["gradient_clip"],
            error_if_nonfinite=True,
        )

        optimizer.step()
        scheduler.step()

        return loss.item(), y.numel()

    print("Модель:", args.kind)
    print("Устройство:", device)
    print("Обучаемых параметров:", parameters)
    print("Батчей за эпоху:", len(train_loader))
    print("Эпох:", config["epochs"])
    print("Параметры:", config)

    if args.benchmark:
        model.train()
        durations = []

        for step, (x, y) in enumerate(train_loader, start=1):
            synchronize(device)
            started = time.perf_counter()

            update_batch(x, y)

            synchronize(device)
            durations.append(time.perf_counter() - started)

            if step >= 20:
                break

        measured = durations[5:] or durations
        seconds_per_step = statistics.mean(measured)

        synchronize(device)
        started = time.perf_counter()
        evaluate_loss(model, validation_loader, device)
        synchronize(device)
        validation_seconds = time.perf_counter() - started

        estimated = config["epochs"] * (
            len(train_loader) * seconds_per_step
            + validation_seconds
        )

        result = {
            "config": config,
            "device": str(device),
            "seconds_per_training_step": seconds_per_step,
            "validation_seconds": validation_seconds,
            "estimated_train_and_validation_seconds": estimated,
        }

        write_json(benchmark_path, result)

        print(f"Среднее время шага: {seconds_per_step:.3f} с")
        print(f"Одна валидация: {validation_seconds:.1f} с")
        print(f"Оценка обучения и валидаций: {estimated / 60:.1f} мин")
        print("Измерительная модель не сохранена.")
        print("BENCHMARK_OK")
        return

    history = []
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    sampled_peak_mps = 0.0

    def save_checkpoint(epoch, path):
        state = {
            "epoch": epoch,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "sampled_peak_mps_driver_gib": sampled_peak_mps,
            "torch_rng": torch.get_rng_state(),
            "mps_rng": (
                torch.mps.get_rng_state()
                if device.type == "mps"
                else None
            ),
        }

        torch.save(state, path)

    if args.resume:
        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["config"] != config:
            raise RuntimeError(
                "Настройки отличаются от сохранённого запуска"
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        best_epoch = checkpoint["best_epoch"]
        sampled_peak_mps = checkpoint["sampled_peak_mps_driver_gib"]

        torch.set_rng_state(checkpoint["torch_rng"])

        if device.type == "mps" and checkpoint["mps_rng"] is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
    else:
        synchronize(device)
        started = time.perf_counter()
        baseline = evaluate_loss(model, validation_loader, device)
        synchronize(device)

        history.append({
            "epoch": 0,
            "train_loss": None,
            "val_loss": baseline["loss"],
            "val_perplexity": baseline["perplexity"],
            "train_seconds": 0.0,
            "validation_seconds": time.perf_counter() - started,
        })

        write_json(
            ARTIFACTS / f"{args.kind}_history.json",
            history,
        )
        save_checkpoint(0, last_path)

        print(
            f"До обучения: val_loss={baseline['loss']:.4f}, "
            f"PPL={baseline['perplexity']:.2f}"
        )

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()

        epoch_loader = make_loader(
            train_tokens,
            shuffle=True,
            seed=SEED + epoch,
        )

        total_loss = 0.0
        total_targets = 0

        synchronize(device)
        started = time.perf_counter()

        for step, (x, y) in enumerate(epoch_loader, start=1):
            loss, count = update_batch(x, y)

            total_loss += loss * count
            total_targets += count

            sampled_peak_mps = max(
                sampled_peak_mps,
                mps_driver_gib(device),
            )

            if step % 200 == 0:
                print(
                    f"epoch={epoch}, "
                    f"step={step}/{len(epoch_loader)}, "
                    f"loss={loss:.4f}"
                )

        synchronize(device)
        train_seconds = time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        validation = evaluate_loss(
            model,
            validation_loader,
            device,
        )
        synchronize(device)
        validation_seconds = time.perf_counter() - started

        sample = generate(
            model,
            tokenizer,
            prompt=PROMPTS[0],
            strategy="temperature",
            seed=SEED,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_targets,
            "val_loss": validation["loss"],
            "val_perplexity": validation["perplexity"],
            "train_target_tokens": total_targets,
            "validation_target_tokens": validation["target_tokens"],
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "learning_rate_after_epoch": optimizer.param_groups[0]["lr"],
            "sample": sample,
        }

        history.append(row)

        improved = validation["loss"] < best_loss
        if improved:
            best_loss = validation["loss"]
            best_epoch = epoch

        write_json(
            ARTIFACTS / f"{args.kind}_history.json",
            history,
        )

        save_checkpoint(epoch, last_path)
        if improved:
            save_checkpoint(epoch, best_path)

        print(
            f"EPOCH {epoch}: "
            f"train_loss={row['train_loss']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, "
            f"PPL={row['val_perplexity']:.2f}, "
            f"train={train_seconds:.1f}s, "
            f"validation={validation_seconds:.1f}s"
        )

    summary = {
        "kind": args.kind,
        "config": config,
        "device": str(device),
        "hardware": HARDWARE,
        "actual_macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "trainable_parameters": parameters,
        "completed_epochs": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_perplexity": math.exp(best_loss),
        "baseline_val_perplexity": history[0]["val_perplexity"],
        "training_seconds": sum(
            row["train_seconds"]
            for row in history
        ),
        "validation_seconds": sum(
            row["validation_seconds"]
            for row in history
        ),
        "peak_process_rss_gib": peak_process_rss_gib(),
        "sampled_peak_mps_driver_gib": sampled_peak_mps,
    }

    write_json(
        ARTIFACTS / f"{args.kind}_summary.json",
        summary,
    )

    print("Лучшая эпоха:", best_epoch)
    print("Лучшая валидационная PPL:", summary["best_val_perplexity"])
    print("Время обучения, мин:", summary["training_seconds"] / 60)
    print("TRAINING_OK")


if __name__ == "__main__":
    main()