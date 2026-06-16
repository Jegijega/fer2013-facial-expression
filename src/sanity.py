from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .data import NUM_CLASSES, IMG_SIZE


@torch.no_grad()
def forward_check(model: nn.Module, device, batch_size: int = 8) -> Dict[str, float]:
    # a fresh net should give loss close to ln(num_classes)
    model.eval()
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    expected = math.log(NUM_CLASSES)
    return {
        "output_shape_ok": float(tuple(logits.shape) == (batch_size, NUM_CLASSES)),
        "initial_loss": loss,
        "expected_loss_ln_C": expected,
        "loss_close_to_expected": float(abs(loss - expected) < 0.7),
    }


def gradient_check(model: nn.Module, device, batch_size: int = 8) -> Dict[str, float]:
    model.train()
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    model.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    n_params = n_with_grad = n_finite = 0
    total_norm = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n_params += 1
        if p.grad is not None:
            n_with_grad += 1
            if torch.isfinite(p.grad).all():
                n_finite += 1
            total_norm += float(p.grad.norm()) ** 2
    return {
        "params": n_params,
        "params_with_grad": n_with_grad,
        "params_finite_grad": n_finite,
        "all_params_have_finite_grad": float(n_params == n_with_grad == n_finite),
        "grad_global_norm": math.sqrt(total_norm),
    }


def overfit_single_batch(model, device, steps=200, batch_size=16, lr=1e-3, target_acc=0.99):
    # if the model can't memorize one batch, the bug is in the code, not the data
    model.train()
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses, accs = [], []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append((logits.argmax(1) == y).float().mean().item())

    return {
        "losses": losses,
        "accuracies": accs,
        "final_loss": losses[-1],
        "final_acc": accs[-1],
        "reached_target": accs[-1] >= target_acc,
    }


def run_all_checks(model: nn.Module, device, verbose: bool = True) -> Dict:
    fwd = forward_check(model, device)
    grad = gradient_check(model, device)
    overfit = overfit_single_batch(model, device)
    if verbose:
        print(f"  initial loss      : {fwd['initial_loss']:.4f} "
              f"(expected ~ ln(7) = {fwd['expected_loss_ln_C']:.4f})")
        print(f"  finite gradients  : {bool(grad['all_params_have_finite_grad'])}")
        print(f"  overfit batch acc : {overfit['final_acc']:.4f} "
              f"(reached ~100%: {overfit['reached_target']})")
    return {"forward": fwd, "gradient": grad, "overfit_batch": overfit}
