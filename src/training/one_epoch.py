import torch
import time
from src.training.gradscaler import *

def train_one_epoch(
    model,
    criterion,
    optimizer,
    train_iter,
    steps_per_epoch: int,
    device: str,
    log_every: int = 50,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",   # "bf16" o "fp16"
    scaler=None,
    grad_clip: float | None = None,
):
    """
    Entrena StyA2KNet por 1 época usando PerceptualLoss.

    Devuelve:
      {
        "loss":    loss_promedio,
        "content": content_loss_promedio,
        "style":   style_loss_promedio,
        "steps":   número de steps efectivamente ejecutados
      }
    """
    model.train()

    if device == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        start_time = time.time()

    running_loss = 0.0
    running_content = 0.0
    running_style = 0.0

    for step, (x_c, x_s) in enumerate(train_iter):
        if step >= steps_per_epoch:
            break

        x_c = x_c.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device=device, enabled=amp_enabled, dtype=amp_dtype):
            y = model(x_c, x_s)               # [B,3,252,252]
            loss, parts = criterion(y, x_c, x_s)
            lc = parts["content"]
            ls = parts["style"]

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss    += loss.item()
        running_content += lc
        running_style   += ls

        if (step + 1) % log_every == 0:
            avg_loss = running_loss / (step + 1)
            avg_lc   = running_content / (step + 1)
            avg_ls   = running_style / (step + 1)

            if device == "cuda":
                end_evt.record()
                torch.cuda.synchronize()
                elapsed_s = start_evt.elapsed_time(end_evt) / 1000.0
            else:
                elapsed_s = time.time() - start_time

            print(
                f"[step {step+1:4d}/{steps_per_epoch}] "
                f"loss={avg_loss:.4f}  "
                f"content={avg_lc:.4f}  style={avg_ls:.4f}  "
                f"time={elapsed_s:.1f}s"
            )

    steps = min(steps_per_epoch, step + 1)
    return {
        "loss":    running_loss / steps,
        "content": running_content / steps,
        "style":   running_style / steps,
        "steps":   steps,
    }