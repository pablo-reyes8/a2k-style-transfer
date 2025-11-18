import os, time
import torchvision.utils as vutils
import torch

from src.training.gradscaler import * 
from src.training.chekpoint import *
from src.training.one_epoch import * 
from src.data.load_data import make_train_iterator


IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)  
IMAGENET_STD_T  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)   

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: [3,H,W] o [B,3,H,W] normalizado a ImageNet -> devuelve mismo shape en [0,1].
    """
    # Asegurarnos de tener shape [B,3,H,W] para aplicar mean/std broadcast
    squeeze_back = False
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [1,3,H,W]
        squeeze_back = True

    x = x.to(dtype=torch.float32)
    mean = IMAGENET_MEAN_T.to(x.device, dtype=x.dtype)
    std  = IMAGENET_STD_T.to(x.device, dtype=x.dtype)

    x = x * std + mean
    x = x.clamp(0.0, 1.0)

    if squeeze_back:
        x = x[0]  # devolvemos [3,H,W]

    return x

def save_triplet_grid(
    x_c: torch.Tensor,    # [B,3,H,W] content
    x_s: torch.Tensor,    # [B,3,H,W] style
    y: torch.Tensor,      # [B,3,H,W] mixed (output)
    out_path: str):
    """
    Toma SIEMPRE el primer elemento del batch (índice 0) y guarda una
    grilla 3x1 (vertical): content, style, mixed.
    """

    c0 = denorm_imagenet(x_c[0])
    s0 = denorm_imagenet(x_s[0])
    y0 = denorm_imagenet(y[0])

    grid = vutils.make_grid(
        torch.stack([c0, s0, y0], dim=0),
        nrow=1,
        padding=2)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"└─ [SAMPLE] grid guardada en {out_path}")

def _rule(n: int = 80) -> str:
    return "-" * n

def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

def train_stya2k(
    model,
    criterion,
    optimizer,
    device: str = "cuda",
    epochs: int = 10,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    grad_clip: float | None = None,
    log_every: int = 50,
    run_name: str = "StyA2K",
    # sampling
    sample_every: int = 1,
    sample_dir: str = "samples_stya2k",
    # checkpoints 
    save_every: int = 0,                     # 0 = no guardar; N = guardar cada N épocas
    ckpt_dir: str = "checkpoints_stya2k",    # carpeta de checkpoints
    # datos
    content_loader = None,
    style_loader   = None,
    train_iter     = None,
    # reanudación 
    start_epoch: int = 0,              # epoca inicial 
    init_global_step: int = 0,         # paso global acumulado previo
    scaler_state_dict: dict | None = None,   # estado previo del GradScaler (si usaste fp16)
    return_state: bool = True          # si True, retorna dict para reanudar luego
):
    """
    Entrenamiento para StyA2K con soporte de reanudación y guardado de checkpoints.

    - Puedes correr primero 20 épocas y luego volver a llamar con:
        start_epoch=20, init_global_step=estado['global_step'],
        scaler_state_dict=estado['scaler_state_dict']
    - save_every > 0: guarda un checkpoint cada 'save_every' épocas y
      también en la última época.
    """

    if content_loader is None and style_loader is None and train_iter is None:
        raise ValueError("Debes pasar content_loader y style_loader, o un train_iter ya construido.")

    os.makedirs(sample_dir, exist_ok=True)
    if save_every > 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    scaler = None
    if amp_enabled and amp_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)
        if scaler_state_dict is not None:
            try:
                scaler.load_state_dict(scaler_state_dict)
            except Exception as e:
                print(f"[WARN] No pude cargar scaler_state_dict: {e}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(_rule())
    print(f"Run: {run_name}")
    print(f"Device: {device} | AMP: {amp_enabled} ({amp_dtype}) | epochs: {epochs} | start_epoch={start_epoch}")
    print(_rule())
    print(f"{'ep':>3} | {'step':>10} | {'loss':>10} | {'content':>10} | {'style':>10} | "
          f"{'imgs':>8} | {'imgs/s':>7} | {'time':>8}")
    print(_rule())

    #  estado acumulado 
    global_step = int(init_global_step) if init_global_step else 0
    total_time = 0.0
    B = (getattr(content_loader, "batch_size", None)
         or getattr(style_loader, "batch_size", None) or 1)

    #  construir/usar iterador 
    if train_iter is None:
        if content_loader is None or style_loader is None:
            raise ValueError("Para construir train_iter se requiere content_loader y style_loader.")
        train_iter = make_train_iterator(content_loader, style_loader)

    if content_loader is None:
        raise ValueError("Se requiere content_loader para definir steps_per_epoch.")
    steps_per_epoch = len(content_loader)

    # bucle de épocas
    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()

        metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_iter=train_iter,          # reutilizamos el mismo iterador
            steps_per_epoch=steps_per_epoch,
            device=device,
            log_every=log_every,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            grad_clip=grad_clip)

        sec = time.time() - t0
        total_time += sec

        ep_steps   = metrics["steps"]
        global_step += ep_steps
        ep_loss    = metrics.get("loss", float("nan"))
        ep_content = metrics.get("content", float("nan"))
        ep_style   = metrics.get("style", float("nan"))
        n_images   = ep_steps * B
        ips  = (n_images / sec) if sec > 0 else 0.0

        print(f"{epoch:3d} | {global_step:10d} | {ep_loss:10.5f} | "
              f"{ep_content:10.5f} | {ep_style:10.5f} | "
              f"{n_images:8d} | {ips:7.1f} | {_fmt_hms(sec):>8}")

        # -------------------- SAMPLING --------------------
        need_sample = ((sample_every > 0 and ((epoch - start_epoch) % sample_every == 0))
                       or (epoch == start_epoch + epochs - 1))
        if need_sample:
            model.eval()
            with torch.no_grad():
                if content_loader is None or style_loader is None:
                    x_c, x_s = next(iter(train_iter))
                else:
                    sample_iter = make_train_iterator(content_loader, style_loader)
                    x_c, x_s = next(iter(sample_iter))
                x_c = x_c.to(device, non_blocking=True)
                x_s = x_s.to(device, non_blocking=True)

                with autocast_ctx(device=device, enabled=amp_enabled, dtype=amp_dtype):
                    y = model(x_c, x_s)

            out_path = os.path.join(sample_dir, f"{run_name}_e{epoch:03d}.png")
            save_triplet_grid(x_c, x_s, y, out_path)
            model.train()

        # ----------------- CHECKPOINTING ------------------
        is_last_epoch = (epoch == start_epoch + epochs - 1)
        need_ckpt = (save_every > 0) and (
            ((epoch - start_epoch + 1) % save_every == 0) or is_last_epoch
        )

        if need_ckpt:
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_e{epoch:03d}.pt")
            extra = {
                "epoch_loss": float(ep_loss),
                "epoch_content": float(ep_content),
                "epoch_style": float(ep_style),
                "images_seen": int(global_step * B),
                "total_time_sec": float(total_time),}
            
            save_checkpoint(path=ckpt_path,
                model=model,optimizer=optimizer,
                scaler=scaler, epoch=epoch,
                global_step=global_step,
                extra=extra,)

    print(_rule())
    print(f"Entrenamiento finalizado en {_fmt_hms(total_time)}")
    print(_rule())

    if return_state:
        return {
            "last_epoch": epoch,
            "global_step": global_step,
            "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),}
