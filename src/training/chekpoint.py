import torch
import os

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    extra: dict | None = None,
):
    """
    Guarda un checkpoint de entrenamiento completo.

    Parámetros
    ----------
    path : str
        Ruta destino (ej. 'checkpoints/stya2k_e005.pt').
    model : nn.Module
        Modelo (puede ser DataParallel).
    optimizer : Optimizer
        Optimizador.
    scaler : GradScaler o None
        Escalador de AMP si usas FP16.
    epoch, global_step : int
        Progreso actual.
    extra : dict
        Metadatos adicionales opcionales (por ej. métricas).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # si es DataParallel, acceder al modelo interno
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "extra": extra or {}}

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path)
    print(f"[CKPT] Guardado en {path}")

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    device: str = "cuda"):
    
    """
    Carga un checkpoint y restaura modelo/optimizador/escalador.

    Devuelve: (epoch, global_step, extra)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {path}")

    ckpt = torch.load(path, map_location=device)

    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model"], strict=True)
    print(f"[CKPT] Modelo cargado desde {path}")

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("[CKPT] Optimizador restaurado")

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        print("[CKPT] GradScaler restaurado")

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    extra = ckpt.get("extra", {})

    print(f"[CKPT] Epoch={epoch}, Global step={global_step}")
    return epoch, global_step, extra
