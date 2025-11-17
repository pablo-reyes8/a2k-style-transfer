import torch
import torch.nn.functional as F
from contextlib import nullcontext

def has_nan_or_inf(x: torch.Tensor) -> bool:
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()

def tstats(x: torch.Tensor, name="tensor"):
    x = x.detach()
    print(f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.min().item():.4f} max={x.max().item():.4f} "
          f"mean={x.mean().item():.4f} std={x.std().item():.4f} "
          f"nan/inf={has_nan_or_inf(x)}")

def gram_matrix_fp32(feat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    Fm = feat.float().view(B, C, H*W)              # FP32
    G  = torch.bmm(Fm, Fm.transpose(1, 2))         # [B,C,C] FP32
    G  = G / (C * H * W)
    return G

@torch.no_grad()
def debug_style_targets(loss_extractor, x_c, x_s):
    """Solo para ver targets (content/style) sin gradiente."""
    print("\n== Targets por VGG (sin gradiente) ==")
    f_c_true = loss_extractor(x_c)
    f_s_true = loss_extractor(x_s)
    for l in f_s_true.keys():
        tstats(f_s_true[l], name=f"STYLE feat[{l}]")
        Gs = gram_matrix_fp32(f_s_true[l])
        tstats(Gs, name=f"STYLE Gram[{l}]")
    return f_c_true, f_s_true

def debug_style_loss_step(
    model,
    loss_extractor,       
    x_c, x_s,               
    device="cuda",
    amp_enabled=True,      
    amp_dtype="fp16",
    clamp_pred=True):
    
    """
    Corre un paso forward de modelo y pérdida de estilo IMPRIMIENDO
    estadísticas y detectando dónde aparecen NaN/Inf.
    """

    if amp_enabled and device == "cuda":
        autocast_model_ctx = torch.amp.autocast(device_type="cuda",
                                                dtype=(torch.float16 if amp_dtype.lower() in ("fp16","float16") else torch.bfloat16))
    else:
        autocast_model_ctx = nullcontext()

    model.eval()
    with autocast_model_ctx:
        y = model(x_c, x_s)     # salida del decoder
    tstats(y, "MODEL output y (antes de clamp)")
    if has_nan_or_inf(y):
        print(">> ALERTA: y contiene NaN/Inf. Aplicando nan_to_num y clamp...")
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    if clamp_pred:
        y = y.clamp(0.0, 1.0)
        tstats(y, "MODEL output y (clamped [0,1])")

    with torch.no_grad():
        f_c_true = loss_extractor(x_c)
        f_s_true = loss_extractor(x_s)

    with torch.cuda.amp.autocast(enabled=False):
        f_pred = loss_extractor(y.float())

    print("\n== Inspección capa a capa (style) ==")
    any_nan = False
    for l in sorted(f_s_true.keys()):
        Fp = f_pred[l]
        Fs = f_s_true[l]
        tstats(Fp, name=f"PRED feat[{l}]")
        tstats(Fs, name=f"STYLE feat[{l}]")

        # Gram en FP32
        Gp = gram_matrix_fp32(Fp)
        Gs = gram_matrix_fp32(Fs)
        tstats(Gp, name=f"PRED Gram[{l}]")
        tstats(Gs, name=f"STYLE Gram[{l}]")

        if has_nan_or_inf(Fp):
            print(f">> NaN/Inf en PRED feat[{l}] — el problema aparece ANTES del Gram.")
            any_nan = True
        if has_nan_or_inf(Gp):
            print(f">> NaN/Inf en PRED Gram[{l}] — overflow ocurriendo en el producto FFᵀ.")
            any_nan = True

    print("\n== Inspección capa de contenido ==")
    for l in sorted(f_c_true.keys()):
        tstats(f_pred[l], name=f"PRED feat[{l}] (content layer)")
        tstats(f_c_true[l], name=f"CONT  feat[{l}] (target)")
        # no hay Gram aquí; solo por si quieres ver el MSE
        mse = F.mse_loss(f_pred[l], f_c_true[l]).item()
        print(f"content MSE[{l}] = {mse:.6f}")

    if not any_nan:
        print(" No se detectaron NaN/Inf en features ni en Gram. "
              "Si aún hay NaN en la pérdida, revisa los pesos de style, el LR o grad_clip.")
    print("-" * 80)
    return {"y": y, "f_pred": f_pred, "f_s_true": f_s_true, "f_c_true": f_c_true}