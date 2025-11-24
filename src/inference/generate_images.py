import torch 
from src.training.gradscaler import * 
from src.training.train_model import *

def save_random_horizontal_sample(
    model,
    train_iter,
    device: str,
    out_path: str,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16"):

    pair_iter = iter(train_iter)
    try:
        x_c, x_s = next(pair_iter)
    except StopIteration:
        return

    b = min(x_c.size(0), x_s.size(0))
    if b == 0:
        return 
        
    idx = torch.randint(0, b, (1,)).item()

    x_c_sel = x_c[idx : idx + 1].to(device, non_blocking=True)
    x_s_sel = x_s[idx : idx + 1].to(device, non_blocking=True)

    # forward del modelo
    model.eval()
    with torch.no_grad():
        with autocast_ctx(device=device, enabled=amp_enabled, dtype=amp_dtype):
            y = model(x_c_sel, x_s_sel) 
    model.train()

    # desnormalizar content/style (ImageNet)
    c0 = denorm_imagenet(x_c_sel[0]).cpu()
    s0 = denorm_imagenet(x_s_sel[0]).cpu()
    
    # Aplicamos Sigmoid para pasar de Logits (-inf, inf) a Imagen (0, 1)
    y_prob = torch.sigmoid(y[0]) 
    
    y0 = y_prob.detach().cpu().float().clamp(0.0, 1.0)

    grid = vutils.make_grid(
        torch.stack([c0, s0, y0], dim=0),
        nrow=3,
        padding=2)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"└─ [RANDOM SAMPLE] Horizontal grid saved to {out_path}")


def save_multiple_random_samples(
    model,
    train_iter,
    device: str,
    out_dir: str,
    n_samples: int = 5,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
):
    """
    Crea una carpeta `out_dir` y guarda `n_samples` grids horizontales
    (content | style | output) usando `save_random_horizontal_sample`.

    Parámetros
    ----------
    model : nn.Module
        Modelo StyA2K entrenado.
    train_iter : iter
        Iterador DualBatchIterator (o similar) que produce (x_c, x_s).
    device : str
        'cuda' o 'cpu'.
    out_dir : str
        Carpeta donde se guardarán las imágenes.
    n_samples : int
        Número de ejemplos a guardar.
    amp_enabled : bool
        Si se usa AMP.
    amp_dtype : str
        'bf16' o 'fp16'.
    """
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_samples):
        out_path = os.path.join(out_dir, f"sample_{i:03d}.png")
        save_random_horizontal_sample(
            model=model,
            train_iter=train_iter,
            device=device,
            out_path=out_path,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype)



