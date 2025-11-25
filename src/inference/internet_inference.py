import requests
from io import BytesIO
from typing import Iterable

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.utils as vutils

from src.data.load_data import IMAGENET_MEAN, IMAGENET_STD
from src.training.train_model import denorm_imagenet


def build_inference_transform(size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_from_source(source: str) -> Image.Image:
    """
    Carga imagen desde URL o path local y la devuelve en RGB.
    """
    try:
        if source.startswith("http"):
            response = requests.get(source)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(source)

        return img.convert("RGB")
    except Exception as e:
        print(f"Error cargando imagen: {source}")
        raise e


def prepare_tensor_from_source(source: str, tfm: T.Compose, device: torch.device) -> torch.Tensor:
    img = load_image_from_source(source)
    return tfm(img).unsqueeze(0).to(device)


def fuse_styles(style_tensors: list[torch.Tensor], weights: Iterable[float] | None = None) -> torch.Tensor:
    """
    Combina múltiples tensores de estilo en un solo mapa ponderado.
    style_tensors: lista de tensores [1,3,H,W] ya normalizados.
    weights: iterable de la misma longitud; si None, usa promedio uniforme.
    """
    if len(style_tensors) == 1:
        return style_tensors[0]

    stack = torch.cat(style_tensors, dim=0)  # [S,3,H,W]
    if weights is None:
        weights_t = torch.ones(stack.size(0), device=stack.device, dtype=stack.dtype) / stack.size(0)
    else:
        weights_list = list(weights)
        if len(weights_list) != stack.size(0):
            raise ValueError("style weights length must match number of style tensors")
        weights_t = torch.tensor(weights_list, device=stack.device, dtype=stack.dtype)
        weights_t = weights_t / weights_t.sum().clamp(min=1e-6)

    fused = (weights_t.view(-1, 1, 1, 1) * stack).sum(dim=0, keepdim=True)
    return fused


def run_style_transfer_inference(
    model,
    content_source: str,
    style_source,
    output_path: str = "resultado.jpg",
    device: str = "cuda",
    size: int = 256,
    style_weights: list[float] | None = None,
    alpha: float = 1.0):
    """
    Recibe links o paths (content + uno o más styles), procesa, corre el modelo y guarda el grid.
    """
    print(f"--- Procesando: {output_path} ---")

    tfm = build_inference_transform(size)

    x_c = prepare_tensor_from_source(content_source, tfm, torch.device(device))
    styles = style_source if isinstance(style_source, (list, tuple)) else [style_source]
    style_tensors = [prepare_tensor_from_source(src, tfm, torch.device(device)) for src in styles]
    x_s = fuse_styles(style_tensors, weights=style_weights)

    # Inferencia (Modelo en Eval)
    model.eval()
    with torch.no_grad():
        y_logits = model(x_c, x_s, alpha=alpha)

    # Post-procesamiento
    y_prob = torch.sigmoid(y_logits)

    # Visualización (Denormalizar Inputs + Juntar)
    c_vis = denorm_imagenet(x_c[0]).cpu()
    s_vis = denorm_imagenet(x_s[0]).cpu()
    y_vis = y_prob[0].cpu().clamp(0.0, 1.0)

    # Crear Grid 3x1
    grid = vutils.make_grid(
        torch.stack([c_vis, s_vis, y_vis], dim=0),
        nrow=3,
        padding=5,
        pad_value=1.0)

    vutils.save_image(grid, output_path)
    print(f"Listo!! Guardado en: {output_path}")

    return output_path
