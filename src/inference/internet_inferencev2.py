import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils
import torch
from src.training.train_model import denorm_imagenet

inference_tf = T.Compose([
    T.Resize((256, 256)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def load_image_from_source(source: str):
    """
    Carga imagen desde URL o Path local y la convierte a RGB.
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

def run_style_transfer_inference(
    model, 
    content_source, 
    style_source, 
    output_path="resultado.jpg", 
    device="cuda"):
    """
    Recibe links o paths, procesa, corre el modelo y guarda el grid.
    """
    print(f"--- Procesando: {output_path} ---")
    
    # Cargar Imágenes
    img_c = load_image_from_source(content_source)
    img_s = load_image_from_source(style_source)
    
    # Preprocesar (Transform + Batch Dim + Device)
    x_c = inference_tf(img_c).unsqueeze(0).to(device)
    x_s = inference_tf(img_s).unsqueeze(0).to(device)
    
    # Inferencia (Modelo en Eval)
    model.eval()
    with torch.no_grad():
        y_logits = model(x_c, x_s)
        
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
        pad_value=1.0 )
    
    vutils.save_image(grid, output_path)
    print(f"Listo!! Guardado en: {output_path}")
    
    return output_path

######## EXAMPLE: 

url_content = "https://cdn.asp.events/CLIENT_Oliver_K_15A4C8AE_5056_B739_54CFDE58102DEF33/sites/sydney-build-2025/media/libraries/sydney-build-blog/Sydney%20Opera%20House%20image.png" 
url_style   = "https://www.arte-mare.eu/wp-content/uploads/2022/09/Van-Gogh-640x440.jpeg" 

run_style_transfer_inference(
    model=model,
    content_source=url_content,
    style_source=url_style,
    output_path="test_vangogh.jpg",
    device=device)