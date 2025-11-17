import torch
import matplotlib.pyplot as plt

def analyze_loader(name, loader):
    print(f"\n===== Análisis del loader: {name} =====")

    total_imgs = len(loader.dataset)
    print(f"Total de imágenes: {total_imgs}")

    num_batches = len(loader)
    print(f"Número de batches: {num_batches}")

    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    print(f"dtype: {batch.dtype}")

    print(f"min pixel: {batch.min().item():.4f}")
    print(f"max pixel: {batch.max().item():.4f}")
    print(f"mean: {batch.mean().item():.4f}")
    print(f"std:  {batch.std().item():.4f}")

    print("NaNs:", torch.isnan(batch).any().item())
    print("Infs:", torch.isinf(batch).any().item())

    print("mean por canal:", batch.mean(dim=(0,2,3)))
    print("std por canal:", batch.std(dim=(0,2,3)))



IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm(x):
    """x: tensor [3,H,W] normalizado → tensor [H,W,3] en [0,1]"""
    x = x.cpu() * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0,1)
    return x.permute(1,2,0).numpy()

def show_examples(train_iter, num_pairs=5):
    """
    Muestra num_pairs de pares (content, style) obtenidos del DualBatchIterator.
    """
    x_content, x_style = next(iter(train_iter))
    num_pairs = min(num_pairs, x_content.size(0), x_style.size(0))

    plt.figure(figsize=(10, 4*num_pairs))

    for i in range(num_pairs):

        plt.subplot(num_pairs, 2, 2*i + 1)
        plt.imshow(denorm(x_content[i]))
        plt.title(f"Content #{i}")
        plt.axis("off")

        plt.subplot(num_pairs, 2, 2*i + 2)
        plt.imshow(denorm(x_style[i]))
        plt.title(f"Style #{i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()