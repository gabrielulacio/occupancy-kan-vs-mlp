import torch
import torch.nn as nn
from kan import KAN

class OccupancyKAN(nn.Module):
    def __init__(self, layers_hidden=[8], grid_size=5, spline_order=3):
        """
        En este caso, se define mediante:
        - layers_hidden: Lista con el número de neuroa en las capas ocultas
        - grid_size (G): Que tan "fina" es la cuadrícula para las funciones spline.
            (si G es mayor más flexibilidad pero más riesgo de sobreajuste)
        - spline_order (k): El grado de polinomio dela curva (3->cubico, estandar)
        """
        super(OccupancyKAN, self).__init__()

        # La arquitectura (width) planeada son de las capas de entrada, las capas KAN ocultas, y la capa de salida
        width = [16] + layers_hidden + [4]

        self.kan = KAN(width=width, grid=grid_size, k=spline_order)

    def forward(self, x):
        """
        El flujo de datos lo maneja pykan internamente
        """
        return self.kan(x)

# Para prueba
if __name__ == "__main__":
    model = OccupancyKAN(layers_hidden=[8])
    print("[INFO] Arquitectura de la KAN inicializada")

    dummy_input = torch.randn(64, 16)
    output = model(dummy_input)

    print(f"[INFO] Forma de entrada: {dummy_input.shape}")
    print(f"[INFO] Forma de salida: {output.shape} -> (batch_size, num_clases)")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parámetros entrenables en esta KAN: {num_params}")