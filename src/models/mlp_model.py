import torch
import torch.nn as nn


class OccupancyMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim1=64, hidden_dim2=32, output_dim=4):
        """
        Inicializamos capas y pesos. Se definen los bloques de construcción del modelo
        En este caso, se utilizó las 16 capas de entrada, dos capas ocultas con 64 y 32 
        neuronas respectivamente, y una capa de salida con 4 neuronas (para las 4 clases de ocupación: 0, 1, 2, 3)
        """
        super(OccupancyMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()

        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        Definimos como viajan los datos a través de la red (Paso hacia delante)
        """
        # Primero pasan por la capa 1 y su activación
        x = self.fc1(x)
        x = self.relu1(x)

        # Luego por la capa 2 y su activación
        x = self.fc2(x)
        x = self.relu2(x)

        # Finalmente por la capa de salida
        x = self.output_layer(x)

        return x

#Para prueba
if __name__ == "__main__":
    model = OccupancyMLP()
    print("[INFO] Arquitectura del MLP:")
    print(model)

    # Prueba con un batch de ejemplo con 64 ejemplos y 16 características cada uno
    dummy_input = torch.randn(64, 16)
    
    output = model(dummy_input)

    print(f"[INFO] Forma de entrada: {dummy_input.shape}")
    print(f"[INFO] Forma de salida: {output.shape} -> (batch_size, num_clases)")

    # Conteo de parametros a aprender
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parámetros entrenables en este MLP: {num_params}")