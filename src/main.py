import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, f1_score

from utils.visualize import plot_training_metrics
from utils.data_loader import get_occupancy_dataloaders
from models.mlp_model import OccupancyMLP
from models.kan_model import OccupancyKAN

def train_and_evaluate(model, train_loader, test_loader, epochs=20, lr=0.001, model_name="Modelo"):
    print(f"\n{'='*50}")
    print(f"[INFO] Iniciando entrenamiento de {model_name}...")
    print(f"{'='*50}")

    # Árbitro y entrenador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time_total = time.time()
    history = {"train_loss": [], "test_acc": [], "test_f1": []}

    for epoch in range(epochs):
        # FASE DE ESTUDIO
        model.train()
        epoch_loss = 0.0
        start_time_epoch = time.time()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad() # Limpiar memoria de entrenador
            outputs = model(X_batch) # Intento de adivinar (Forward pass)
            loss = criterion(outputs, y_batch) # Evalua error
            loss.backward() # Backpropagation
            optimizer.step() # Ajustar las neuronas (Update)

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time_epoch

        # FASE DE EVALUACIÓN
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')

        history["train_loss"].append(avg_train_loss)
        history["test_acc"].append(acc)
        history["test_f1"].append(f1)

        print(f"Epoch {epoch+1:02d}/{epochs} | Tiempo: {epoch_time:.2f}s | Loss: {avg_train_loss:.4f} | Acc: {acc:.4f} | F1-Score: {f1:.4f}")

    total_time = time.time() - start_time_total
    print(f"\n[INFO] Entrenamiento de {model_name} finalizado en {total_time:.2f} segundos.")
    
    history['total_time'] = total_time
    return history

if __name__ == "__main__":

    # 1. Preparación de los datos
    csv_path = Path(__file__).resolve().parents[1] / "data" / "Occupancy_Estimation.csv"
    train_dl, test_dl, scaler = get_occupancy_dataloaders(csv_path, batch_size=64)

    # 2. Instanciar los contendientes
    mlp_model = OccupancyMLP(input_dim=16, hidden_dim1=64, hidden_dim2=32, output_dim=4)
    kan_model = OccupancyKAN(layers_hidden=[8])

    # 3. Entrenamiento (con 15 épocas para ver resultados rápidamente)
    print("\n\n >>> PERCEPTRÓN MULTICAPA (MLP) <<<")
    history_mlp = train_and_evaluate(mlp_model, train_dl, test_dl, epochs=15, model_name="MLP Clásico")

    print("\n>>> RED KOLMOGOROV-ARNOLD (KAN) <<<")
    history_kan = train_and_evaluate(kan_model, train_dl, test_dl, epochs=15, model_name="Red KAN")

    # 4. Resumen 
    print(f"\n{'='*50}")
    print("RESUMEN DEL ENFRENTAMIENTO")
    print(f"{'='*50}")
    print(f"MLP -> F1-Score final: {history_mlp['test_f1'][-1]:.4f} | Tiempo total: {history_mlp['total_time']:.2f}s")
    print(f"KAN -> F1-Score final: {history_kan['test_f1'][-1]:.4f} | Tiempo total: {history_kan['total_time']:.2f}s")

    plot_training_metrics(history_mlp, history_kan)


