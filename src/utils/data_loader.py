import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def get_occupancy_dataloaders(csv_path, batch_size=64, test_size=0.2):
    """
    Carga, preprocesa y empaqueta el dataset de ocupación para PyTorch
    """
    df = pd.read_csv(csv_path)

    df = df.dropna()

    X_df = df.drop(columns=['Date','Time','Room_Occupancy_Count'])
    y_df = df['Room_Occupancy_Count']

    X = X_df.values
    y = y_df.values

    # Shuffle = False para lograr mantener la secuencia temporal de los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Datos procesados con éxito. 16 caracteristicas extraídas")
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")

    return train_loader, test_loader, scaler

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    csv_file = project_root / "data" / "Occupancy_Estimation.csv"
    train_dl, test_dl, sc = get_occupancy_dataloaders(csv_file)

    #Ver un batch de ejemplo
    x_batch, y_batch = next(iter(train_dl))
    print(f"\nForma del batch de entrada: {x_batch.shape} (batch_size, num_features)")
    print(f"Forma del batch Y: {y_batch.shape} (batch_size)")