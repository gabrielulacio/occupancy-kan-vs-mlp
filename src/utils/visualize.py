import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_metrics(history_mlp, history_kan, save_dir=None):
    project_root = Path(__file__).resolve().parents[2]
    output_dir = Path(save_dir) if save_dir is not None else project_root / "results"

    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_mlp["train_loss"]) + 1)

    # Grafica: Comparación de Pérdida (Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_mlp["train_loss"], label="MLP Loss", marker='o')
    plt.plot(epochs, history_kan["train_loss"], label="KAN Loss", marker='s')
    plt.title("Curva de Aprendizaje (Loss) - MLP vs KAN")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_comparison.png")
    plt.close()

    # Grafica: Comparación de F1-Score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_mlp["test_f1"], label="MLP F1-Score", color='green', marker='o')
    plt.plot(epochs, history_kan["test_f1"], label="KAN F1-Score", color='blue', marker='s')
    plt.title("Rendimiento de Clasificación (F1-Score) - MLP vs KAN")
    plt.xlabel("Épocas")
    plt.ylabel("Macro F1-Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png")
    plt.close()

    # Gráfica: Comparación de tiempo (En barras)
    plt.figure(figsize=(8, 5))
    modelos = ['MLP Clásico', 'Red KAN']
    tiempos = [history_mlp["total_time"], history_kan["total_time"]]
    colores = ['green', 'blue']
    
    barras = plt.bar(modelos, tiempos, color=colores, width=0.5)
    plt.title("Tiempo total de entrenamiento (15 Épocas)")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, yval + 0.5, f'{yval:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png")
    plt.close()

    print(f"[INFO] Gráficas guardadas exitosamente en: {output_dir}")