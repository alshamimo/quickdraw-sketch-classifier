"""
Main training script - trains NN and CNN models on QuickDraw data.

Orchestrates the complete training pipeline:
1. Loads and preprocesses QuickDraw drawing data
2. Trains both fully-connected NN and convolutional CNN models
3. Evaluates final performance on test set
4. Saves trained models, results, and visualizations

Run this script directly to execute the full training workflow.
"""
import config
import os
import torch
import json
from src.preprocessing import Preprocessor
from src.models import QuickDrawCNN, QuickDrawNN
from src.train import Train
from src.evaluate import evaluate
from src.visualize import generate_all_plots

def save_results(history, final_acc, model_name, classes):
    """
    Save training metrics to JSON file.

    Stores final validation accuracy and full training history
    for later analysis and comparison between models.

    Args:
        history: Training history dict with loss/accuracy per epoch
        final_acc: Final validation accuracy score
        model_name: Identifier for the model ('nn' or 'cnn')
        classes: List of class names used in training
    """
    os.makedirs("results", exist_ok=True)
    result = {
        "model": model_name,
        "final_val_acc": final_acc,
        "classes": classes,
        "history": history
    }
    path = f"results/{model_name}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Results successfully saved to {path}")

def main():
    """
    Execute full training pipeline for both models.

    Creates data loaders, trains NN and CNN models, evaluates performance,
    saves model weights and results, generates comparison plots.
    """
    preprocessor = Preprocessor(config.CLASSES, config.MAX_SAMPLES)
    train_loader, test_loader = preprocessor.get_loaders(
        config.TEST_SPLIT,
        config.RANDOM_SEED,
        config.BATCH_SIZE
    )

    train_params = {
        "data_loader": train_loader,
        "test_loader": test_loader,
        "epochs":      config.EPOCHS,
        "lr":          config.LR,
        "device":      config.DEVICE   
    }

    nn_model  = QuickDrawNN()
    cnn_model = QuickDrawCNN()

    nn_history  = Train(nn_model,  **train_params).train(eval_func=evaluate)
    cnn_history = Train(cnn_model, **train_params).train(eval_func=evaluate)

    nn_acc  = nn_history['val_acc'][-1]
    cnn_acc = cnn_history['val_acc'][-1]

    os.makedirs("train_results", exist_ok=True)
    torch.save(nn_model.state_dict(),  "train_results/nn_model.pth")
    torch.save(cnn_model.state_dict(), "train_results/cnn_model.pth")
    print("Models saved in train_results/")

    generate_all_plots(
        nn_model=nn_model,
        cnn_model=cnn_model,
        nn_history=nn_history,
        cnn_history=cnn_history,
        nn_acc=nn_acc,
        cnn_acc=cnn_acc,
        test_loader=test_loader,
        device=config.DEVICE,
        classes=config.CLASSES
    )
    save_results(nn_history,  nn_acc,  "nn",  config.CLASSES)
    save_results(cnn_history, cnn_acc, "cnn", config.CLASSES)

if __name__ == "__main__":
    main()