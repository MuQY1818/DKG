import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def visualize_training_history(history_file, output_file):
    """
    Reads a training history JSON file and generates plots for
    loss and performance metrics.
    """
    if not os.path.exists(history_file):
        print(f"Error: History file not found at '{history_file}'")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 6))

    # --- Plot 1: Training Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Validation Metrics ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_auc'], 'go-', label='Validation AUC')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Validation Metrics vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    print(f"Visualization saved to '{output_file}'")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training history from a JSON file.')
    parser.add_argument(
        '--history_file', 
        type=str, 
        default='models/training_history_seed42.json',
        help='Path to the training history JSON file.'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='training_visualization.png',
        help='Path to save the output visualization image.'
    )
    args = parser.parse_args()
    
    visualize_training_history(args.history_file, args.output_file) 