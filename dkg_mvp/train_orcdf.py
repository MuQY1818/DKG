"""
Training script for the ORCDF model.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import argparse
import os
import random

from .data_loader import DataLoader
from .orcdf.model import ORCDF

def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_sparse_tensor(matrix, device):
    """Convert a numpy matrix to a sparse torch tensor."""
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.toarray()
    
    # Find indices of non-zero elements
    coo = torch.from_numpy(matrix).to_sparse()
    
    indices = coo.indices()
    values = coo.values()
    shape = coo.size()
    
    return torch.sparse_coo_tensor(indices, values, shape, device=device).float()

def create_flipped_graphs(a_matrix, ia_matrix, flip_ratio):
    """
    Creates flipped versions of the response graphs for consistency regularization.
    
    Args:
        a_matrix (torch.sparse.FloatTensor): The original correct response matrix.
        ia_matrix (torch.sparse.FloatTensor): The original incorrect response matrix.
        flip_ratio (float): The percentage of edges to flip.

    Returns:
        Tuple[torch.sparse.FloatTensor, torch.sparse.FloatTensor]:
            - a_matrix_flipped
            - ia_matrix_flipped
    """
    a_indices = a_matrix.coalesce().indices()
    a_values = a_matrix.coalesce().values()
    ia_indices = ia_matrix.coalesce().indices()
    ia_values = ia_matrix.coalesce().values()

    # --- Flip edges from A to IA ---
    num_a_edges = a_indices.shape[1]
    a_flip_count = int(num_a_edges * flip_ratio)
    a_flip_indices = torch.randperm(num_a_edges)[:a_flip_count]
    
    a_to_ia_indices = a_indices[:, a_flip_indices]
    a_to_ia_values = torch.ones(a_flip_count, device=a_matrix.device)
    
    # Keep non-flipped edges
    a_keep_mask = torch.ones(num_a_edges, dtype=torch.bool)
    a_keep_mask[a_flip_indices] = False
    a_indices_new = a_indices[:, a_keep_mask]
    a_values_new = a_values[a_keep_mask]
    
    # --- Flip edges from IA to A ---
    num_ia_edges = ia_indices.shape[1]
    ia_flip_count = int(num_ia_edges * flip_ratio)
    ia_flip_indices = torch.randperm(num_ia_edges)[:ia_flip_count]

    ia_to_a_indices = ia_indices[:, ia_flip_indices]
    ia_to_a_values = torch.ones(ia_flip_count, device=ia_matrix.device)

    # Keep non-flipped edges
    ia_keep_mask = torch.ones(num_ia_edges, dtype=torch.bool)
    ia_keep_mask[ia_flip_indices] = False
    ia_indices_new = ia_indices[:, ia_keep_mask]
    ia_values_new = ia_values[ia_keep_mask]

    # --- Combine to create new sparse matrices ---
    final_a_indices = torch.cat([a_indices_new, ia_to_a_indices], dim=1)
    final_a_values = torch.cat([a_values_new, ia_to_a_values], dim=0)
    
    final_ia_indices = torch.cat([ia_indices_new, a_to_ia_indices], dim=1)
    final_ia_values = torch.cat([ia_values_new, a_to_ia_values], dim=0)

    shape = a_matrix.shape
    a_matrix_flipped = torch.sparse_coo_tensor(final_a_indices, final_a_values, shape)
    ia_matrix_flipped = torch.sparse_coo_tensor(final_ia_indices, final_ia_values, shape)

    return a_matrix_flipped.coalesce(), ia_matrix_flipped.coalesce()


def train_epoch(model, optimizer, loss_fn, dataloader, device, a_matrix, ia_matrix, q_matrix, args):
    model.train()
    total_loss = 0.0
    
    # Create flipped graphs for this epoch
    a_matrix_flipped, ia_matrix_flipped = create_flipped_graphs(a_matrix, ia_matrix, args.flip_ratio)

    for batch in dataloader:
        student_ids, problem_ids, labels = batch
        student_ids = student_ids.to(device)
        problem_ids = problem_ids.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        # Forward pass on original graph
        preds_original, s_embeds_orig, p_embeds_orig = model(student_ids, problem_ids, a_matrix, ia_matrix, q_matrix, return_embeds=True)
        
        # Forward pass on flipped graph
        _, s_embeds_flipped, p_embeds_flipped = model(student_ids, problem_ids, a_matrix_flipped, ia_matrix_flipped, q_matrix, return_embeds=True)

        # Calculate losses
        bce_loss = loss_fn(preds_original, labels)
        reg_loss = model.get_regularization(s_embeds_orig, s_embeds_flipped, p_embeds_orig, p_embeds_flipped)
        
        loss = bce_loss + args.reg_lambda * reg_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, device, a_matrix, ia_matrix, q_matrix):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            student_ids, problem_ids, labels = batch
            student_ids = student_ids.to(device)
            problem_ids = problem_ids.to(device)

            preds = model(student_ids, problem_ids, a_matrix, ia_matrix, q_matrix)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds > 0.5)
    
    return auc, acc


def main(args):
    """Main training function."""
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    dataset_path = os.path.join(project_root, 'dataset')
    
    loader = DataLoader(dataset_path)
    data = loader.load_orcdf_data(nrows=args.nrows)
    if not data:
        print("Failed to load data. Exiting.")
        return

    # --- 2. Prepare Data for Training ---
    interactions = data['interactions']
    train_interactions, val_interactions = train_test_split(
        interactions, test_size=0.2, random_state=args.seed
    )

    # Convert to PyTorch DataLoaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor([i[0] for i in train_interactions]),
        torch.LongTensor([i[1] for i in train_interactions]),
        torch.LongTensor([i[2] for i in train_interactions])
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor([i[0] for i in val_interactions]),
        torch.LongTensor([i[1] for i in val_interactions]),
        torch.LongTensor([i[2] for i in val_interactions])
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    a_matrix = to_sparse_tensor(data['a_matrix'], device)
    ia_matrix = to_sparse_tensor(data['ia_matrix'], device)
    q_matrix = to_sparse_tensor(data['q_matrix'], device)
    
    # --- 3. Initialize Model, Optimizer, and Loss ---
    print("Initializing model...")
    model = ORCDF(
        num_students=data['num_students'],
        num_problems=data['num_problems'],
        num_skills=data['num_skills'],
        embed_dim=args.embed_dim,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCELoss()

    print("Model initialized:")
    print(model)
    
    # --- 4. Training Loop ---
    print("\nStarting training process...")
    best_auc = 0.0
    output_dir = os.path.join(project_root, 'models')
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f'orcdf_best_model_seed{args.seed}.pt')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader, device, a_matrix, ia_matrix, q_matrix, args)
        val_auc, val_acc = evaluate_epoch(model, val_loader, device, a_matrix, ia_matrix, q_matrix)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved with AUC: {best_auc:.4f}")

    print("\nTraining finished.")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the ORCDF model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Dimension of embeddings.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RGC layers.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--flip_ratio', type=float, default=0.1, help='Ratio of edges to flip for regularization.')
    parser.add_argument('--reg_lambda', type=float, default=0.1, help='Lambda for consistency regularization loss.')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from data for debugging.')
    
    args = parser.parse_args()
    main(args) 