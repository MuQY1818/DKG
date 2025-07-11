import sys
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.nn import BCEWithLogitsLoss
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, GraphConv

# Ensure the dkg_mvp module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dkg_mvp.dkg_builder import DKGBuilder
from dkg_mvp.data_loader import DataLoader

# --- Step 1: Define GNN Model and Decoder ---

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.view(-1)

# --- Step 2: Training and Testing Functions ---

def train(model, decoder, data, optimizer, loss_fn):
    model.train()
    decoder.train()
    optimizer.zero_grad()
    
    z_dict = model(data.x_dict, data.edge_index_dict)
    
    edge_label_index = data['student', 'solve', 'problem'].edge_label_index
    edge_label = data['student', 'solve', 'problem'].edge_label
    # Use the training mask to select edges for training
    train_mask = data['student', 'solve', 'problem'].train_mask
    
    src_z = z_dict['student'][edge_label_index[0][train_mask]]
    dst_z = z_dict['problem'][edge_label_index[1][train_mask]]
    
    pred = decoder(src_z, dst_z)
    loss = loss_fn(pred, edge_label[train_mask])
    
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, decoder, data, mask):
    model.eval()
    decoder.eval()
    
    z_dict = model(data.x_dict, data.edge_index_dict)
    
    edge_label_index = data['student', 'solve', 'problem'].edge_label_index
    edge_label = data['student', 'solve', 'problem'].edge_label
    
    # Apply the given mask to select edges for evaluation
    src_z = z_dict['student'][edge_label_index[0][mask]]
    dst_z = z_dict['problem'][edge_label_index[1][mask]]
    
    pred = decoder(src_z, dst_z)
    
    ground_truth = edge_label[mask]
    pred_class = (pred > 0).float()
    
    # Handle cases where a class is not present in the mask (e.g., small val/test sets)
    if len(torch.unique(ground_truth)) < 2:
        auc = 0.5  # Cannot compute AUC if only one class is present
    else:
        auc = roc_auc_score(ground_truth.cpu(), pred.cpu())

    acc = accuracy_score(ground_truth.cpu(), pred_class.cpu())
    f1 = f1_score(ground_truth.cpu(), pred_class.cpu(), zero_division=0)
    
    return auc, acc, f1

# --- Step 3: Data Loading and Preparation ---

def convert_to_pyg(dkg_graph, interactions):
    """Converts the NetworkX DKG to a PyG HeteroData object."""
    
    # 1. Build node maps from the original IDs to dense indices
    student_ids = sorted(interactions['student_id'].unique())
    problem_ids = sorted(interactions['problem_id'].unique())
    skill_ids = sorted([
        data['skill_id'] for _, data in dkg_graph.nodes(data=True) 
        if data.get('type') == 'skill' and 'skill_id' in data
    ])

    student_map = {nid: i for i, nid in enumerate(student_ids)}
    problem_map = {nid: i for i, nid in enumerate(problem_ids)}
    skill_map = {nid: i for i, nid in enumerate(skill_ids)}

    # 2. Create edge indices for 'solve' relations using the pre-computed indices
    solve_src = interactions['student_id_idx'].values
    solve_dst = interactions['problem_id_idx'].values
    solve_edge_index = torch.tensor([solve_src, solve_dst], dtype=torch.long)
    
    # 3. The labels correspond directly to the interactions DataFrame
    edge_label = torch.FloatTensor(interactions['correct'].values)

    # 4. Create edge indices for 'require' relations
    require_edges = []
    for u, v, data in dkg_graph.edges(data=True):
        if (data.get('type') == 'require' and 
            dkg_graph.nodes[u].get('type') == 'problem' and 
            dkg_graph.nodes[v].get('type') == 'skill'):
            
            p_id = dkg_graph.nodes[u].get('problem_id')
            s_id = dkg_graph.nodes[v].get('skill_id')
            
            if p_id in problem_map and s_id in skill_map:
                require_edges.append((problem_map[p_id], skill_map[s_id]))

    require_edge_index = torch.tensor(list(zip(*require_edges)), dtype=torch.long) if require_edges else torch.empty((2, 0), dtype=torch.long)

    pyg_data = HeteroData()

    # 5. Initialize node features based on the map sizes
    embedding_dim = 64
    pyg_data['student'].x = torch.randn(len(student_ids), embedding_dim)
    pyg_data['problem'].x = torch.randn(len(problem_ids), embedding_dim)
    pyg_data['skill'].x = torch.randn(len(skill_ids), embedding_dim)

    # 6. Add edges
    pyg_data['student', 'solve', 'problem'].edge_index = solve_edge_index
    pyg_data['problem', 'require', 'skill'].edge_index = require_edge_index

    # 7. Add labels for link prediction
    pyg_data['student', 'solve', 'problem'].edge_label = edge_label
    pyg_data['student', 'solve', 'problem'].edge_label_index = solve_edge_index

    # 8. Add reverse edges
    pyg_data = T.ToUndirected()(pyg_data)

    print("PyG data object created:", pyg_data)
    return pyg_data


def build_dkg_and_pyg_data():
    """Builds the DKG and converts it to PyG format."""
    print("Loading Filtered Skill Builder dataset...")
    loader = DataLoader('dataset')
    # 使用新的数据加载方法
    data_dict = loader.load_skill_builder_data()
    
    if data_dict is None:
        print("Failed to load data. Exiting.")
        sys.exit(1) # Or handle the error appropriately
    
    # 不再需要对交互数据进行子集筛选
    # data_dict['interactions'] = data_dict['interactions'].head(20000)
    
    dkg_builder = DKGBuilder()
    dkg_graph = dkg_builder.build_from_data(data_dict)
    
    pyg_data = convert_to_pyg(dkg_graph, data_dict['interactions'])
    
    num_students = dkg_graph.graph.get('num_students', 0)
    num_problems = dkg_graph.graph.get('num_problems', 0)
    num_skills = dkg_graph.graph.get('num_skills', 0)
    
    print(f"  - {num_students} students")
    print(f"  - {num_problems} problems")
    print(f"  - {num_skills} skills")
    
    print("\nStep 2: Building DKG from historical data and converting to PyG format...")
    
    return pyg_data, dkg_graph, data_dict

# --- Step 4: Main Training Loop ---

def save_embeddings(model, data, student_map, problem_map, skill_map):
    """Saves the node embeddings from the trained model."""
    print("\n--- Saving Node Embeddings ---")
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Generate embeddings for all nodes
        z_dict = model(data.x_dict, data.edge_index_dict)
        
    # Create a directory to save embeddings
    output_dir = os.path.join('models', 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Save Student Embeddings ---
    student_embeddings = z_dict['student'].cpu().numpy()
    student_ids = list(student_map.keys())
    student_df = pd.DataFrame(student_embeddings, index=student_ids)
    student_df.index.name = 'student_id'
    student_path = os.path.join(output_dir, 'student_embeddings.csv')
    student_df.to_csv(student_path)
    print(f"Student embeddings saved to {student_path}")

    # --- Save Problem Embeddings ---
    problem_embeddings = z_dict['problem'].cpu().numpy()
    problem_ids = list(problem_map.keys())
    problem_df = pd.DataFrame(problem_embeddings, index=problem_ids)
    problem_df.index.name = 'problem_id'
    problem_path = os.path.join(output_dir, 'problem_embeddings.csv')
    problem_df.to_csv(problem_path)
    print(f"Problem embeddings saved to {problem_path}")

    # --- Save Skill Embeddings ---
    skill_embeddings = z_dict['skill'].cpu().numpy()
    skill_ids = list(skill_map.keys())
    skill_df = pd.DataFrame(skill_embeddings, index=skill_ids)
    skill_df.index.name = 'skill_id'
    skill_path = os.path.join(output_dir, 'skill_embeddings.csv')
    skill_df.to_csv(skill_path)
    print(f"Skill embeddings saved to {skill_path}")

def main():
    data, dkg_graph, data_dict = build_dkg_and_pyg_data()
    
    # --- Create masks for splitting supervision edges ---
    edge_label_index = data['student', 'solve', 'problem'].edge_label_index
    num_edges = edge_label_index.size(1)
    
    if num_edges == 0:
        print("No 'solve' edges found to split. Exiting.")
        return

    indices = torch.randperm(num_edges)

    train_size = int(num_edges * 0.8)
    val_size = int(num_edges * 0.1)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data['student', 'solve', 'problem'].train_mask = train_mask
    data['student', 'solve', 'problem'].val_mask = val_mask
    data['student', 'solve', 'problem'].test_mask = test_mask

    print(f"\nSupervision edges split:")
    print(f"  - Training: {train_mask.sum()} edges")
    print(f"  - Validation: {val_mask.sum()} edges")
    print(f"  - Test: {test_mask.sum()} edges")
    
    print("\nStep 3: Initializing and training the GNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    embedding_dim = 64
    hidden_channels = 128

    encoder = GNNEncoder(hidden_channels, embedding_dim)
    decoder = LinkPredictor(embedding_dim, hidden_channels, 1)

    model = to_hetero(encoder, data.metadata(), aggr='sum').to(device)
    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), 
        lr=0.001
    )
    loss_fn = BCEWithLogitsLoss()
    
    data = data.to(device)
    
    # Extract masks after moving data to device
    train_mask = data['student', 'solve', 'problem'].train_mask
    val_mask = data['student', 'solve', 'problem'].val_mask
    test_mask = data['student', 'solve', 'problem'].test_mask

    print("\nEpoch | Loss     | Train AUC | Val AUC  | Train F1 | Val F1")
    print("-----------------------------------------------------------------")
    for epoch in range(1, 101):
        loss = train(model, decoder, data, optimizer, loss_fn)
        train_auc, train_acc, train_f1 = test(model, decoder, data, train_mask)
        val_auc, val_acc, val_f1 = test(model, decoder, data, val_mask)
        print(f"{epoch:5d} | {loss:.4f} | {train_auc:9.4f} | {val_auc:8.4f} | {train_f1:8.4f} | {val_f1:6.4f}")

    print("\n--- Final Performance on Test Set ---")
    test_auc, test_acc, test_f1 = test(model, decoder, data, test_mask)
    print(f"Test AUC: {test_auc:.4f}, Test Accuracy: {test_acc:.4f}, Test F1-Score: {test_f1:.4f}")

    # Get the maps for saving
    student_ids = sorted(data_dict['interactions']['student_id'].unique())
    problem_ids = sorted(data_dict['interactions']['problem_id'].unique())
    skill_ids = sorted([
        node_data['skill_id'] for _, node_data in dkg_graph.nodes(data=True)
        if node_data.get('type') == 'skill' and 'skill_id' in node_data
    ])
    
    student_map = {nid: i for i, nid in enumerate(student_ids)}
    problem_map = {nid: i for i, nid in enumerate(problem_ids)}
    skill_map = {nid: i for i, nid in enumerate(skill_ids)}

    save_embeddings(model, data, student_map, problem_map, skill_map)

if __name__ == '__main__':
    main() 