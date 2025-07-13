"""
ORCDF Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RGCLayer(nn.Module):
    """
    Response-aware Graph Convolutional Layer (RGC)
    
    This layer performs graph convolution separately on the correct and incorrect
    response subgraphs, as described in the ORCDF paper.
    """
    def __init__(self):
        super(RGCLayer, self).__init__()
        # The RGC layer itself does not have learnable parameters in its simplest form.
        # Parameters for transformations will be in the main model.

    def forward(self, student_embeds, problem_embeds, a_matrix, ia_matrix):
        """
        Forward pass for the RGC layer.

        Args:
            student_embeds (torch.Tensor): Embeddings for students (num_students, embed_dim)
            problem_embeds (torch.Tensor): Embeddings for problems (num_problems, embed_dim)
            a_matrix (torch.sparse.FloatTensor): Adjacency matrix for correct responses (num_students, num_problems)
            ia_matrix (torch.sparse.FloatTensor): Adjacency matrix for incorrect responses (num_students, num_problems)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated student embeddings (num_students, embed_dim)
                - Updated problem embeddings (num_problems, embed_dim)
        """
        # --- Correct Response Channel ---
        # Aggregate problem embeddings for each student based on correct answers
        # Formula: E_S_c = A * E_P
        student_embeds_c = torch.sparse.mm(a_matrix, problem_embeds)
        
        # Aggregate student embeddings for each problem based on correct answers
        # Formula: E_P_c = A^T * E_S
        problem_embeds_c = torch.sparse.mm(a_matrix.t(), student_embeds)

        # --- Incorrect Response Channel ---
        # Aggregate problem embeddings for each student based on incorrect answers
        # Formula: E_S_ic = IA * E_P
        student_embeds_ic = torch.sparse.mm(ia_matrix, problem_embeds)

        # Aggregate student embeddings for each problem based on incorrect answers
        # Formula: E_P_ic = IA^T * E_S
        problem_embeds_ic = torch.sparse.mm(ia_matrix.t(), student_embeds)

        # --- Aggregation ---
        # The paper suggests aggregating these, for example, by summing them up.
        # This is a simplified aggregation. The full ORCDF model will handle the
        # combination of these intermediate embeddings.
        
        # For now, we return the separate channels to be combined in the main model.
        # This provides more flexibility.
        
        return student_embeds_c, student_embeds_ic, problem_embeds_c, problem_embeds_ic


class ORCDF(nn.Module):
    """
    Oversmoothing-Resistant Cognitive Diagnosis Framework (ORCDF)
    """
    def __init__(self, num_students, num_problems, num_skills, embed_dim, num_layers):
        super(ORCDF, self).__init__()
        self.num_students = num_students
        self.num_problems = num_problems
        self.num_skills = num_skills
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embedding layers
        self.student_embeds = nn.Embedding(num_students, embed_dim)
        self.problem_embeds = nn.Embedding(num_problems, embed_dim)
        self.skill_embeds = nn.Embedding(num_skills, embed_dim)

        # RGC Layers
        self.rgc_layers = nn.ModuleList([RGCLayer() for _ in range(num_layers)])
        
        # Prediction layer
        self.predict = nn.Linear(embed_dim * 2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, student_ids, problem_ids, a_matrix, ia_matrix, q_matrix, return_embeds=False):
        """
        Main forward pass for the ORCDF model.

        Args:
            student_ids (torch.Tensor): Batch of student IDs.
            problem_ids (torch.Tensor): Batch of problem IDs.
            a_matrix (torch.sparse.FloatTensor): Correct response matrix.
            ia_matrix (torch.sparse.FloatTensor): Incorrect response matrix.
            q_matrix (torch.sparse.FloatTensor): Problem-skill matrix.

        Returns:
            torch.Tensor: Predicted probability of correctness.
        """
        # --- RGC Propagation ---
        all_student_embeds = [self.student_embeds.weight]
        all_problem_embeds = [self.problem_embeds.weight]

        current_s_embeds = self.student_embeds.weight
        current_p_embeds = self.problem_embeds.weight

        for i in range(self.num_layers):
            # Propagate through RGC layer
            s_c, s_ic, p_c, p_ic = self.rgc_layers[i](current_s_embeds, current_p_embeds, a_matrix, ia_matrix)
            
            # Aggregate student and problem embeddings from both channels
            current_s_embeds = s_c + s_ic
            current_p_embeds = p_c + p_ic
            
            all_student_embeds.append(current_s_embeds)
            all_problem_embeds.append(current_p_embeds)
        
        # Average pooling over all layers (including initial embeddings)
        final_student_embeds = torch.mean(torch.stack(all_student_embeds, dim=0), dim=0)
        final_problem_embeds = torch.mean(torch.stack(all_problem_embeds, dim=0), dim=0)

        # --- Skill Propagation ---
        # Propagate skill embeddings to problems
        # Formula: E_P_s = Q * E_K
        problem_skill_embeds = torch.sparse.mm(q_matrix, self.skill_embeds.weight)
        
        # Add skill information to problem embeddings
        final_problem_embeds = final_problem_embeds + problem_skill_embeds

        # --- Prediction ---
        # Get embeddings for the current batch
        batch_student_embeds = final_student_embeds[student_ids]
        batch_problem_embeds = final_problem_embeds[problem_ids]

        # Concatenate student and problem embeddings for prediction
        interaction_embeds = torch.cat([batch_student_embeds, batch_problem_embeds], dim=1)

        # Predict the outcome
        prediction = self.predict(interaction_embeds)
        
        if return_embeds:
            return torch.sigmoid(prediction).squeeze(), final_student_embeds, final_problem_embeds
        else:
            return torch.sigmoid(prediction).squeeze()

    def get_regularization(self, student_embeds_1, student_embeds_2, problem_embeds_1, problem_embeds_2):
        """
        Calculates the consistency regularization loss.
        """
        l2_loss = nn.MSELoss(reduction='mean')
        reg_students = l2_loss(student_embeds_1, student_embeds_2)
        reg_problems = l2_loss(problem_embeds_1, problem_embeds_2)
        return (reg_students + reg_problems) / 2 