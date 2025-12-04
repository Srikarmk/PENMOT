# import torch
# import torch.nn as nn
#
#
# def log_sinkhorn_iterations(log_alpha, n_iters=5):
#     for _ in range(n_iters):
#         log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
#         log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
#     return log_alpha
#
#
# def sinkhorn_algorithm(cost_matrix, n_iters=5, tau=0.1, eps=1e-8):
#     batch_size, n, m = cost_matrix.shape
#
#     log_alpha = -cost_matrix / tau
#
#     log_alpha = log_sinkhorn_iterations(log_alpha, n_iters)
#
#     assignment = torch.exp(log_alpha)
#
#     return assignment
#
#
# class SinkhornMatcher(nn.Module):
#     def __init__(self, n_iters=5, tau=0.1, hard_assignment=False):
#         super().__init__()
#         self.n_iters = n_iters
#         self.tau = tau
#         self.hard_assignment = hard_assignment
#
#         print(f"[OK] SinkhornMatcher: {n_iters} iters, tau={tau}, hard={hard_assignment}")
#
#     def forward(self, cost_matrix):
#         if cost_matrix.dim() == 2:
#             cost_matrix = cost_matrix.unsqueeze(0)
#
#         assignment = sinkhorn_algorithm(cost_matrix, self.n_iters, self.tau)
#
#         if self.hard_assignment:
#             row_indices = torch.argmax(assignment, dim=2)
#             hard_assignment = torch.zeros_like(assignment)
#             batch_indices = torch.arange(assignment.size(0)).unsqueeze(1)
#             col_indices = torch.arange(assignment.size(1)).unsqueeze(0)
#             hard_assignment[batch_indices, col_indices, row_indices] = 1.0
#             assignment = hard_assignment
#
#         return assignment.squeeze(0) if assignment.size(0) == 1 else assignment
#
#
# def hungarian_algorithm(cost_matrix):
#     from scipy.optimize import linear_sum_assignment
#
#     if cost_matrix.dim() == 2:
#         cost_matrix = cost_matrix.unsqueeze(0)
#
#     batch_size, n, m = cost_matrix.shape
#     assignments = []
#
#     for b in range(batch_size):
#         cost = cost_matrix[b].detach().cpu().numpy()
#         row_ind, col_ind = linear_sum_assignment(cost)
#
#         assignment = torch.zeros((n, m), device=cost_matrix.device)
#         assignment[row_ind, col_ind] = 1.0
#         assignments.append(assignment)
#
#     result = torch.stack(assignments)
#     return result.squeeze(0) if batch_size == 1 else result

import torch
import torch.nn as nn


def log_sinkhorn_iterations(log_alpha, n_iters=20, eps=1e-8, check_convergence=False):
    """
    Log-domain Sinkhorn iterations for numerical stability.
    
    Args:
        log_alpha: Log assignment matrix [batch, n_det, n_track]
        n_iters: Maximum number of iterations
        eps: Convergence threshold (if check_convergence=True)
        check_convergence: Whether to check for early stopping
    """
    for i in range(n_iters):
        if check_convergence and i > 0:
            log_alpha_prev = log_alpha.clone()
        
        # Row normalization (detections)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        # Column normalization (tracks)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        
        # Early stopping if converged
        if check_convergence and i > 0:
            diff = torch.abs(log_alpha - log_alpha_prev).max()
            if diff < eps:
                break
    
    return log_alpha


def sinkhorn_algorithm(cost_matrix, n_iters=20, tau=0.1, eps=1e-8, unbalanced=True):
    """
    Sinkhorn algorithm for differentiable assignment.
    
    Args:
        cost_matrix: Cost matrix [batch, n_det, n_track]
        n_iters: Number of Sinkhorn iterations
        tau: Temperature parameter (higher = softer assignments)
        eps: Small constant for numerical stability
        unbalanced: If True, allows unbalanced assignment (important for tracking!)
    
    Returns:
        Soft assignment matrix [batch, n_det, n_track]
    """
    batch_size, n_det, n_track = cost_matrix.shape
    
    # Clamp cost matrix for numerical stability
    cost_matrix = torch.clamp(cost_matrix, min=-1e3, max=1e3)
    
    # Convert to log domain with temperature scaling
    log_alpha = -cost_matrix / tau
    
    if unbalanced:
        # Add dustbin rows/columns for unbalanced assignment
        # This allows detections to be "unmatched" and tracks to be "unmatched"
        dustbin_det = torch.full((batch_size, n_det, 1), 0.0, 
                                  device=cost_matrix.device, dtype=cost_matrix.dtype)
        dustbin_track = torch.full((batch_size, 1, n_track), 0.0, 
                                    device=cost_matrix.device, dtype=cost_matrix.dtype)
        dustbin_corner = torch.full((batch_size, 1, 1), 0.0,
                                     device=cost_matrix.device, dtype=cost_matrix.dtype)
        
        # Augment log_alpha with dustbins
        log_alpha = torch.cat([log_alpha, dustbin_det], dim=2)  # Add column [batch, n_det, n_track+1]
        dustbin_row = torch.cat([dustbin_track, dustbin_corner], dim=2)  # Create full row [batch, 1, n_track+1]
        log_alpha = torch.cat([log_alpha, dustbin_row], dim=1)  # Add row [batch, n_det+1, n_track+1]
    
    # Run Sinkhorn iterations in log domain
    log_alpha = log_sinkhorn_iterations(log_alpha, n_iters, eps, check_convergence=False)
    
    # Convert back from log domain
    assignment = torch.exp(log_alpha)
    
    if unbalanced:
        # Remove dustbin rows/columns
        assignment = assignment[:, :n_det, :n_track]
    
    # Normalize to ensure valid probabilities
    assignment = torch.clamp(assignment, min=eps, max=1.0)
    
    return assignment


class SinkhornMatcher(nn.Module):
    def __init__(self, n_iters=20, tau=0.1, hard_assignment=False, unbalanced=True):
        """
        Sinkhorn matcher for optimal transport assignment.
        
        Args:
            n_iters: Number of Sinkhorn iterations
            tau: Temperature parameter (0.05-0.2 typical; higher = softer)
            hard_assignment: If True, convert to hard 0/1 assignment
            unbalanced: If True, allows unbalanced assignment (recommended for tracking)
        """
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau
        self.hard_assignment = hard_assignment
        self.unbalanced = unbalanced

        print(f"[OK] SinkhornMatcher: {n_iters} iters, tau={tau}, hard={hard_assignment}, unbalanced={unbalanced}")

    def forward(self, cost_matrix):
        if cost_matrix.dim() == 2:
            cost_matrix = cost_matrix.unsqueeze(0)

        assignment = sinkhorn_algorithm(
            cost_matrix, 
            n_iters=self.n_iters, 
            tau=self.tau,
            unbalanced=self.unbalanced
        )

        if self.hard_assignment:
            hard_assignment = self.greedy_assignment(assignment)
            assignment = hard_assignment

        return assignment.squeeze(0) if assignment.size(0) == 1 else assignment

    def greedy_assignment(self, soft_assignment):
        batch_size, n_det, n_track = soft_assignment.shape
        hard = torch.zeros_like(soft_assignment)

        for b in range(batch_size):
            assignment_copy = soft_assignment[b].clone()

            for _ in range(min(n_det, n_track)):
                flat_idx = torch.argmax(assignment_copy)
                det_idx = flat_idx // n_track
                track_idx = flat_idx % n_track

                if assignment_copy[det_idx, track_idx] < 0.1:
                    break

                hard[b, det_idx, track_idx] = 1.0

                assignment_copy[det_idx, :] = -float('inf')
                assignment_copy[:, track_idx] = -float('inf')

        return hard


def hungarian_algorithm(cost_matrix):
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.dim() == 2:
        cost_matrix = cost_matrix.unsqueeze(0)

    batch_size, n, m = cost_matrix.shape
    assignments = []

    for b in range(batch_size):
        cost = cost_matrix[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        assignment = torch.zeros((n, m), device=cost_matrix.device)
        assignment[row_ind, col_ind] = 1.0
        assignments.append(assignment)

    result = torch.stack(assignments)
    return result.squeeze(0) if batch_size == 1 else result