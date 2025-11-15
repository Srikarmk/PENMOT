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


def log_sinkhorn_iterations(log_alpha, n_iters=20):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha


def sinkhorn_algorithm(cost_matrix, n_iters=20, tau=0.05, eps=1e-8):
    batch_size, n, m = cost_matrix.shape

    log_alpha = -cost_matrix / tau

    log_alpha = log_sinkhorn_iterations(log_alpha, n_iters)

    assignment = torch.exp(log_alpha)

    return assignment


class SinkhornMatcher(nn.Module):
    def __init__(self, n_iters=20, tau=0.05, hard_assignment=False):
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau
        self.hard_assignment = hard_assignment

        print(f"[OK] SinkhornMatcher: {n_iters} iters, tau={tau}, hard={hard_assignment}")

    def forward(self, cost_matrix):
        if cost_matrix.dim() == 2:
            cost_matrix = cost_matrix.unsqueeze(0)

        assignment = sinkhorn_algorithm(cost_matrix, self.n_iters, self.tau)

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