import torch
import torch.nn as nn
import torch.nn.functional as F


class AssignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_assignment, gt_assignment):
        pred_log = torch.log(pred_assignment + 1e-8)
        loss = -torch.sum(gt_assignment * pred_log) / gt_assignment.size(0)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, detection_features, track_features, assignment):
        det_norm = F.normalize(detection_features, p=2, dim=1)
        track_norm = F.normalize(track_features, p=2, dim=1)

        similarity = torch.mm(det_norm, track_norm.t()) / self.temperature

        pos_mask = assignment > 0.5
        neg_mask = assignment <= 0.5

        if pos_mask.sum() > 0:
            pos_sim = similarity[pos_mask]

            denom = torch.exp(similarity).sum(dim=1)
            pos_indices = pos_mask.nonzero(as_tuple=False)
            pos_denom = denom[pos_indices[:, 0]]

            pos_loss = -torch.log(torch.exp(pos_sim) / (pos_denom + 1e-8) + 1e-8)
            pos_loss = pos_loss.mean()
        else:
            pos_loss = torch.tensor(0.0, device=similarity.device)

        if neg_mask.sum() > 0:
            neg_sim = similarity[neg_mask]
            neg_loss = F.relu(neg_sim - self.margin).mean()
        else:
            neg_loss = torch.tensor(0.0, device=similarity.device)

        return pos_loss + neg_loss


class IDConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, current_ids, previous_ids, assignment):
        if len(previous_ids) == 0 or len(current_ids) == 0:
            return torch.tensor(0.0, device=assignment.device)

        matched_indices = (assignment > 0.5).nonzero(as_tuple=False)

        if len(matched_indices) == 0:
            return torch.tensor(0.0, device=assignment.device)

        id_switches = 0
        for det_idx, track_idx in matched_indices:
            current_id = current_ids[det_idx]
            previous_id = previous_ids[track_idx]
            if current_id != previous_id:
                id_switches += 1

        loss = torch.tensor(id_switches / len(matched_indices),
                            dtype=torch.float32, device=assignment.device)
        return loss


class CombinedTrackingLoss(nn.Module):
    def __init__(self, w_assignment=1.0, w_contrastive=0.5, w_consistency=0.3):
        super().__init__()
        self.w_assignment = w_assignment
        self.w_contrastive = w_contrastive
        self.w_consistency = w_consistency

        self.assignment_loss = AssignmentLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.consistency_loss = IDConsistencyLoss()

    def forward(self, pred_assignment, gt_assignment,
                detection_features, track_features,
                current_ids, previous_ids):
        l_assign = self.assignment_loss(pred_assignment, gt_assignment)
        l_contrast = self.contrastive_loss(detection_features, track_features, gt_assignment)
        l_consist = self.consistency_loss(current_ids, previous_ids, gt_assignment)

        total_loss = (self.w_assignment * l_assign +
                      self.w_contrastive * l_contrast +
                      self.w_consistency * l_consist)

        return total_loss, {
            'assignment_loss': l_assign.item(),
            'contrastive_loss': l_contrast.item(),
            'consistency_loss': l_consist.item(),
            'total_loss': total_loss.item()
        }