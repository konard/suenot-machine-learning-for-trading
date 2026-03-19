import torch
import torch.nn.functional as F


def infonce_loss(anchors, positives, negatives=None, temperature=0.07):
    """
    Computes the InfoNCE loss for contrastive learning.

    If negatives are not provided, uses other samples in the batch as negatives
    (in-batch negative sampling).

    Args:
        anchors:    (batch_size, embed_dim) — anchor embeddings
        positives:  (batch_size, embed_dim) — positive embeddings (paired with anchors)
        negatives:  (batch_size, num_neg, embed_dim) or None — explicit negatives
        temperature: float — temperature scaling parameter

    Returns:
        loss: scalar tensor — mean InfoNCE loss
    """
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)

    # Positive logits: (batch_size,)
    pos_logits = (anchors * positives).sum(dim=-1, keepdim=True) / temperature

    if negatives is not None:
        # Explicit negatives: (batch_size, num_neg, embed_dim)
        negatives = F.normalize(negatives, dim=-1)
        # neg_logits: (batch_size, num_neg)
        neg_logits = torch.bmm(negatives, anchors.unsqueeze(-1)).squeeze(-1) / temperature
        # Concatenate: (batch_size, 1 + num_neg)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
    else:
        # In-batch negatives: every other sample in the batch is a negative
        # Similarity matrix: (batch_size, batch_size)
        sim_matrix = torch.mm(anchors, positives.t()) / temperature
        logits = sim_matrix

    # Labels: the positive is always at index 0 (explicit) or diagonal (in-batch)
    batch_size = anchors.size(0)
    if negatives is not None:
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)
    else:
        labels = torch.arange(batch_size, device=anchors.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def infonce_loss_with_accuracy(anchors, positives, negatives=None, temperature=0.07):
    """
    Computes InfoNCE loss and classification accuracy (for monitoring).

    Returns:
        loss: scalar tensor
        accuracy: float — fraction of correctly identified positives
    """
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)

    pos_logits = (anchors * positives).sum(dim=-1, keepdim=True) / temperature

    batch_size = anchors.size(0)

    if negatives is not None:
        negatives = F.normalize(negatives, dim=-1)
        neg_logits = torch.bmm(negatives, anchors.unsqueeze(-1)).squeeze(-1) / temperature
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)
    else:
        logits = torch.mm(anchors, positives.t()) / temperature
        labels = torch.arange(batch_size, device=anchors.device)

    loss = F.cross_entropy(logits, labels)

    # Accuracy
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()

    return loss, accuracy
