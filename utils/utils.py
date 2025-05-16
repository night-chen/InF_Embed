import json
import torch
import torch.distributed as dist

from typing import List, Union, Optional, Tuple, Mapping, Dict


def save_json_to_file(objects: Union[List, dict], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list), 'Only list can be saved in line by line format'

    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


@torch.no_grad()
def select_grouped_indices(scores: torch.Tensor,
                           group_size: int,
                           start: int = 0) -> torch.Tensor:
    assert len(scores.shape) == 2
    batch_size = scores.shape[0]
    # assert batch_size * group_size <= scores.shape[1]

    indices = torch.arange(0, group_size, dtype=torch.long)
    indices = indices.repeat(batch_size, 1)
    indices += torch.arange(0, batch_size, dtype=torch.long).unsqueeze(-1) * group_size
    indices += start

    return indices.to(scores.device)


def full_contrastive_scores_and_labels(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = True,
        contrast_mode: str = "same_tower") -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    train_n_passages = key.shape[0] // query.shape[0]
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    labels = labels * train_n_passages

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    if not use_all_pairs:
        return qk, labels

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    
    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf'))

    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf'))

    # Select which scores to use based on contrast_mode
    if contrast_mode == "qk":
        # qk: contrast passage
        scores = qk
    elif contrast_mode == "kq":
        # kq: contrast query + instruction
        scores = kq
    elif contrast_mode == "no_trick":
        # qk, kq: no trick
        kq.fill_diagonal_(float('-inf'))
        scores = torch.cat([qk, kq], dim=-1)
    elif contrast_mode == "same_tower":
        # qk, kq, qq, kk: same tower
        kq.fill_diagonal_(float('-inf'))
        scores = torch.cat([qk, kq, qq, kk], dim=-1)
    else:
        raise ValueError(f"Unknown contrast_mode: {contrast_mode}")

    # query, passage: contrast instruction
    # instruction, passage: contrast query
    return scores, labels

def full_contrastive_scores_and_labels_add(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = True,
        contrast_mode: str = "same_tower") -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    train_n_passages = key.shape[0] // query.shape[0]
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    labels = labels * train_n_passages

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    if not use_all_pairs:
        return qk, labels

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    
    qq = torch.mm(query, query.t())

    kk = torch.mm(sliced_key, sliced_key.t())

    # Select which scores to use based on contrast_mode
    if contrast_mode == "qk":
        # qk: contrast passage
        scores = qk
    elif contrast_mode == "kq":
        # kq: contrast query + instruction
        scores = kq
    elif contrast_mode == "no_trick":
        # qk, kq: no trick
        scores = torch.cat([qk, kq], dim=-1)
    elif contrast_mode == "same_tower":
        # qk, kq, qq, kk: same tower
        scores = torch.cat([qk, kq, qq, kk], dim=-1)
    else:
        raise ValueError(f"Unknown contrast_mode: {contrast_mode}")

    # query, passage: contrast instruction
    # instruction, passage: contrast query
    return scores, labels


def angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 1.0):
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()

    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    # (a+bi) / (c+di)
    # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
    # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
    # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    y_pred = torch.concat((re, im), dim=1)
    y_pred = torch.abs(torch.sum(y_pred, dim=1)) * tau  # absolute delta angle
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def slice_batch_dict(batch_dict: Dict[str, torch.Tensor], prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in batch_dict.items() if k.startswith(prefix)}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, round_digits: int = 3):
        self.name = name
        self.round_digits = round_digits
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '{}: {}'.format(self.name, round(self.avg, self.round_digits))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        #f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        f"trainable params: {trainable_params} || all params: {all_param}"
    )

def full_contrastive_scores_and_labels_with_neg(
        query: torch.Tensor,         # Shape: (W*B, D)
        key: torch.Tensor,           # Shape: (W*B*N, D)
        neg_query: torch.Tensor,     # Shape: (W, B, B, D)
        contrast_mode: str = "same_tower",
        div_neg_batch: int = 2,
        use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    # Infer dimensions
    WB, B_dived, D = neg_query.shape
    N = key.shape[0] // query.shape[0] # train_n_passages
    B = B_dived * div_neg_batch
    W = WB // B

    assert query.shape == (W * B, D)
    assert key.shape == (W * B * N, D)

    # Calculate positive labels (indices of positive passages)
    labels = torch.arange(0, W * B, dtype=torch.long, device=query.device)
    labels = labels * N # Shape: (W*B)

    # qk: Scores between queries and all passages
    # (W*B, D) x (D, W*B*N) -> (W*B, W*B*N)
    qk = torch.mm(query, key.t())

    # sliced_key: Positive passage embeddings corresponding to each query
    # Shape: (W*B, D)
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # kq: Scores between positive passages and all queries
    # (W*B, D) x (D, W*B) -> (W*B, W*B)
    kq = torch.mm(sliced_key, query.t())

    # qq: Scores between queries
    # (W*B, D) x (D, W*B) -> (W*B, W*B)
    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf')) # Mask self-similarity

    # kk: Scores between positive passages
    # (W*B, D) x (D, W*B) -> (W*B, W*B)
    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf')) # Mask self-similarity

    # neg_query: w*b * b * d
    # sliced_key: w*b * d * 1
    sliced_passage = sliced_key.view(W*B, D, 1)
    qq_neg = torch.bmm(neg_query, sliced_passage).squeeze(-1)

    if contrast_mode == "same_tower_with_neg":
        kq.fill_diagonal_(float('-inf')) # Ensure kq diagonal (p_i vs q_i) is masked if needed
        qq_neg[:, 0] = float('-inf') 
        scores = torch.cat([qk, kq, qq, kk, qq_neg], dim=-1)
    elif contrast_mode == "qk_with_neg":
        qq_neg[:, 0] = float('-inf')
        scores = torch.cat([qk, qq_neg], dim=-1)
    elif contrast_mode == "kq_with_neg":
        qq_neg[:, 0] = float('-inf')
        scores = torch.cat([kq, qq_neg], dim=-1)
    elif contrast_mode == "no_trick_with_neg":
        kq.fill_diagonal_(float('-inf'))
        qq_neg[:, 0] = float('-inf')
        scores = torch.cat([qk, kq, qq_neg], dim=-1)
    elif contrast_mode == "only_neg":
        scores = torch.cat([qq_neg], dim=-1)
        labels = torch.zeros(W*B, dtype=torch.long, device=query.device)
    else:
        raise ValueError(f"Unknown contrast_mode: {contrast_mode}")

    return scores, labels




def full_contrastive_scores_and_labels_with_neg_add(
        query: torch.Tensor,         # Shape: (W*B, D)
        key: torch.Tensor,           # Shape: (W*B*N, D)
        neg_query: torch.Tensor,     # Shape: (W, B, B, D)
        contrast_mode: str = "same_tower",
        div_neg_batch: int = 2,
        use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    # Infer dimensions
    WB, B_dived, D = neg_query.shape
    N = key.shape[0] // query.shape[0] # train_n_passages
    B = B_dived * div_neg_batch
    W = WB // B

    assert query.shape == (W * B, D)
    assert key.shape == (W * B * N, D)

    # Calculate positive labels (indices of positive passages)
    labels = torch.arange(0, W * B, dtype=torch.long, device=query.device)
    labels = labels * N # Shape: (W*B)

    neg_labels = torch.zeros(W*B, dtype=torch.long, device=query.device)
    # qk: Scores between queries and all passages
    # (W*B, D) x (D, W*B*N) -> (W*B, W*B*N)
    qk = torch.mm(query, key.t())

    # sliced_key: Positive passage embeddings corresponding to each query
    # Shape: (W*B, D)
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # kq: Scores between positive passages and all queries
    # (W*B, D) x (D, W*B) -> (W*B, W*B)
    kq = torch.mm(sliced_key, query.t())

    # qq: Scores between queries
    # (W*B, D) x (D, W*B) -> (W*B, W*B)
    qq = torch.mm(query, query.t())


    # kk: Scores between positive passages
    # (W*B, D) x (D, W*B) -> (W*B, W*B)
    kk = torch.mm(sliced_key, sliced_key.t())

    # neg_query: w*b * b * d
    # sliced_key: w*b * d * 1
    sliced_passage = sliced_key.view(W*B, D, 1)
    qq_neg = torch.bmm(neg_query, sliced_passage).squeeze(-1)

    if contrast_mode == "same_tower_with_neg":
        scores = torch.cat([qk, kq, qq, kk, qq_neg], dim=-1)
    elif contrast_mode == "qk_with_neg":
        scores = torch.cat([qk, qq_neg], dim=-1)
    elif contrast_mode == "kq_with_neg":
        scores = torch.cat([kq, qq_neg], dim=-1)
    elif contrast_mode == "no_trick_with_neg":
        scores = torch.cat([qk, kq, qq_neg], dim=-1)
    elif contrast_mode == "only_neg":
        scores = torch.cat([qq_neg], dim=-1)
    else:
        raise ValueError(f"Unknown contrast_mode: {contrast_mode}")

    return scores, labels, neg_labels


if __name__ == '__main__':
    query = torch.randn(4, 16)
    key = torch.randn(4 * 3, 16)
    scores, labels = full_contrastive_scores_and_labels(query, key)
    print(scores.shape)
    print(labels)