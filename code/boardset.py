"""BoardSet: 같은 보드·패널(또는 같은 시각 묶음)의 AOI 호출을 한 집합으로 함께 보는 신경망.

구조
  항목 인코더: 수치 특징 → MLP, 범주 특징 → 임베딩 합
  집합 층    : 같은 집합 안에서만 주의(attention)하는 Transformer 인코더 2층 (패딩 마스크)
  출력       : [항목 인코딩, 집합 층 출력]을 이어 MLP → 진짜 불량일 로짓(항목마다)
집합 안 항목 수는 제각각이라 크기가 비슷한 집합끼리 묶어 패딩한다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


def prep_numeric(X_tr: np.ndarray, X_list):
    """부호 보존 log1p → 훈련 기준 표준화 → ±8 자르기. 결측은 0(평균)으로 두고 결측 표시 열을 붙인다."""
    def tf(X):
        return np.sign(X) * np.log1p(np.abs(X))
    A = tf(X_tr.astype(np.float64))
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    miss_cols = np.where(np.isnan(X_tr).any(axis=0))[0]
    out = []
    for X in [X_tr] + list(X_list):
        Z = (tf(X.astype(np.float64)) - mu) / sd
        M = np.isnan(Z[:, miss_cols]).astype(np.float32)
        Z = np.nan_to_num(np.clip(Z, -8, 8), nan=0.0).astype(np.float32)
        out.append(np.concatenate([Z, M], axis=1))
    return out


class BoardSetNet(nn.Module):
    def __init__(self, n_num, cat_cards, d=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.num = nn.Sequential(nn.Linear(n_num, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d))
        self.embs = nn.ModuleList([nn.Embedding(c + 1, d) for c in cat_cards])  # 0 = 처음 보는 값
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.setenc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, xn, xc, pad):
        h = self.num(xn)
        for j, e in enumerate(self.embs):
            h = h + e(xc[..., j])
        z = self.setenc(h, src_key_padding_mask=pad)
        return self.head(torch.cat([h, z], dim=-1)).squeeze(-1)


def make_batches(set_ids, rows, token_budget=8192, shuffle=False, rng=None):
    """set_ids: 각 행의 집합 번호(rows 안의 행만). 반환: 배치 목록, 각 배치는 집합별 행 번호 목록."""
    order = np.argsort(set_ids[rows], kind="stable")
    r_sorted = rows[order]
    s_sorted = set_ids[r_sorted]
    cut = np.flatnonzero(np.diff(s_sorted)) + 1
    sets = np.split(r_sorted, cut)
    sets.sort(key=len)
    batches, cur, cur_max = [], [], 0
    for s in sets:
        m = max(cur_max, len(s))
        if cur and m * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, cur_max = [], 0
            m = len(s)
        cur.append(s)
        cur_max = m
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def to_tensors(batch, XN, XC, y, device):
    L = max(len(s) for s in batch)
    B = len(batch)
    xn = np.zeros((B, L, XN.shape[1]), np.float32)
    xc = np.zeros((B, L, XC.shape[1]), np.int64)
    yy = np.zeros((B, L), np.float32)
    pad = np.ones((B, L), bool)
    for i, s in enumerate(batch):
        k = len(s)
        xn[i, :k], xc[i, :k], pad[i, :k] = XN[s], XC[s], False
        if y is not None:
            yy[i, :k] = y[s]
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(xn), t(xc), t(yy), t(pad)


def predict(model, batches, XN, XC, n_rows, device):
    model.eval()
    out = np.zeros(n_rows, np.float32)
    with torch.no_grad():
        for b in batches:
            xn, xc, _, pad = to_tensors(b, XN, XC, None, device)
            p = torch.sigmoid(model(xn, xc, pad)).cpu().numpy()
            for i, s in enumerate(b):
                out[s] = p[i, :len(s)]
    return out


def fit_predict(XN_tr, XC_tr, y_tr, set_tr, XN_list, XC_list, set_list, cat_cards, seed, n_init=1, **kw):
    """n_init개 초기화로 학습해 점수를 평균한다(같은 훈련 자료, 다른 난수)."""
    outs = None
    for k in range(n_init):
        o = _fit_predict_one(XN_tr, XC_tr, y_tr, set_tr, XN_list, XC_list, set_list, cat_cards,
                             seed * 1000 + k, **kw)
        outs = o if outs is None else [a + b for a, b in zip(outs, o)]
    return [a / n_init for a in outs]


def _fit_predict_one(XN_tr, XC_tr, y_tr, set_tr, XN_list, XC_list, set_list, cat_cards, seed,
                     device="cuda", max_epochs=40, patience=6, val_frac=0.1, lr=2e-3, verbose=False,
                     d=64, layers=2, heads=4, dropout=0.1, pw="sqrt", val_groups=None):
    """훈련 집합 일부(val_frac, 집합 단위)를 떼어 조기 종료에 쓰고, 나머지로 학습한다.

    XN_*: 이미 prep_numeric을 거친 배열. XC_*: 범주 번호(0 = 모름). set_*: 집합 번호.
    반환: XN_list 각각에 대한 점수.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # 조기 종료용 검증 몫은 집합 단위로 뗀다. val_groups를 주면(집합 크기 1 변형) 그 묶음(보드) 단위로 뗀다.
    vg = set_tr if val_groups is None else val_groups
    uniq = np.unique(vg)
    val_sets = rng.choice(uniq, size=max(1, int(len(uniq) * val_frac)), replace=False)
    is_val = np.isin(vg, val_sets)
    tr_rows, va_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    pos = max(float(y_tr[tr_rows].sum()), 1.0)
    ratio = (len(tr_rows) - pos) / pos
    pos_weight = torch.tensor(math.sqrt(ratio) if pw == "sqrt" else (ratio if pw == "full" else 1.0), device=device)
    model = BoardSetNet(XN_tr.shape[1], cat_cards, d=d, heads=heads, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    va_batches = make_batches(set_tr, va_rows)
    best, best_state, bad_epochs = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        for b in make_batches(set_tr, tr_rows, shuffle=True, rng=rng):
            xn, xc, yy, pad = to_tensors(b, XN_tr, XC_tr, y_tr, device)
            logit = model(xn, xc, pad)
            loss = (lossf(logit, yy) * (~pad)).sum() / (~pad).sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = predict(model, va_batches, XN_tr, XC_tr, len(y_tr), device)[va_rows]
        ap = average_precision_score(y_tr[va_rows], pv) if y_tr[va_rows].sum() else 0.0
        if verbose:
            print("epoch", ep, "val AP", round(ap, 4), flush=True)
        if ap > best:
            best, bad_epochs = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    outs = []
    for XN, XC, S in zip(XN_list, XC_list, set_list):
        rows = np.arange(len(XN))
        outs.append(predict(model, make_batches(S, rows), XN, XC, len(XN), device))
    return outs
