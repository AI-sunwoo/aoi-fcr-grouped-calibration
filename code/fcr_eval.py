"""보정 규칙과 평가 지표 공용 함수 (2026-11-23, 3차). 48번 분석과 같은 정의.

pooled : p(z) = (1 + R(z)) / (n + 1) <= alpha   (교차 conformal, 기대 놓침 규칙)
pac    : R(z) <= r*,  r* = max{r : P(Bin(n, alpha) <= r) <= delta}  (높은 확신 규칙, 교차적합에서는 근사)
R(z)   = sum_k #{c in C_k : c <= f_k(z)},  C_k = k겹 모델이 보류한 개발 불량의 점수
"""
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import average_precision_score

ALPHAS = [0.01, 0.02, 0.05, 0.10]
DELTA = 0.05


def pooled_rank(oof, fold, yd, ev):
    R, n = np.zeros(ev.shape[1]), 0
    for k in range(ev.shape[0]):
        c = np.sort(oof[(fold == k) & (yd == 1)])
        R += np.searchsorted(c, ev[k], side="right")
        n += len(c)
    return R, n


def pac_r(n, a, delta=DELTA):
    r = np.arange(n + 1)
    ok = r[binom.cdf(r, n, a) <= delta]
    return int(ok.max()) if len(ok) else -1


def passed(rule, oof, fold, yd, ev, a):
    R, n = pooled_rank(oof, fold, yd, ev)
    return (1 + R) / (n + 1) <= a if rule == "pooled" else R <= pac_r(n, a)


def slip_vol(ye, ps):
    return (float((ps & (ye == 1)).sum() / max(ye.sum(), 1)), float((ps & (ye == 0)).sum() / max((ye == 0).sum(), 1)))


def board_escape(ye, board, ps):
    d = pd.DataFrame({"b": board, "y": ye, "p": ps})
    dd = d[d.y == 1].groupby("b")["p"].all()
    return float(dd.mean()) if len(dd) else float("nan")


def fold_ap(yd, oof, fold):
    aps = [average_precision_score(yd[fold == k], oof[fold == k]) for k in np.unique(fold)
           if 0 < yd[fold == k].sum() < (fold == k).sum()]
    return float(np.mean(aps))


def summarise(yd, ye, board_ev, oof, fold, ev):
    """한 번의 교차적합 결과 → 지표 dict."""
    r = {"dev_ap_fold": fold_ap(yd, oof, fold), "eval_ap": float(average_precision_score(ye, ev.mean(0)))}
    for a in ALPHAS:
        for rule in ["pooled", "pac"]:
            ps = passed(rule, oof, fold, yd, ev, a)
            r[f"{rule}_slip@{a}"], r[f"{rule}_vol@{a}"] = slip_vol(ye, ps)
            if rule == "pooled":
                r[f"esc@{a}"] = board_escape(ye, board_ev, ps)
    return r


def siemens_half(df):
    """Siemens 시간순 반분의 자르는 위치 (2026-11-23, 모의 심사 1번 반영).

    SIEMENS_SPLIT 환경 변수가 "day"이면 행 수의 절반에 가장 가까운 날짜 경계(UTC 날짜 문자열 기준)에서 자른다
    → 개발·평가가 날짜와 타임스탬프 묶음을 하나도 공유하지 않는다. 그 밖에는 행 수의 절반(Pfab·Rothering 2025와
    같은 조건, 날짜 하나와 묶음 하나가 양쪽에 걸침).
    """
    import os

    import pandas as pd
    n = len(df)
    h = n // 2
    if os.environ.get("SIEMENS_SPLIT", "row") != "day":
        return h
    day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").values
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    return int(starts[np.argmin(np.abs(starts - h))])
