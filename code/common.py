"""공통 함수: 문턱값 고르기와 현장 지표(놓침, 확인 물량 감소)."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

# 점수가 높을수록 진짜 불량(1)이다. 점수가 문턱값 t보다 낮으면 "가성으로 보고 사람 확인 없이 통과"시킨다.


def slip_and_volume(y, score, t):
    """놓침 비율 = 진짜 불량 중 통과된 비율, 확인 물량 감소 = 가성 중 통과된 비율."""
    y = np.asarray(y).astype(int)
    passed = np.asarray(score) < t
    n_bad = max(int(y.sum()), 1)
    n_good = max(int((1 - y).sum()), 1)
    slip = float((passed & (y == 1)).sum() / n_bad)
    volume = float((passed & (y == 0)).sum() / n_good)
    return slip, volume


def threshold_for_slip(y, score, slip_target):
    """놓침 비율이 slip_target 이하가 되는 가장 큰 문턱값(가장 많이 통과)을 고른다.

    진짜 불량의 점수를 오름차순으로 놓고, 아래에서 k개까지만 통과를 허용한다.
    k = floor(slip_target * 불량 수). 문턱값은 k+1번째로 낮은 불량 점수이다.
    """
    y = np.asarray(y).astype(int)
    bad = np.sort(np.asarray(score)[y == 1])
    if len(bad) == 0:
        return -np.inf
    k = int(np.floor(slip_target * len(bad)))
    return float(bad[min(k, len(bad) - 1)])


def threshold_for_f1(y, score):
    """F1을 가장 크게 하는 문턱값(점수 >= t 이면 불량으로 판정)."""
    y = np.asarray(y).astype(int)
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 400)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y, (score >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def evaluate(y, score, t_f1, t_slip):
    y = np.asarray(y).astype(int)
    slip, volume = slip_and_volume(y, score, t_slip)
    return {
        "n": int(len(y)),
        "n_bad": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)) if y.sum() else None,
        "f1_bad": float(f1_score(y, (score >= t_f1).astype(int), zero_division=0)),
        "slip": slip,
        "volume_reduction": volume,
    }


SLIP_TARGETS = [0.01, 0.02, 0.05, 0.10]


def volume_at_slips(y_cal, score_cal, y_te, score_te, targets=SLIP_TARGETS):
    """보정 자료(교차검증 예측)로 정한 문턱값과, 시험 정답을 보고 정한 문턱값(참고)에서의 놓침·물량."""
    out = {}
    for s in targets:
        t = threshold_for_slip(y_cal, score_cal, s)
        to = threshold_for_slip(y_te, score_te, s)
        sl, vo = slip_and_volume(y_te, score_te, t)
        slo, voo = slip_and_volume(y_te, score_te, to)
        out[str(s)] = {"slip": sl, "volume_reduction": vo, "oracle_slip": slo, "oracle_volume_reduction": voo}
    return out
