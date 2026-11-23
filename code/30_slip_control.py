"""놓침 한도를 지키는 문턱값 운용 시뮬레이션 (2026-11-23).

평가 기간을 시간순 덩어리(chunk)로 나누고, 덩어리마다 그 앞까지 '실제로 알 수 있었던' 정보만으로
문턱값을 정해 적용한다. 점수 모델은 개발 기간으로 학습해 고정한다(재학습 없음).

라벨이 생기는 방식(현실 조건)
- 점수가 문턱값 이상이라 사람에게 간 호출: 작업자가 확인하므로 라벨이 생긴다.
- 자동 통과한 호출: 감사 비율 a로 무작위 표본만 사람이 확인해 라벨이 생긴다(가중치 1/a).

정책
  static_grouped : 개발 기간 묶음 단위 OOF로 정한 고정 문턱값
  static_leaky   : 개발 기간 행 단위 OOF로 정한 고정 문턱값(Pfab식, 있을 때만)
  recal          : 최근 W개 덩어리의 관측 라벨(가중)로 놓침 추정치가 한도 이하가 되는 가장 큰 문턱값.
                   최근 불량이 m_min개보다 적으면 개발 OOF 불량을 보탠다.
  recal_ucb      : recal과 같되 놓침의 윌슨 상한(유효 표본 수 사용)이 한도 이하가 되게(보수적)
  oracle         : 덩어리마다 정답을 보고 고른 문턱값(상한 참고값)

사용: python -I 30_slip_control.py <결과 폴더(점수)> <출력 폴더> <모델 이름들(쉼표)>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_slip  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = sys.argv[3].split(",")
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.01, 0.02, 0.05]
AUDIT = [0.05, 0.10]
W, M_MIN, N_CHUNKS, Z = 4, 50, 20, 1.2816  # Z: 단측 90% 상한


def wilson_upper(p, n_eff):
    if n_eff <= 0:
        return 1.0
    den = 1 + Z * Z / n_eff
    c = p + Z * Z / (2 * n_eff)
    r = Z * np.sqrt(p * (1 - p) / n_eff + Z * Z / (4 * n_eff * n_eff))
    return min(1.0, (c + r) / den)


def choose_threshold(sc_def, w_def, alpha, ucb):
    """불량 항목의 (점수, 가중치)로, 추정 놓침 <= alpha 인 가장 큰 문턱값. 통과 = 점수 < 문턱값."""
    if len(sc_def) == 0:
        return -np.inf
    o = np.argsort(sc_def)
    s, w = sc_def[o], w_def[o]
    tot = w.sum()
    n_eff = tot * tot / (w * w).sum()
    cum = np.concatenate([[0.0], np.cumsum(w)])  # cum[k] = 아래 k개(통과)의 가중 합
    best = s[0]  # 아무 불량도 통과시키지 않는 문턱값
    for k in range(1, len(s)):
        p = cum[k] / tot
        est = wilson_upper(p, n_eff) if ucb else p
        if est <= alpha:
            best = s[k]
        else:
            break
    return float(best)


def simulate(y_ev, sc_ev, y_dev, oof_dev, chunks, alpha, audit, policy, rng, oof_leaky=None):
    if policy == "static_grouped":
        t_fix = threshold_for_slip(y_dev, oof_dev, alpha)
    elif policy == "static_leaky":
        t_fix = threshold_for_slip(y_dev, oof_leaky, alpha)
    dev_def = oof_dev[y_dev == 1]
    hist = []  # 덩어리별 관측: (불량 점수, 가중치)
    out = []
    n_audit = 0
    for k, idx in enumerate(chunks):
        yy, ss = y_ev[idx], sc_ev[idx]
        if policy.startswith("static"):
            t = t_fix
        elif policy == "oracle":
            t = threshold_for_slip(yy, ss, alpha)
        else:
            rec = hist[-W:]
            sd = np.concatenate([h[0] for h in rec]) if rec else np.array([])
            wd = np.concatenate([h[1] for h in rec]) if rec else np.array([])
            if wd.sum() < M_MIN:  # 최근 불량이 모자라면 개발 기간 불량을 보탠다
                sd = np.concatenate([sd, dev_def])
                wd = np.concatenate([wd, np.ones(len(dev_def))])
            t = choose_threshold(sd, wd, alpha, ucb=(policy == "recal_ucb"))
        slip, vol = slip_and_volume(yy, ss, t)
        out.append({"slip": slip, "vol": vol, "n_bad": int(yy.sum()), "missed": int(((ss < t) & (yy == 1)).sum()),
                    "passed_good": int(((ss < t) & (yy == 0)).sum()), "n_good": int((yy == 0).sum())})
        # 이 덩어리에서 생긴 라벨(사람에게 간 것 + 감사한 통과 건)
        passed = ss < t
        audited = passed & (rng.random(len(ss)) < audit)
        n_audit += int(audited.sum())
        obs_def = (yy == 1) & (~passed | audited)
        w = np.where(passed, 1.0 / audit, 1.0)
        hist.append((ss[obs_def], w[obs_def]))
    tot_bad = sum(o["n_bad"] for o in out)
    tot_good = sum(o["n_good"] for o in out)
    valid = [o for o in out if o["n_bad"] >= 10]
    return {"pooled_slip": sum(o["missed"] for o in out) / max(tot_bad, 1),
            "pooled_vol": sum(o["passed_good"] for o in out) / max(tot_good, 1),
            "max_chunk_slip": max(o["slip"] for o in valid) if valid else None,
            "frac_chunks_over": float(np.mean([o["slip"] > alpha for o in valid])) if valid else None,
            "audited": n_audit, "chunks": out}


meta = dict(np.load(RES / "siemens_meta.npz"))
y, half = meta["y"], int(meta["half"])
y_dev, y_ev = y[:half], y[half:]
chunks = np.array_split(np.arange(len(y_ev)), N_CHUNKS)
report = {}
for model in MODELS:
    runs = sorted(RES.glob(f"siemens_{model}_seed*.npz"))
    if not runs:
        continue
    report[model] = {}
    for alpha in ALPHAS:
        for audit in AUDIT:
            pols = ["static_grouped", "recal", "recal_ucb", "oracle"]
            if "oof_leaky" in np.load(runs[0]).files:
                pols.insert(1, "static_leaky")
            for pol in pols:
                rs = []
                for i, f in enumerate(runs):
                    r = dict(np.load(f))
                    rng = np.random.default_rng(1000 + i)
                    rs.append(simulate(y_ev, r["eval"], y_dev, r["oof_grouped"], chunks, alpha, audit, pol, rng,
                                       r.get("oof_leaky")))
                key = f"a{alpha}_audit{audit}_{pol}"
                agg = {k: float(np.mean([x[k] for x in rs])) for k in
                       ["pooled_slip", "pooled_vol", "max_chunk_slip", "frac_chunks_over", "audited"]}
                report[model][key] = agg
                print(model, key, {k: round(v, 4) for k, v in agg.items()}, flush=True)
(OUT / "slip_control_siemens.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
