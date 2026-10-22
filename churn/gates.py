"""Ворота качества: champion/challenger.

Новая модель уходит в прод, только если на одном и том же свежем
отложенном периоде она не хуже текущей прод-модели.
"""

from dataclasses import asdict, dataclass
from typing import Optional

from sklearn.metrics import roc_auc_score

from churn.data import TARGET, load_csv
from churn.registry import Registry


@dataclass
class GateDecision:
    promote: bool
    reason: str
    challenger_version: str
    challenger_auc: float
    champion_version: Optional[str] = None
    champion_auc: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide(registry: Registry, challenger_version: str, min_auc: float, min_improvement: float) -> GateDecision:
    challenger = registry.load(challenger_version)
    holdout = load_csv(registry.version_dir(challenger_version) / "holdout.csv")
    y = holdout[TARGET].to_numpy()
    ch_auc = round(float(roc_auc_score(y, challenger.predict_proba(holdout))), 4)

    if ch_auc < min_auc:
        return GateDecision(False, f"AUC {ch_auc} ниже порога {min_auc}", challenger_version, ch_auc)

    champion_version = registry.production_version()
    if champion_version is None:
        return GateDecision(True, "в проде нет модели", challenger_version, ch_auc)
    if champion_version == challenger_version:
        return GateDecision(False, "эта версия уже в проде", challenger_version, ch_auc, champion_version, ch_auc)

    champion = registry.load(champion_version)
    champ_auc = round(float(roc_auc_score(y, champion.predict_proba(holdout))), 4)
    if ch_auc >= champ_auc + min_improvement:
        reason = f"challenger {ch_auc} >= champion {champ_auc} + {min_improvement}"
        return GateDecision(True, reason, challenger_version, ch_auc, champion_version, champ_auc)
    reason = f"challenger {ch_auc} хуже champion {champ_auc}, оставляем текущую"
    return GateDecision(False, reason, challenger_version, ch_auc, champion_version, champ_auc)
