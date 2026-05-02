from dataclasses import dataclass
from pathlib import Path

from model.tikr_2016_2024 import Analysis


@dataclass
class Parameters:
    # Chemin du projet
    script_path: Path | str

    # Initialisation des paramètres
    weights: dict[str, float]
    weights_sum: float
    seuil_freq_select: float
    taux_sans_risque: float

    start_year: int
    end_year: int

    nbe_monte_carlo: int
    nbe_stocks: int

    analysis: Analysis | None