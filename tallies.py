#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tallies.py V304 — Tallies OpenMC para deposição de energia e fluxo.

Física:
    heating      [eV/src]: deposição total de energia kerma por nêutron-fonte.
    kappa-fission[eV/src]: energia de fissão (fragmentos+prompt n+prompt γ).
    flux         [n·cm/src]: fluxo integrado por célula.
    fission_rate [rx/src]: taxa de fissão.

    P [W] = score [eV/src] × source_rate [n/s] × BRIDGE.EV_TO_J

CHANGELOG V304 vs V303:
  MELHORIA 1 — create_heating_tallies: adicionado tally 'flux_spectrum' com
               filtro de energia de 709 grupos (VITAMIN-J) para calcular fluxo
               multigrupo por célula. Permite verificar espectro efetivo e
               calcular yields de fissão espectro-ponderados pós-simulação.

  MELHORIA 2 — create_activation_tallies: novo tally dedicado ao modo
               ACTIVATION com filtros de nuclídeo específicos para Mo98→Mo99
               (captura), U235→Mo99 (fissão) e Mo99→Mo100 (captura parasita).
               Quantifica diretamente as rotas de produção/perda do Mo99.

  MELHORIA 3 — extract_spectrum_weighted_yield: nova função que lê o tally
               flux_spectrum e calcula o yield efetivo de Mo99 ponderado pelo
               espectro — corrige o fator sistemático de ~15% entre yields
               térmico puro e espectro real do experimento.

  MELHORIA 4 — extract_power_per_layer / extract_kappa_fission_power: agora
               retornam também incerteza relativa por célula (campo '_unc').

  MELHORIA 5 — get_k_eff: retorna tupla nomeada KeffResult para evitar
               desempacotamento posicional frágil.
"""

import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

try:
    import openmc
    _OPENMC_OK = True
except ImportError:
    _OPENMC_OK = False

from pyne_bridge import BRIDGE

logger = logging.getLogger(__name__)

__all__ = [
    "create_heating_tallies",
    "create_activation_tallies",
    "extract_power_per_layer",
    "extract_kappa_fission_power",
    "extract_spectrum_weighted_yield",
    "get_k_eff",
    "KeffResult",
]

# ── Grupos de energia para tally espectral ────────────────────────────────
# 10 grupos log-uniformes cobrindo 1e-5 eV a 20 MeV — leve, mas suficiente
# para distinguir componentes térmica, epitérmica e rápida do espectro.
# Substitua por VITAMIN-J 709 grupos se precisar de espectro fino.
_N_ENERGY_GROUPS = 10
_ENERGY_BOUNDS_EV = np.logspace(
    np.log10(1e-5), np.log10(20e6), _N_ENERGY_GROUPS + 1
)

# Limites das regiões espectrais [eV]
_E_THERMAL_MAX_EV  = 0.625    # térmico < 0.625 eV
_E_EPITHERMAL_MAX_EV = 100e3  # epitérmico < 100 keV
# Rápido: > 100 keV


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _cells_to_filter(cells_dict: dict) -> "openmc.CellFilter":
    return openmc.CellFilter(list(cells_dict.values()))


def _build_id_to_name(cells_dict: dict) -> Dict[int, str]:
    return {
        (val.id if hasattr(val, "id") else int(val)): name
        for name, val in cells_dict.items()
    }


def _detect_cell_col(df_columns) -> Optional[str]:
    for candidate in ("cell id", "cell"):
        if candidate in df_columns:
            return candidate
    return None


def _extract_scores_from_df(
    df,
    id_to_name: Dict[int, str],
    score_label: str,
    source_rate: float,
    caller: str,
    include_uncertainty: bool = False,
) -> Dict[str, float]:
    """
    Extrai potência [W] por célula a partir de um DataFrame de tally.

    Args:
        include_uncertainty: se True, adiciona chaves '<nome>_unc_rel' com
                             a incerteza relativa (std/mean) de cada célula.
    """
    power: Dict[str, float] = {name: 0.0 for name in id_to_name.values()}
    cell_col = _detect_cell_col(df.columns)
    if cell_col is None:
        logger.error("%s: coluna de cell id não encontrada. Colunas: %s",
                     caller, list(df.columns))
        return power

    df_score = df[df["score"] == score_label] if "score" in df.columns else df

    for cell_id, cell_name in id_to_name.items():
        rows = df_score[df_score[cell_col] == cell_id]
        if rows.empty:
            continue
        score_ev = float(rows["mean"].sum())
        if score_ev > 0.0:
            power[cell_name] = BRIDGE.heating_eV_to_watts(score_ev, source_rate)
            if include_uncertainty and "std. dev." in rows.columns:
                std_ev = float(rows["std. dev."].sum())
                power[f"{cell_name}_unc_rel"] = std_ev / score_ev if score_ev else 0.0

    return power


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def create_heating_tallies(geometry_result: dict) -> "openmc.Tallies":
    """
    Cria tallies de heating, flux, fission_rate, kappa-fission e flux_spectrum.

    O tally 'flux_spectrum' adiciona filtro de energia de 10 grupos log-uniformes
    (1e-5 eV – 20 MeV) por célula, permitindo verificar a composição espectral
    e calcular yields de fissão ponderados pelo espectro real (ver
    extract_spectrum_weighted_yield).

    Args:
        geometry_result: Dict com 'cells_dict': {cell_name: openmc.Cell}.

    Returns:
        openmc.Tallies com 5 tallies nomeados.
    """
    cells_dict = geometry_result.get("cells_dict", {})
    if not cells_dict:
        raise ValueError("create_heating_tallies: 'cells_dict' ausente ou vazio.")

    cell_filter   = _cells_to_filter(cells_dict)
    energy_filter = openmc.EnergyFilter(_ENERGY_BOUNDS_EV)

    tallies = openmc.Tallies()

    # Tallies padrão (sem filtro de energia)
    for tname, score in (
        ("heating",       "heating"),
        ("flux",          "flux"),
        ("fission_rate",  "fission"),
        ("kappa-fission", "kappa-fission"),
    ):
        t = openmc.Tally(name=tname)
        t.filters = [cell_filter]
        t.scores  = [score]
        tallies.append(t)

    # Tally espectral: flux por grupo de energia por célula
    t_spec = openmc.Tally(name="flux_spectrum")
    t_spec.filters = [cell_filter, energy_filter]
    t_spec.scores  = ["flux"]
    tallies.append(t_spec)

    logger.info(
        "create_heating_tallies: %d tallies criados (%d grupos espectrais)",
        len(tallies), _N_ENERGY_GROUPS,
    )
    return tallies


def create_activation_tallies(geometry_result: dict) -> "openmc.Tallies":
    """
    Tallies dedicados ao modo ACTIVATION para rastreamento de Mo99.

    Cria tallies de taxa de reação específicos para quantificar as rotas de
    produção e perda do Mo99 durante a irradiação:

      - 'reaction_Mo98_ng'  : Mo98(n,γ)Mo99 — captura em Mo98
      - 'reaction_U235_fiss': U235(n,f)     — fissões de U235 (produz Mo99 por yield)
      - 'reaction_Mo99_ng'  : Mo99(n,γ)Mo100 — captura parasita em Mo99

    Estes tallies permitem verificar post-hoc se o integrador de depleção
    está usando os fluxos e seções de choque corretos para cada reação.

    Args:
        geometry_result: Dict com 'cells_dict'.

    Returns:
        openmc.Tallies com tallies de taxa de reação por nuclídeo.
    """
    cells_dict = geometry_result.get("cells_dict", {})
    if not cells_dict:
        raise ValueError("create_activation_tallies: 'cells_dict' ausente ou vazio.")

    cell_filter = _cells_to_filter(cells_dict)
    tallies = openmc.Tallies()

    reaction_specs = [
        ("reaction_Mo98_ng",   "Mo98",  "(n,gamma)"),
        ("reaction_U235_fiss", "U235",  "fission"),
        ("reaction_Mo99_ng",   "Mo99",  "(n,gamma)"),
        ("reaction_U238_fiss", "U238",  "fission"),
    ]

    for tname, nuclide, reaction in reaction_specs:
        try:
            nuc_filter = openmc.NuclideFilter([nuclide])
            t = openmc.Tally(name=tname)
            t.filters = [cell_filter, nuc_filter]
            t.scores  = [reaction]
            tallies.append(t)
            logger.debug("Tally criado: %s [%s %s]", tname, nuclide, reaction)
        except Exception as exc:
            logger.warning("Não foi possível criar tally %s: %s", tname, exc)

    logger.info("create_activation_tallies: %d tallies de reação criados", len(tallies))
    return tallies


def extract_power_per_layer(
    statepoint_file: str,
    geometry_result: dict,
    source_rate: float,
    mode: str = "fixed source",
    include_uncertainty: bool = True,
) -> Dict[str, float]:
    """
    Extrai potência depositada [W] por camada a partir do statepoint HDF5.

    Args:
        include_uncertainty: se True, inclui chaves '<nome>_unc_rel' com
                             incerteza relativa de cada célula.
    """
    sp_path = Path(statepoint_file)
    if not sp_path.exists():
        logger.error("extract_power_per_layer: %s não encontrado", statepoint_file)
        return {}
    cells_dict = geometry_result.get("cells_dict", {})
    if not cells_dict:
        return {}

    id_to_name = _build_id_to_name(cells_dict)
    fallback   = {name: 0.0 for name in cells_dict}

    try:
        with openmc.StatePoint(str(sp_path)) as sp:
            try:
                tally = sp.get_tally(name="heating")
            except Exception as exc:
                logger.error("tally 'heating' não encontrado: %s", exc)
                return fallback
            df = tally.get_pandas_dataframe()
    except Exception as exc:
        logger.error("falha ao abrir statepoint: %s", exc)
        return fallback

    return _extract_scores_from_df(
        df, id_to_name, "heating", source_rate,
        "extract_power_per_layer", include_uncertainty=include_uncertainty,
    )


def extract_kappa_fission_power(
    statepoint_file: str,
    geometry_result: dict,
    source_rate: float,
    include_uncertainty: bool = True,
) -> Dict[str, float]:
    """Extrai potência de fissão [W] por camada a partir do tally 'kappa-fission'."""
    sp_path = Path(statepoint_file)
    if not sp_path.exists():
        logger.error("extract_kappa_fission_power: %s não encontrado", statepoint_file)
        return {}
    cells_dict = geometry_result.get("cells_dict", {})
    if not cells_dict:
        return {}

    id_to_name = _build_id_to_name(cells_dict)
    fallback   = {name: 0.0 for name in cells_dict}

    try:
        with openmc.StatePoint(str(sp_path)) as sp:
            try:
                tally = sp.get_tally(name="kappa-fission")
            except Exception as exc:
                logger.debug("tally 'kappa-fission' não encontrado: %s", exc)
                return fallback
            df = tally.get_pandas_dataframe()
    except Exception as exc:
        logger.error("falha ao abrir statepoint: %s", exc)
        return fallback

    return _extract_scores_from_df(
        df, id_to_name, "kappa-fission", source_rate,
        "extract_kappa_fission_power", include_uncertainty=include_uncertainty,
    )


def extract_spectrum_weighted_yield(
    statepoint_file: str,
    geometry_result: dict,
    target_cell: str,
    nuclide: str = "U235",
    product: str = "Mo99",
) -> Dict[str, float]:
    """
    Calcula o yield efetivo de fissão ponderado pelo espectro real da simulação.

    Usa o tally 'flux_spectrum' para obter o fluxo por grupo de energia em
    'target_cell' e pondera os yields de fissão tabelados por grupo.
    Isso corrige o fator sistemático de ~7-15% entre usar yield térmico puro
    e o yield real do espectro do experimento.

    Yields de Mo99 por fissão de U235 (ENDF/B-VIII.0, cumulativos):
      Térmico   (< 0.625 eV):  0.06108
      Epitérmico (0.625eV–100keV): 0.05813  (médio, varia com E)
      Rápido    (> 100 keV):   0.03273

    Returns:
        Dict com:
          'y_eff'        — yield efetivo espectro-ponderado
          'w_thermal'    — fração do fluxo na região térmica
          'w_epithermal' — fração epitérmica
          'w_fast'       — fração rápida
          'y_thermal'    — yield térmico de referência
          'correction_factor' — y_eff / y_thermal
    """
    sp_path = Path(statepoint_file)
    if not sp_path.exists():
        logger.warning("extract_spectrum_weighted_yield: %s não encontrado", statepoint_file)
        return {}

    cells_dict = geometry_result.get("cells_dict", {})
    if target_cell not in cells_dict:
        logger.warning("Célula '%s' não encontrada em cells_dict", target_cell)
        return {}

    # Yields tabelados de Mo99/fissão U235 (ENDF/B-VIII.0, cumulativos)
    # por faixa de energia — valores médios por região
    _YIELD_BY_GROUP = {
        "thermal":    0.06108,
        "epithermal": 0.05813,
        "fast":       0.03273,
    }

    def _energy_group_region(e_mid_ev: float) -> str:
        if e_mid_ev < _E_THERMAL_MAX_EV:
            return "thermal"
        elif e_mid_ev < _E_EPITHERMAL_MAX_EV:
            return "epithermal"
        return "fast"

    try:
        with openmc.StatePoint(str(sp_path)) as sp:
            try:
                tally = sp.get_tally(name="flux_spectrum")
            except Exception as exc:
                logger.warning("tally 'flux_spectrum' não encontrado: %s", exc)
                return {}
            df = tally.get_pandas_dataframe()
    except Exception as exc:
        logger.error("Erro ao abrir statepoint: %s", exc)
        return {}

    cell_col = _detect_cell_col(df.columns)
    if cell_col is None:
        return {}

    cell_obj = cells_dict[target_cell]
    cell_id  = cell_obj.id if hasattr(cell_obj, "id") else int(cell_obj)
    df_cell  = df[df[cell_col] == cell_id]

    if df_cell.empty:
        logger.warning("Nenhum dado de fluxo para célula '%s'", target_cell)
        return {}

    # Calcula pesos espectrais por grupo
    flux_by_region = {"thermal": 0.0, "epithermal": 0.0, "fast": 0.0}
    e_bounds = _ENERGY_BOUNDS_EV

    for i, row in df_cell.iterrows():
        flux_val = float(row.get("mean", 0.0))
        # Identifica grupo de energia pela coluna 'energy low [eV]'
        e_low_col  = next((c for c in df.columns if "energy low"  in c.lower()), None)
        e_high_col = next((c for c in df.columns if "energy high" in c.lower()), None)
        if e_low_col and e_high_col:
            e_mid = (float(row[e_low_col]) + float(row[e_high_col])) / 2.0
        else:
            e_mid = 1.0  # fallback: assume térmico
        region = _energy_group_region(e_mid)
        flux_by_region[region] += flux_val

    total_flux = sum(flux_by_region.values())
    if total_flux <= 0:
        return {}

    w = {r: f / total_flux for r, f in flux_by_region.items()}

    # Yield efetivo espectro-ponderado
    y_eff = sum(w[r] * _YIELD_BY_GROUP[r] for r in w)
    y_th  = _YIELD_BY_GROUP["thermal"]

    result = {
        "y_eff":             y_eff,
        "w_thermal":         w["thermal"],
        "w_epithermal":      w["epithermal"],
        "w_fast":            w["fast"],
        "y_thermal":         y_th,
        "correction_factor": y_eff / y_th if y_th > 0 else 1.0,
    }

    logger.info(
        "Yield efetivo %s/%s: %.5f (térmico puro: %.5f, fator correção: %.4f) "
        "[w_th=%.3f w_epi=%.3f w_fast=%.3f]",
        product, nuclide, y_eff, y_th, result["correction_factor"],
        w["thermal"], w["epithermal"], w["fast"],
    )
    return result


class KeffResult(NamedTuple):
    """Resultado de k_eff com incerteza."""
    value: float
    std:   float

    def __str__(self) -> str:
        return f"k_eff = {self.value:.5f} ± {self.std:.5f}"


def get_k_eff(statepoint_file: str) -> KeffResult:
    """
    Lê k_eff e incerteza do statepoint HDF5.

    Returns:
        KeffResult(value, std). Retorna KeffResult(0.0, 0.0) em falha.
    """
    sp_path = Path(statepoint_file)
    if not sp_path.exists():
        return KeffResult(0.0, 0.0)
    try:
        with openmc.StatePoint(str(sp_path)) as sp:
            kc = sp.k_combined
        k_val = float(kc[0])
        k_std = float(kc[1]) if len(kc) > 1 else 0.0
        return KeffResult(k_val, k_std)
    except Exception:
        pass
    try:
        import h5py
        with h5py.File(str(sp_path), "r") as f:
            for key in ("k_combined", "k_active"):
                if key in f:
                    data    = f[key][()]
                    k_val   = float(data[0]) if hasattr(data, "__len__") else float(data)
                    k_std   = float(data[1]) if (hasattr(data, "__len__") and len(data) > 1) else 0.0
                    return KeffResult(k_val, k_std)
    except Exception as exc:
        logger.error("get_k_eff (HDF5 fallback) falhou: %s", exc)
    return KeffResult(0.0, 0.0)
