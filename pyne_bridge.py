#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyne_bridge.py V4.0 — Ponte central: Material Library · Correlações Térmicas
                       Interface OpenMC · Cálculos Nucleares

Nomes de materiais aceitos: exatamente os da tabela MATERIALS abaixo.
Sem sistema de aliases — use os nomes canônicos no Input-fases.txt.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyne.material import Material
from pyne.material_library import MaterialLibrary
from pyne import nucname
from pyne import data as pynedata

import openmc

from config import ValidationLimits, TNLoopConfig, PhysicsConstants

_logger = logging.getLogger(__name__)
_VLIMITS = ValidationLimits()

try:
    import pyne as _pyne_pkg
    NUC_DATA = str(_pyne_pkg.nuc_data)
except Exception:
    NUC_DATA = str(Path.home() / ".local/lib/python3.12/site-packages/pyne/nuc_data.h5")


# ─────────────────────────────────────────────────────────────────────────────
# TABELA DE MATERIAIS
# Chave = nome canônico (use exatamente este no Input-fases.txt)
# Valor = (composição em frações mássicas, densidade g/cm³)
#
# Frações de enriquecimento 20% U-235:
#   U235 frac = enrichment * u_frac_in_compound
#   U238 frac = (1 - enrichment) * u_frac_in_compound
#
# UAl2: U=47.5% U+2Al em massa  → enr 20% → U235=9.50% U238=38.00% Al27=52.50%
# UAl3: U=39.7% U+3Al em massa  → enr 20% → U235=7.94% U238=31.76% Al27=60.30%
# UAl4: U=33.6% U+4Al em massa  → enr 20% → U235=6.72% U238=26.88% Al27=66.40%
# ─────────────────────────────────────────────────────────────────────────────

MATERIALS: Dict[str, Tuple[Dict[str, float], float]] = {
    # ── Combustíveis ──────────────────────────────────────────────────────────
    "U_metal_20":    ({"U235": 0.200,   "U238": 0.800},                            19.1),
    "U_metal_93":    ({"U235": 0.930,   "U238": 0.070},                            19.1),
    "UO2_20":        ({"U235": 0.1763,  "U238": 0.7052, "O16":  0.1185},           10.97),
    "UO2_93":        ({"U235": 0.8162,  "U238": 0.0638, "O16":  0.1200},           10.97),
    "UAl2_20":       ({"U235": 0.0950,  "U238": 0.3800, "Al27": 0.5250},           8.1),
    "UAl3_20":       ({"U235": 0.0794,  "U238": 0.3176, "Al27": 0.6030},           6.8),
    "UAl4_20":       ({"U235": 0.0672,  "U238": 0.2688, "Al27": 0.6640},           6.1),
    # ── Alumínio ──────────────────────────────────────────────────────────────
    "Al":            ({"Al27": 1.0},                                                2.70),
    "Al6061":        ({"Al27": 0.9790,  "Mg24": 0.0100, "Si28": 0.0060,
                       "Cu63": 0.0030,  "Cr52": 0.0020},                           2.70),
    # ── Outros metais ─────────────────────────────────────────────────────────
    "Zn":            ({"Zn64": 0.4863,  "Zn66": 0.2790, "Zn67": 0.0410,
                       "Zn68": 0.1875,  "Zn70": 0.0062},                           7.133),
    "SS316":         ({"Fe56": 0.6550,  "Ni58": 0.1200, "Cr52": 0.1700,
                       "Mo96": 0.0250,  "Mn55": 0.0200, "Si28": 0.0100},           8.0),
    # ── Moderadores / refrigerantes ───────────────────────────────────────────
    "H2O":           ({"H1":  0.1119,   "O16":  0.8881},                           0.997),
    # ── Orgânicos / blindagem ─────────────────────────────────────────────────
    "oleo_mineral":  ({"H1":  0.1380,   "C12":  0.8620},                           0.87),
    "parafina":      ({"H1":  0.1484,   "C12":  0.8516},                           0.93),
}

# ─────────────────────────────────────────────────────────────────────────────
# Correlações termofísicas  {nome: {k, cp, rho, melting_K, doppler_pcm_K}}
# k  [W/m·K]  cp [J/kg·K]  rho [g/cm³]
# ─────────────────────────────────────────────────────────────────────────────

_THERMAL_CORR: Dict = {
    "U_metal_20": {
        "k":             lambda T: 21.7  + 0.0153 * T,
        "cp":            lambda T: 116.0,
        "rho":           lambda T: 19.1,
        "melting_K":     1405.0,
        "doppler_pcm_K": -2.8,
    },
    "UO2_20": {
        "k":             lambda T: 3.52  / (1.0 + 2.5e-4 * T) + 6.19e-11 * T**3,
        "cp":            lambda T: 235.0 + 0.127 * T,
        "rho":           lambda T: 10.97 * (1.0  - 1.0e-5 * (T - 293.0)),
        "melting_K":     3113.0,
        "doppler_pcm_K": -3.2,
    },
    "UAl2_20": {
        "k":             lambda T: 25.0  + 0.02  * (T - 293.0),
        "cp":            lambda T: 520.0,
        "rho":           lambda T: 8.1,
        "melting_K":     1623.0,
        "doppler_pcm_K": -2.2,
    },
    "UAl3_20": {
        "k":             lambda T: 22.0  + 0.02  * (T - 293.0),
        "cp":            lambda T: 530.0,
        "rho":           lambda T: 6.8,
        "melting_K":     1583.0,
        "doppler_pcm_K": -2.1,
    },
    "UAl4_20": {
        "k":             lambda T: 19.0  + 0.02  * (T - 293.0),
        "cp":            lambda T: 540.0,
        "rho":           lambda T: 6.1,
        "melting_K":     1560.0,
        "doppler_pcm_K": -2.0,
    },
    "Al": {
        "k":             lambda T: 237.0 - 0.05  * (T - 293.0),
        "cp":            lambda T: 900.0 + 0.40  * (T - 293.0),
        "rho":           lambda T: 2.70  * (1.0  - 7.0e-6 * (T - 293.0)),
        "melting_K":     933.0,
        "doppler_pcm_K": 0.0,
    },
    "Al6061": {
        "k":             lambda T: 167.0 + 0.04  * (T - 293.0),
        "cp":            lambda T: 896.0 + 0.50  * (T - 293.0),
        "rho":           lambda T: 2.70  * (1.0  - 7.2e-6 * (T - 293.0)),
        "melting_K":     855.0,
        "doppler_pcm_K": 0.0,
    },
    "SS316": {
        "k":             lambda T: 14.6  + 1.27e-2 * (T - 293.0),
        "cp":            lambda T: 500.0 + 0.20    * (T - 293.0),
        "rho":           lambda T: 8.0,
        "melting_K":     1700.0,
        "doppler_pcm_K": 0.0,
    },
    "H2O": {
        "k":             lambda T: 0.600 + 2.0e-3  * (T - 293.0),
        "cp":            lambda T: 4182.0,
        "rho":           lambda T: max(0.1, 1.0 - 3.0e-4 * (T - 293.0)),
        "melting_K":     373.0,
        "doppler_pcm_K": 0.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PyNEBridge
# ─────────────────────────────────────────────────────────────────────────────

class PyNEBridge:
    """Ponte central: Material Library · Correlações Térmicas · Interface OpenMC"""

    _EV_TO_J: float = PhysicsConstants.EV_TO_J
    _N_A:     float = PhysicsConstants.N_A

    def __init__(self) -> None:
        self.lib: MaterialLibrary = self._build_library()

    # ── Seção A: Material Library ─────────────────────────────────────────────

    def _build_library(self) -> MaterialLibrary:
        lib = MaterialLibrary()
        for name, (comp, density) in MATERIALS.items():
            mat = Material(comp, density=density)
            mat.metadata["name"] = name
            lib[name] = mat
        return lib

    def _get(self, name: str) -> Material:
        key = name.encode() if isinstance(name, str) else name
        if name in self.lib:
            return self.lib[name]
        if key in self.lib:
            return self.lib[key]
        available = sorted(str(k.decode() if isinstance(k, bytes) else k)
                           for k in self.lib.keys())
        raise KeyError(
            f"Material '{name}' não encontrado na biblioteca.\n"
            f"Nomes aceitos: {available}"
        )

    def list_materials(self) -> List[str]:
        return sorted(str(k.decode() if isinstance(k, bytes) else k)
                      for k in self.lib.keys())

    # ── Seção B: Correlações Térmicas ─────────────────────────────────────────

    def _get_corr(self, name: str) -> Dict:
        if name in _THERMAL_CORR:
            return _THERMAL_CORR[name]
        raise KeyError(
            f"Correlação térmica não encontrada para '{name}'. "
            f"Disponíveis: {list(_THERMAL_CORR.keys())}"
        )

    def get_k(self, name: str, TK: float) -> float:
        return self._get_corr(name)["k"](TK)

    def get_cp(self, name: str, TK: float) -> float:
        return self._get_corr(name)["cp"](TK)

    def get_rho(self, name: str, TK: float) -> float:
        return self._get_corr(name)["rho"](TK)

    def get_melting_k(self, name: str) -> float:
        return self._get_corr(name)["melting_K"]

    def get_doppler_pcmk(self, name: str) -> float:
        return self._get_corr(name)["doppler_pcm_K"]

    # ── Seção C: Interface OpenMC ─────────────────────────────────────────────

    def to_openmc_material(self, name: str, density: Optional[float] = None,
                            temperature_K: Optional[float] = None,
                            T_K: Optional[float] = None) -> openmc.Material:
        if temperature_K is None and T_K is not None:
            temperature_K = T_K
        pyne_mat = self._get(name)
        mat  = openmc.Material(name=name)
        # FIX B3: usar densidade T-dependente quando temperatura fornecida e correlação disponível
        if density is not None:
            dens = density
        elif temperature_K is not None and name in _THERMAL_CORR:
            dens = float(_THERMAL_CORR[name]["rho"](temperature_K))
            _logger.debug("to_openmc_material(%s): densidade T-dependente %.4f g/cm³ @ %.1fK",
                          name, dens, temperature_K)
        else:
            dens = abs(float(pyne_mat.density))
        if dens <= 0.0:
            raise ValueError(f"Densidade inválida ({dens}) para '{name}'")
        for nuc_id, frac in pyne_mat.comp.items():
            if frac <= 0.0:
                continue
            try:
                mat.add_nuclide(nucname.name(nuc_id), float(frac), "wo")
            except Exception as exc:
                _logger.warning("add_nuclide(%s) falhou: %s", nuc_id, exc)
        mat.set_density("g/cm3", dens)
        mat.depletable = True
        if temperature_K is not None:
            mat.temperature = float(temperature_K)
        return mat

    # ── Seção D: Cálculos Nucleares ───────────────────────────────────────────

    def heating_eV_to_watts(self, heating_eV_per_src: float, sourcerate_ns: float) -> float:
        return heating_eV_per_src * self._EV_TO_J * sourcerate_ns

    def calculate_isotope_production(self, nucname_str: str, total_fissions: float,
                                      fission_yield: float,
                                      irradiation_times: List[float]) -> Dict:
        lam       = float(pynedata.decay_const(nucname_str))
        A         = float(pynedata.atomic_mass(nucname_str))
        prod_rate = total_fissions * fission_yield
        results   = {}
        for t in irradiation_times:
            N = (prod_rate / lam) * (1.0 - math.exp(-lam * t)) if lam > 0 else prod_rate * t
            results[f"t={t:.1f}s"] = {"atoms": N, "mass_g": N * A / self._N_A}
        return results

    # ── Seção E: Utilitários ──────────────────────────────────────────────────

    def apply_underrelaxation(self, T_new: float, T_old: float,
                               alpha: Optional[float] = None) -> float:
        a = alpha if alpha is not None else TNLoopConfig.RELAXATION_FACTOR
        return a * T_new + (1.0 - a) * T_old

    apply_under_relaxation = apply_underrelaxation  # alias backward compat

    def validate_temperature(self, matname: str, TK: float) -> bool:
        if TK < _VLIMITS.TEMP_MIN_K:
            _logger.warning("%s: T=%.1fK < TEMP_MIN_K", matname, TK)
            return False
        if TK > _VLIMITS.TEMP_MAX_K:
            _logger.warning("%s: T=%.1fK > TEMP_MAX_K", matname, TK)
            return False
        try:
            if TK >= self.get_melting_k(matname):
                _logger.warning("%s: T=%.1fK >= T_fusao=%.1fK!", matname, TK,
                                self.get_melting_k(matname))
                return False
        except KeyError:
            pass
        return True

    # ── Tallies helpers ───────────────────────────────────────────────────────

    def create_heating_tallies(self, geometry_result: Dict) -> openmc.Tallies:
        cells_dict = geometry_result.get("cells_dict", {})
        tallies = openmc.Tallies()
        # FIX B1: incluir kappa-fission (consistente com tallies.py V304)
        for tname, score in (("heating", "heating"), ("flux", "flux"),
                              ("fission_rate", "fission"), ("kappa-fission", "kappa-fission")):
            t = openmc.Tally(name=tname)
            if cells_dict:
                t.filters = [openmc.CellFilter(list(cells_dict.values()))]
            t.scores = [score]
            tallies.append(t)
        return tallies

    def extract_power_per_layer(self, statepoint_file: str, geometry_result: Dict,
                                 sourcerate_ns: float,
                                 mode: str = "activation") -> Dict[str, float]:
        cells_dict  = geometry_result.get("cells_dict", {})
        power_dist: Dict[str, float] = {n: 0.0 for n in cells_dict}
        # FIX B2: verificar existência antes de abrir
        if not Path(statepoint_file).exists():
            _logger.error("extract_power_per_layer: arquivo não encontrado: %s", statepoint_file)
            return power_dist
        try:
            with openmc.StatePoint(statepoint_file) as sp:
                try:
                    tally = sp.get_tally(name="heating")
                except Exception as exc:
                    _logger.warning("Tally 'heating' não encontrado: %s", exc)
                    return power_dist
                df = tally.get_pandas_dataframe()
                for cell_name, cell_id in cells_dict.items():
                    try:
                        cid = cell_id.id if hasattr(cell_id, "id") else int(cell_id)
                        col = "cell id" if "cell id" in df.columns else "cell"
                        row = df[df[col] == cid]
                        if not row.empty:
                            power_dist[cell_name] = self.heating_eV_to_watts(
                                float(row["mean"].sum()), sourcerate_ns
                            )
                    except Exception as exc:
                        _logger.debug("heating celula '%s': %s", cell_name, exc)
        except Exception as exc:
            _logger.error("extract_power_per_layer: falha ao abrir statepoint: %s", exc)
        return power_dist

    def get_keff(self, statepoint_file: str) -> Tuple[float, float]:
        with openmc.StatePoint(statepoint_file) as sp:
            for attr in ("k_combined", "keff", "k_active"):
                val = getattr(sp, attr, None)
                if val is not None:
                    try:
                        return float(val[0]), float(val[1]) if len(val) > 1 else 0.0
                    except (TypeError, IndexError):
                        try:
                            return float(val.nominal_value), float(val.std_dev)
                        except Exception:
                            pass
        _logger.warning("get_keff: nenhum atributo k encontrado em %s", statepoint_file)
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
BRIDGE = PyNEBridge()

__all__ = ["BRIDGE", "PyNEBridge", "MATERIALS"]
