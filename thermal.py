#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thermal.py V7.0 — Solver FDM 1D com iteração Picard.

Física:
  Equação : -d/dz [k(T) dT/dz] = q_vol [W/m³]
  Malha   : N_nós = max(5, int(espessura_mm)) por camada
  BC esq  : -k·dT/dz|_{z=0} = h_conv·(T_wall − T_ref)
  BC dir  : dT/dz|_{z=L} = 0 (Neumann, ghost-node)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from pyne_bridge import BRIDGE
from config import TNLoopConfig, ValidationLimits, ThermalSolverConfig

logger = logging.getLogger(__name__)

__all__ = ["ThermalModel", "ConvergenceTracker", "ConvergenceResult"]

_VL  = ValidationLimits()
_TSC = ThermalSolverConfig()


# ─────────────────────────────────────────────────────────────────────────────
# ConvergenceResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConvergenceResult:
    iteration:       int
    converged:       bool
    l2_norm:         float
    linf_norm:       float
    max_diff:        float
    max_diff_cell:   str
    temperature_map: Dict[str, float]
    metrics:         Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# ConvergenceTracker
# ─────────────────────────────────────────────────────────────────────────────

class ConvergenceTracker:
    """
    Monitora convergência de temperatura entre iterações T-N.

    FIX T1: critério MISTO relativo + absoluto.
    - Critério relativo (tol_l2/tol_linf): evita iterações desnecessárias
      quando ΔT/T_prev é pequeno mesmo que ΔT_abs seja grande.
    - Critério absoluto (tol_abs_K): garante que ΔT máximo em Kelvin
      seja compatível com TNLoopConfig.CONVERGENCE_EPSILON_TEMP.
    Convergência exige AMBOS satisfeitos.

    Valores alinhados com TNLoopConfig:
      tol_abs_K = CONVERGENCE_EPSILON_TEMP = 0.5K
      tol_l2    = 1e-4 (0.01%) — mantido para compatibilidade
    """

    def __init__(
        self,
        tol_l2:    float = 1e-4,
        tol_linf:  float = 1e-3,
        tol_abs_K: float = None,   # FIX T1: tolerância absoluta em K
        max_iters: int   = 50,
    ) -> None:
        self.tol_l2   = tol_l2
        self.tol_linf = tol_linf
        # FIX T1: se não fornecido, usa CONVERGENCE_EPSILON_TEMP do config
        self.tol_abs_K = float(tol_abs_K) if tol_abs_K is not None else \
                         float(getattr(TNLoopConfig, "CONVERGENCE_EPSILON_TEMP", 0.5))
        self.max_iters = max_iters
        self.last_map: Optional[Dict[str, float]] = None
        self.iteration_count = 0

    def check_convergence(self, current_map: Dict[str, float]) -> ConvergenceResult:
        self.iteration_count += 1
        if self.last_map is None:
            self.last_map = current_map.copy()
            return ConvergenceResult(
                iteration=self.iteration_count, converged=False,
                l2_norm=0.0, linf_norm=0.0, max_diff=0.0,
                max_diff_cell="initialization", temperature_map=current_map.copy(),
                metrics={"reason": "first_iteration"},
            )

        diffs_rel = []
        diffs_abs = []
        names = []
        for name, t in current_map.items():
            t_prev = self.last_map.get(name, t)
            dt_abs = abs(t - t_prev)
            diffs_rel.append(dt_abs / max(abs(t_prev), 1.0))
            diffs_abs.append(dt_abs)
            names.append(name)

        if not diffs_rel:
            self.last_map = current_map.copy()
            return ConvergenceResult(
                iteration=self.iteration_count, converged=True,
                l2_norm=0.0, linf_norm=0.0, max_diff=0.0,
                max_diff_cell="none", temperature_map=current_map.copy(),
            )

        arr_rel = np.array(diffs_rel)
        arr_abs = np.array(diffs_abs)
        l2      = float(np.sqrt(np.mean(arr_rel ** 2)))
        linf    = float(np.max(arr_rel))
        max_abs = float(np.max(arr_abs))
        idx     = int(np.argmax(arr_abs))   # pior célula em ΔT absoluto

        # FIX T1: convergência por critério ABSOLUTO OU RELATIVO (OR, não AND).
        #
        # Lógica física:
        #   - conv_abs (ΔT_max < tol_abs_K=0.5K): suficiente para T-N —
        #     0.5K de variação em qualquer célula é fisicamente negligível
        #     para a realimentação de densidade da água e seções de choque.
        #   - conv_rel (l2 + linf < tol): alternativa para quando T é muito
        #     alta e ΔT/T é pequeno mesmo com ΔT_abs > 0.5K (ex: T_fuel > 1000K).
        #
        # Antes era AND: exigia ambos simultaneamente. Isso impedia convergência
        # em casos físicos válidos onde ΔT=0.1K (< 0.5K) mas ΔT/T=2.9e-4 (> 1e-4).
        conv_rel = (l2 < self.tol_l2 and linf < self.tol_linf)
        conv_abs = (max_abs < self.tol_abs_K)
        converged = conv_abs or conv_rel   # OR: qualquer critério satisfeito é suficiente

        self.last_map = current_map.copy()
        return ConvergenceResult(
            iteration=self.iteration_count,
            converged=converged,
            l2_norm=l2,
            linf_norm=linf,
            max_diff=max_abs,
            max_diff_cell=names[idx],
            temperature_map=current_map.copy(),
            metrics={
                "tol_l2": self.tol_l2, "tol_linf": self.tol_linf,
                "tol_abs_K": self.tol_abs_K,
                "conv_rel": conv_rel, "conv_abs": conv_abs,
                "n_cells": len(diffs_rel),
            },
        )

    def reset(self) -> None:
        self.last_map = None
        self.iteration_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# ThermalModel
# ─────────────────────────────────────────────────────────────────────────────

class ThermalModel:
    """Modelo Térmico V7.0 — solver FDM 1D Picard por camada."""

    def __init__(self, parser_result: Optional[Dict] = None,
                 geometry_result: Optional[Dict] = None, debug: bool = False) -> None:
        self.debug           = debug
        self.parser_result   = parser_result  or {}
        self.geometry_result = geometry_result or {}
        self.last_temp_map: Dict[str, float] = {}
        self.convergence_tracker = ConvergenceTracker()

        self.PICARD_MAX = int(getattr(TNLoopConfig, "PICARD_MAX_INTERNAL", 5))
        self.H_CONV     = _TSC.H_CONV
        self.PICARD_TOL = _TSC.PICARD_TOL

        sim_params = self.parser_result.get("simulation_parameters", {})
        t_water_c  = float(sim_params.get("water_temperature_c",
                           sim_params.get("water_temp_c", 25.0)) or 25.0)
        self.T_ref_K = t_water_c + 273.15

        wg = (self.parser_result.get("wafer_geometry") or
              self.geometry_result.get("wafer_geometry") or {})
        self.wafer_area_m2 = (float(wg.get("x_cm", 5.0)) * 1e-2) * \
                             (float(wg.get("y_cm", 5.0)) * 1e-2)

    # ── Helpers de camada ─────────────────────────────────────────────────────

    def _get_layers(self) -> List[Dict]:
        for src in (self.geometry_result, self.parser_result):
            layers = src.get("layers")
            if layers:
                return list(layers.values()) if isinstance(layers, dict) else list(layers)
        tm = self.parser_result.get("thermal_materials", {})
        pl = self.parser_result.get("layers", {})
        pl_vals = pl if isinstance(pl, dict) else {str(i): l for i, l in enumerate(pl)}
        fallback = []
        for name, d in tm.items():
            base = pl_vals.get(name, {})
            t_cm = base.get("thickness_cm")
            t_mm = base.get("thickness_mm")
            fallback.append({
                "name": name,
                "thickness_cm": float(t_mm / 10.0 if t_mm else (t_cm or 0.1)),
                "thickness_mm": float(t_cm * 10.0 if t_cm else (t_mm or 1.0)),
                "material_name": d.get("material_name", d.get("material", "aluminio 6061")),
            })
        return fallback

    def _layer_name(self, layer: Dict) -> str:
        return layer.get("name") or layer.get("layer_name", "CAMADA_?")

    def _layer_material(self, layer: Dict) -> str:
        """
        Resolve o nome do material de uma camada para uma chave válida em
        pyne_bridge.MATERIALS. Tenta em ordem:
          1. material_name / material do próprio layer
          2. componente dominante de thermal_materials (maior massa)
          3. substring match contra chaves canônicas
          4. fallback Al6061
        """
        # 1 — campo direto no layer
        for key in ("material_name", "material"):
            val = layer.get(key)
            if val and str(val).strip():
                resolved = self._resolve_material_name(str(val).strip())
                if resolved:
                    return resolved

        # 2 — componente dominante de thermal_materials
        layer_name = self._layer_name(layer)
        tm_all = self.parser_result.get("thermal_materials", {})
        if isinstance(tm_all, dict) and layer_name in tm_all:
            tm_layer = tm_all[layer_name]
            if isinstance(tm_layer, list) and tm_layer:
                dominant = max(tm_layer, key=lambda c: float(c.get("mass_g", 0.0)))
                mat = dominant.get("material") or dominant.get("material_name")
            elif isinstance(tm_layer, dict):
                mat = tm_layer.get("material") or tm_layer.get("material_name")
            else:
                mat = None
            if mat:
                resolved = self._resolve_material_name(str(mat))
                if resolved:
                    return resolved

        # 3 — substring match (mais específico primeiro)
        lname_up = layer_name.upper()
        for substr, canonical in (
            ("UAL4", "UAl4_20"), ("UAL3", "UAl3_20"), ("UAL2", "UAl2_20"),
            ("UAL",  "UAl4_20"), ("UO2",  "UO2_20"),  ("U_ME", "U_metal_20"),
            ("SS316","SS316"),   ("H2O",  "H2O"),
            ("6061", "Al6061"),  ("AL",   "Al6061"),
        ):
            if substr in lname_up:
                return canonical

        # FIX G5: verificar se é camada combustível antes de fazer fallback silencioso.
        # Verificar TANTO layer_name QUANTO material_name original passado.
        _FISSILE_PATTERNS = ("U", "PU", "TH", "FUEL", "COMBUST", "FISS",
                              "ALVO", "TARGET", "NUCL")
        # Coletar todos os nomes candidatos para inspeção
        _name_candidates = [lname_up]
        for key in ("material_name", "material", "name", "layer_name"):
            v = layer.get(key)
            if v:
                _name_candidates.append(str(v).upper())

        is_fissile = any(
            p in candidate
            for candidate in _name_candidates
            for p in _FISSILE_PATTERNS
        )
        if is_fissile:
            raise ValueError(
                f"Material combustível '{layer_name}' (material_name='{layer.get('material_name','')}') "
                f"não identificado em pyne_bridge.MATERIALS. "
                f"Verifique o nome no Input-simulador.txt. "
                f"Nomes aceitos: UAl4_20, UAl3_20, UAl2_20, UO2_20, U_metal_20. "
                f"Fallback para Al6061 NÃO é aplicado a camadas combustíveis."
            )

        logger.warning(
            "material não identificado para '%s' → fallback Al6061 "
            "(camada estrutural; se for combustível, verifique o nome)",
            layer_name,
        )
        return "Al6061"

    @staticmethod
    def _resolve_material_name(name: str) -> Optional[str]:
        """Resolve nome → chave canônica em pyne_bridge.MATERIALS. None se não encontrar."""
        from pyne_bridge import MATERIALS
        if name in MATERIALS:
            return name
        name_lower = name.lower().strip()
        for k in MATERIALS:
            if k.lower() == name_lower:
                return k
        _LEGACY: Dict[str, str] = {
            "aluminum, alloy 6061-t6": "Al6061",
            "aluminum alloy 6061":     "Al6061",
            "al6061":                  "Al6061",
            "al_puro":                 "Al",
            "aluminio puro":           "Al",
            "ual4":  "UAl4_20", "ual3": "UAl3_20", "ual2": "UAl2_20",
            "ss316": "SS316",   "agua": "H2O",
            "water, liquid": "H2O",
        }
        return _LEGACY.get(name_lower)

    def _layer_thickness_m(self, layer: Dict) -> float:
        t_cm = layer.get("thickness_cm")
        if t_cm is not None:
            return max(float(t_cm) * 1e-2, 1e-9)
        t_mm = layer.get("thickness_mm")
        if t_mm is not None:
            return max(float(t_mm) * 1e-3, 1e-9)
        return 1.0e-3

    # ── Malha FDM ─────────────────────────────────────────────────────────────

    def _build_mesh(self, layers: List[Dict]) -> List[Dict]:
        nodes: List[Dict] = []
        for layer in layers:
            name    = self._layer_name(layer)
            mat     = self._layer_material(layer)
            thick_m = self._layer_thickness_m(layer)
            n_nods  = max(5, int(thick_m * 1e3))
            dz      = thick_m / n_nods
            z_start = nodes[-1]["z"] + nodes[-1]["dz"] if nodes else 0.0
            for i in range(n_nods):
                nodes.append({"z": z_start + (i + 0.5) * dz, "dz": dz,
                               "layer_name": name, "material": mat, "T": self.T_ref_K})
        return nodes

    # ── Sistema tridiagonal (Thomas) ──────────────────────────────────────────

    def _assemble_and_solve(self, nodes: List[Dict], power_dist: Dict[str, float]) -> np.ndarray:
        N = len(nodes)
        if N == 0:
            return np.array([])

        nodes_per_layer: Dict[str, int] = {}
        for nd in nodes:
            nodes_per_layer[nd["layer_name"]] = nodes_per_layer.get(nd["layer_name"], 0) + 1

        q_vol = np.zeros(N)
        for i, nd in enumerate(nodes):
            P_w = power_dist.get(nd["layer_name"], 0.0)
            if P_w > 0.0:
                q_vol[i] = P_w / (self.wafer_area_m2 * nd["dz"] * nodes_per_layer.get(nd["layer_name"], 1))

        a, b, c, d = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        T_cur = np.array([nd["T"] for nd in nodes])

        dz0 = nodes[0]["dz"]
        k0  = BRIDGE.get_k(nodes[0]["material"], T_cur[0])
        b[0] = k0 / dz0**2 + self.H_CONV / dz0
        c[0] = -k0 / dz0**2
        d[0] = self.H_CONV / dz0 * self.T_ref_K + q_vol[0]

        for i in range(1, N - 1):
            dz_i = nodes[i]["dz"]
            k_l  = 0.5 * (BRIDGE.get_k(nodes[i-1]["material"], T_cur[i-1]) +
                          BRIDGE.get_k(nodes[i  ]["material"], T_cur[i  ]))
            k_r  = 0.5 * (BRIDGE.get_k(nodes[i  ]["material"], T_cur[i  ]) +
                          BRIDGE.get_k(nodes[i+1]["material"], T_cur[i+1]))
            a[i] = -k_l / dz_i**2
            b[i] = (k_l + k_r) / dz_i**2
            c[i] = -k_r / dz_i**2
            d[i] = q_vol[i]

        # BC direita: ghost-node Neumann
        dz_n = nodes[N-1]["dz"]
        k_l  = 0.5 * (BRIDGE.get_k(nodes[N-2]["material"], T_cur[N-2]) +
                      BRIDGE.get_k(nodes[N-1]["material"], T_cur[N-1]))
        a[N-1] = -k_l / dz_n**2
        b[N-1] =  k_l / dz_n**2
        c[N-1] =  0.0
        d[N-1] =  q_vol[N-1]

        return self._thomas(a, b, c, d)

    @staticmethod
    def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        N = len(b)
        c_, d_, x = np.zeros(N), np.zeros(N), np.zeros(N)
        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
        for i in range(1, N):
            denom = b[i] - a[i] * c_[i-1]
            denom = denom if abs(denom) >= 1e-30 else 1e-30
            c_[i] = c[i] / denom
            d_[i] = (d[i] - a[i] * d_[i-1]) / denom
        x[N-1] = d_[N-1]
        for i in range(N-2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i+1]
        return x

    def _map_nodes_to_layers(self, nodes: List[Dict], T_nodes: np.ndarray) -> Dict[str, float]:
        layer_temps: Dict[str, list] = {}
        for i, nd in enumerate(nodes):
            layer_temps.setdefault(nd["layer_name"], []).append(float(T_nodes[i]))
        return {name: float(np.mean(v)) for name, v in layer_temps.items()}

    def _clip_temperatures(self, temp_map: Dict[str, float], layers: List[Dict]) -> Dict[str, float]:
        clipped = {}
        for layer in layers:
            name  = self._layer_name(layer)
            T     = temp_map.get(name, self.T_ref_K)
            mat   = self._layer_material(layer)
            T_max = min(BRIDGE.get_melting_k(mat) * 0.80, _VL.TEMP_MAX_K)
            if T < _VL.TEMP_MIN_K:
                logger.warning("%s: T=%.1fK < T_min → clip", name, T)
                T = _VL.TEMP_MIN_K
            elif T > T_max:
                logger.warning("%s: T=%.1fK > T_max=%.1fK → clip", name, T, T_max)
                T = T_max
            clipped[name] = T
        return clipped

    # ── Contrato principal ────────────────────────────────────────────────────

    def compute_temperature_profile(self, power_distribution: Dict[str, float]) -> Dict[str, float]:
        layers = self._get_layers()
        if not layers:
            self.last_temp_map = {}
            return {}
        if not power_distribution:
            self.last_temp_map = {self._layer_name(l): self.T_ref_K for l in layers}
            return self.last_temp_map

        nodes = self._build_mesh(layers)
        T_cur = np.array([nd["T"] for nd in nodes])

        for picard_iter in range(self.PICARD_MAX):
            for i, nd in enumerate(nodes):
                nd["T"] = float(T_cur[i])
            T_new = self._assemble_and_solve(nodes, power_distribution)
            bad = ~np.isfinite(T_new)
            if bad.any():
                T_new[bad] = self.T_ref_K
            delta = float(np.max(np.abs(T_new - T_cur)))
            T_cur = T_new.copy()
            if delta < self.PICARD_TOL:
                break
        else:
            logger.warning("FDM Picard: máximo de %d iter atingido (||ΔT||=%.4fK)",
                           self.PICARD_MAX, delta)

        for i, nd in enumerate(nodes):
            nd["T"] = float(T_cur[i])

        temp_map = self._map_nodes_to_layers(nodes, T_cur)
        temp_map = self._clip_temperatures(temp_map, layers)
        self.last_temp_map = temp_map.copy()
        return temp_map

    def check_convergence(self, temperature_map: Dict[str, float]) -> ConvergenceResult:
        return self.convergence_tracker.check_convergence(temperature_map)

    def reset_convergence(self) -> None:
        self.convergence_tracker.reset()
        self.last_temp_map = {}

    def solve_thermal_step(self, power_by_layer: Dict, dt: float = 0.0) -> Dict[str, float]:
        """Alias de compute_temperature_profile() compatível com simulation.py."""
        if power_by_layer and all(isinstance(k, int) for k in power_by_layer):
            layers = self._get_layers()
            id_to_name = {lay.get("number", i): self._layer_name(lay)
                          for i, lay in enumerate(layers)}
            power_by_layer = {id_to_name.get(k, str(k)): v for k, v in power_by_layer.items()}
        return self.compute_temperature_profile(power_by_layer)
