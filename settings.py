#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
settings.py V225 — Cria openmc.Settings e resolve bibliotecas de dados nucleares.

CHANGELOG V225 vs V224:
  MELHORIA PRINCIPAL — DepletionAutoTuner integrado: DT_H_DEPLETION e
    DEPLETION_INTEGRATOR são escolhidos automaticamente a partir do DT_H
    do usuário. O usuário não precisa conhecer esses parâmetros internos.

    Tabela de decisão automática (limiares em DepletionAutoTuner, config.py):
      DT_H ≤  6h  → DT_dep = DT/2,  integrador = celi
      DT_H ≤ 12h  → DT_dep = DT/2,  integrador = celi   ← produção padrão
      DT_H ≤ 24h  → DT_dep = DT/4,  integrador = celi
      DT_H >  24h → DT_dep = DT/6,  integrador = si_celi

    Para DT_H=12h (produção): DT_dep=6h, CELI, 2 sub-passos por output.
    Override explícito via DT_H_DEPLETION ou DEPLETION_INTEGRATOR no input.

  FIX: bloco de alerta de passo temporal removido — coberto pelo AutoTuner.
  FIX: leitura de dt_h_depletion consolidada no AutoTuner.
  NOVO: depletion_params['auto_tune_band'] e ['n_substeps'] no dict resultado.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import openmc
    _OPENMC_OK = True
except ImportError:
    _OPENMC_OK = False

from config import (
    ValidationLimits, NuclearDataPaths, ChainDataProxy,
    SimulationDefaults, PhysicsConstants, SimulationModes,
    DepletionAutoTuner,
)

_VL  = ValidationLimits()
_SD  = SimulationDefaults
_PC  = PhysicsConstants
logger = logging.getLogger(__name__)

# Alias público mantido para backward compat
_ChainDataProxy = ChainDataProxy

# t½ Mo99 = 65.94 h — usado para alertas de passo temporal
_T_HALF_MO99_H: float = 65.94
# Fração de t½ que define o limite de alerta para DT_H_DEPLETION
_DT_SAFEGUARD_FRACTION: float = 1.0 / 3.0  # Δt ≤ t½/3 recomendado


# ─────────────────────────────────────────────────────────────────────────────
# LibraryHierarchy
# ─────────────────────────────────────────────────────────────────────────────

class LibraryHierarchy:
    """Descoberta e validação de bibliotecas de cross-sections."""

    def __init__(self) -> None:
        self._available: List[Tuple[str, Path]] = []

    def discover(self) -> Tuple[Optional[str], Optional[Path]]:
        seen: set = set()
        for name, path in NuclearDataPaths.XS_CANDIDATES:
            if name in seen:
                continue
            seen.add(name)
            if path.exists():
                self._available.append((name, path))
                logger.info("✓ %s → %s", name, path)
            else:
                logger.warning("✗ %s não encontrado: %s", name, path)

        if not self._available:
            env_xs = os.environ.get("OPENMC_CROSS_SECTIONS")
            if env_xs and Path(env_xs).exists():
                self._available.append(("env:OPENMC_CROSS_SECTIONS", Path(env_xs)))
            else:
                logger.error("Nenhuma biblioteca de cross-sections encontrada!")
                return None, None

        name, path = self._available[0]
        logger.info("Biblioteca primária: %s", name)
        return name, path

    @property
    def available(self) -> List[Tuple[str, Path]]:
        return list(self._available)


# ─────────────────────────────────────────────────────────────────────────────
# SettingsBuilder
# ─────────────────────────────────────────────────────────────────────────────

class SettingsBuilder:
    """Constrói openmc.Settings e resolve todos os dados nucleares."""

    VERSION = "V225"

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    @staticmethod
    def _norm(parser_data: dict, *keys, default=None):
        for k in keys:
            v = parser_data.get(k) or parser_data.get("simulation_parameters", {}).get(k)
            if v is not None:
                return v
        return default

    def _find_chain(self, parser_data: dict) -> Optional[Path]:
        inp = self._norm(parser_data, "chain_file", "chainfile", "CHAIN_FILE")
        if inp:
            for c in (Path(inp), Path.home() / "nuclear_data" / inp, Path.cwd() / inp):
                if c.exists():
                    logger.info("✓ Chain file (input): %s", c)
                    return c
            logger.warning("Chain '%s' declarado no input mas não encontrado", inp)
        for c in NuclearDataPaths.CHAIN_CANDIDATES:
            if c.exists():
                logger.info("✓ Chain file (auto): %s", c)
                return c
        logger.error("Chain file não encontrado. Defina CHAIN_FILE no input.")
        return None

    @staticmethod
    def _build_timesteps(dt_h: float, total_h: float) -> List[float]:
        """
        FIX V224: retorna lista de DURAÇÕES em SEGUNDOS para openmc.deplete.

        A versão anterior (V223) retornava n+1 pontos temporais em horas
        [0, Δt, 2Δt, ..., T], o que causava dois problemas:
          1. Se maestro usava ts[1:] como durações, o último elemento valia T
             (e.g. 168h) em vez de Δt (12h) — super-depleção no último passo.
          2. Unidades em horas em vez de segundos (OpenMC espera segundos).

        Esta versão retorna n durações uniformes em segundos, ajustando o
        último passo para absorver eventuais resíduos de arredondamento.

        Args:
            dt_h:    duração de cada passo [h]
            total_h: duração total da irradiação [h]

        Returns:
            Lista de n floats em segundos. sum(result)/3600 == total_h.
        """
        n = max(1, round(total_h / dt_h))
        # n-1 passos completos + 1 passo residual (absorve erro de arredondamento)
        remainder_h = total_h - (n - 1) * dt_h
        steps_h = [dt_h] * (n - 1) + [remainder_h]

        # Verificação de sanidade: residual não pode ser negativo.
        # Pode ser ligeiramente > dt_h quando total_h não é múltiplo exato de dt_h
        # e round() arredonda para baixo (ex: 170h / 12h → 14 passos, último = 14h).
        # Isso é fisicamente correto; apenas avisamos se > 1.5×dt_h.
        if remainder_h < 0 or remainder_h > dt_h * 1.5:
            logger.warning(
                "_build_timesteps: residual inesperado: n=%d, dt_h=%s, "
                "total_h=%s, remainder=%s h",
                n, dt_h, total_h, remainder_h,
            )

        steps_s = [s * _PC.SECONDS_PER_HOUR for s in steps_h]
        logger.debug(
            "_build_timesteps: %d passos × %.2fh (último=%.2fh), total=%.4fh",
            n, dt_h, remainder_h, sum(steps_h),
        )
        return steps_s

    @staticmethod
    def _build_timesteps_depletion(
        dt_depletion_h: float,
        dt_output_h: float,
        total_h: float,
    ) -> Tuple[List[float], List[int]]:
        """
        Retorna timesteps finos para o integrador e índices dos pontos de output.

        Divide cada intervalo de output (dt_output_h) em sub-passos de
        dt_depletion_h para maior precisão do integrador de depleção.
        Útil quando dt_output_h > t½(nuclídeo) / 3.

        Args:
            dt_depletion_h: passo interno do integrador [h]
            dt_output_h:    passo de saída de inventário [h]
            total_h:        tempo total [h]

        Returns:
            (timesteps_s, output_indices):
              timesteps_s    — durações em segundos para openmc.deplete
              output_indices — índices (base-0) dos passos que coincidem com
                               pontos de output (para salvar inventário)
        """
        n_output = max(1, round(total_h / dt_output_h))
        # sub-passos por intervalo de output
        n_sub = max(1, round(dt_output_h / dt_depletion_h))

        timesteps_s: List[float] = []
        output_indices: List[int] = []
        dt_sub_s = (dt_output_h / n_sub) * _PC.SECONDS_PER_HOUR

        for i in range(n_output):
            is_last_output = (i == n_output - 1)
            for j in range(n_sub):
                is_last_sub = (j == n_sub - 1)
                # Último sub-passo do último output: ajusta residual
                if is_last_output and is_last_sub:
                    already_s = sum(timesteps_s)
                    remaining_s = total_h * _PC.SECONDS_PER_HOUR - already_s
                    timesteps_s.append(max(remaining_s, dt_sub_s * 0.01))
                else:
                    timesteps_s.append(dt_sub_s)
                if is_last_sub:
                    output_indices.append(len(timesteps_s) - 1)

        logger.debug(
            "_build_timesteps_depletion: %d passos totais, %d pontos de output "
            "(n_sub=%d × n_output=%d)",
            len(timesteps_s), len(output_indices), n_sub, n_output,
        )
        return timesteps_s, output_indices

    @staticmethod
    def _output_times_h(dt_h: float, total_h: float) -> List[float]:
        """Posições temporais absolutas em horas para output de inventário."""
        n = max(1, round(total_h / dt_h))
        times = [round(i * dt_h, 10) for i in range(n + 1)]
        if abs(times[-1] - total_h) > 1e-6:
            times[-1] = total_h
        return times

    def build(self, parser_data: dict, geometry_result: dict) -> dict:
        sp   = parser_data.get("simulation_parameters", parser_data)
        meta = parser_data.get("metadata", {})

        def _get(*keys, default=None, cast=None):
            for k in keys:
                for src in (sp, meta, parser_data):
                    v = src.get(k)
                    if v is not None:
                        return cast(v) if cast else v
            return default

        # ── Parâmetros temporais ──────────────────────────────────────────
        dt_output_h = _get("dt_h", "DTH",
                            default=_SD.DT_H_OUTPUT, cast=float)
        total_h     = _get("total_time_h", "TEMPO_TOTAL_H",
                            default=_SD.TOTAL_TIME_H, cast=float)
        cooling_h   = _get("cooling_time_h",
                            default=_SD.COOLING_TIME_H, cast=float)

        # ── Parâmetros MC ─────────────────────────────────────────────────
        particles = _get("nparticles", "neutrons_por_passo",
                          default=_SD.NPARTICLES, cast=int)
        batches   = _get("nbatches",   "batches",
                          default=_SD.NBATCHES, cast=int)
        inactive  = _get("ninactivebatches", "inactive_batches",
                          default=_SD.NINACTIVE, cast=int)
        flux      = _get("flux", "fluxo",
                          default=_SD.FLUX, cast=float)
        src_temp  = _get("source_temp_k", "fonte_temperatura_k",
                          default=_SD.SOURCE_TEMP_K, cast=float)
        src_ev    = _get("energy_ev", "fonte_energia_ev",
                          default=None, cast=float)

        # ── Depleção — auto-tune transparente ────────────────────────────
        # O usuário define apenas DT_H no input. DT_H_DEPLETION e integrador
        # são escolhidos automaticamente pelo DepletionAutoTuner.
        # Se o usuário definir explicitamente DT_H_DEPLETION ou
        # DEPLETION_INTEGRATOR no input, esses valores prevalecem (override).
        sim_mode = _get("simulation_mode", "SIMULATION_MODE",
                        default=SimulationModes.ACTIVATION)

        user_dt_dep  = _get("dt_h_depletion", "DTH_DEPLETION", default=None, cast=float)
        user_integ   = _get("depletion_integrator", "INTEGRADOR", default=None)

        dep_params = DepletionAutoTuner.tune(
            dt_output_h          = dt_output_h,
            user_dt_depletion_h  = user_dt_dep,
            user_integrator      = user_integ,
        )
        dt_depletion_h = dep_params.dt_depletion_h
        dep_integrator = dep_params.integrator
        dep_params.log_summary(logger.info)

        dep_normalization = _get("depletion_normalization", "NORMALIZACAO",
                                  default=_SD.DEPLETION_NORMALIZATION)
        if SimulationModes.needs_power(sim_mode):
            dep_normalization = "fission-q"
            logger.info("Modo %s: normalization forçada para 'fission-q'", sim_mode)

        # ── Geometria ─────────────────────────────────────────────────────
        wg   = geometry_result.get("wafer_geometry", {})
        x_cm = float(wg.get("x_cm", wg.get("xcm", _SD.WAFER_SIDE_CM)))
        y_cm = float(wg.get("y_cm", wg.get("ycm", _SD.WAFER_SIDE_CM)))
        area = x_cm * y_cm

        # ── Dados nucleares ───────────────────────────────────────────────
        lib_hier = LibraryHierarchy()
        lib_name, xs_path = lib_hier.discover()
        if xs_path is None:
            return {"success": False, "error": "Nenhuma biblioteca XS encontrada"}

        chain_path = self._find_chain(parser_data)
        if chain_path is None:
            return {"success": False, "error": "Chain file não encontrado"}

        if _OPENMC_OK:
            openmc.config["cross_sections"] = str(xs_path)

        # ── source_rate — MODO FLUX V242 ───────────────────────────────────────
        # FIX V242: No modo FLUX, NÃO há calibração. O fluxo nominal do reator
        # é prescrito diretamente por material no IndependentOperator.
        # source_rate_initial é apenas informativo para logs.
        
        source_rate_initial = flux * area
        if source_rate_initial < _VL.SOURCE_RATE_MIN:
            raise ValueError(
                f"source_rate_initial={source_rate_initial:.3e} n/s < mínimo ({_VL.SOURCE_RATE_MIN:.0e} n/s). "
                f"Verifique FLUXO e dimensões do wafer."
            )
        
        # Modo FLUX: calibração desativada - fluxo prescrito diretamente
        calibration_required = False
        logger.info(
            "MODO FLUX: source_rate_initial=%.4e n/s (informativo), flux=%s n/cm²/s prescrito",
            source_rate_initial, flux,
        )

        # ── openmc.Settings ───────────────────────────────────────────────
        if _OPENMC_OK:
            omc_settings = openmc.Settings()

            # FIX V224: eigenvalue precisa de inactive > 0 para convergência
            if SimulationModes.is_eigenvalue(sim_mode):
                omc_settings.run_mode = "eigenvalue"
                n_inactive = inactive if inactive > 0 else _SD.NINACTIVE_EIGENVALUE
                omc_settings.inactive = n_inactive
                logger.info(
                    "Modo eigenvalue: inactive=%d batches para convergência k_eff",
                    n_inactive,
                )
            else:
                omc_settings.run_mode = "fixed source"
                # Para fixed source, inactive=0 é correto (sem convergência de fonte)
                if inactive > 0:
                    omc_settings.inactive = inactive

            omc_settings.particles = particles
            omc_settings.batches   = batches
            omc_settings.output    = {"summary": True}

            # ── Temperatura ───────────────────────────────────────────────
            temp_treatment = _get(
                "temperature_treatment", "temp_treatment",
                default="interpolation",
            )
            temp_default_k = _get(
                "temperature_default_k", "temp_default",
                default=294.0, cast=float,
            )
            omc_settings.temperature = {
                "method":    temp_treatment,
                "default":   temp_default_k,
                "range":     [250.0, 2500.0],
                "tolerance": 200.0,
                "multipole": False,
            }
            logger.info(
                "temperature: method=%s  default=%.1f K",
                temp_treatment, temp_default_k,
            )
        else:
            omc_settings = None

        # ── Timesteps ─────────────────────────────────────────────────────
        # FIX V224: timesteps_s são DURAÇÕES em segundos para openmc.deplete.
        # Se DT_H_DEPLETION < DT_H_OUTPUT, usa sub-stepping automático.
        use_substeps = dt_depletion_h < dt_output_h - 1e-6
        if use_substeps:
            timesteps_s, output_indices = self._build_timesteps_depletion(
                dt_depletion_h, dt_output_h, total_h,
            )
            n_steps = len(timesteps_s)
            logger.info(
                "Sub-stepping ativo: %d passos internos (Δt=%.1fh) → %d pontos output",
                n_steps, dt_depletion_h, len(output_indices),
            )
        else:
            timesteps_s    = self._build_timesteps(dt_output_h, total_h)
            n_steps        = len(timesteps_s)
            output_indices = list(range(n_steps))  # todo passo é um output
            logger.info(
                "Timesteps: %d passos × %.2fh em segundos",
                n_steps, dt_output_h,
            )

        # Posições temporais absolutas em horas (para eixo X de gráficos/output)
        output_times_h = self._output_times_h(dt_output_h, total_h)

        # ── Source rate por passo — MODO FLUX V242 ───────────────────────────────
        # FIX V242: No modo FLUX, source_rates é informativo. O IndependentOperator
        # usa fluxes_list = [flux] * n_materiais diretamente, sem calibração.
        source_rates_initial = [source_rate_initial] * n_steps

        # ── Energia da fonte ──────────────────────────────────────────────
        energy_type = "single" if src_ev else "maxwell"
        energy_ev   = src_ev if src_ev else src_temp * _PC.KB_EV

        return {
            "success":  True,
            "version":  self.VERSION,

            "openmc_settings": omc_settings,

            "temporal_params": {
                "n_timesteps":      n_steps,
                "n_output_points":  len(output_indices),
                "dt_output_h":      dt_output_h,
                "dt_depletion_h":   dt_depletion_h,
                "total_time_h":     total_h,
                "cooling_time_h":   cooling_h,
                "output_indices":   output_indices,
                "output_times_h":   output_times_h,
                "chain_file":       str(chain_path),
                "n_timesteps_legacy": max(1, round(total_h / dt_output_h)),
                "dt_h":             dt_output_h,
            },

            "depletion_params": {
                "integrator":       dep_integrator,
                "normalization":    dep_normalization,
                "timesteps_s":      timesteps_s,
                # FIX BUG 5: exportar timesteps internos (sub-passos) separadamente.
                # simulation.py usava np.diff(output_times_h)*3600 → Δt=12h, ignorando
                # os sub-passos de 6h calculados pelo DepletionAutoTuner.
                # 'timesteps_internos_s' contém as durações reais do integrador (6h cada).
                "timesteps_internos_s": timesteps_s if use_substeps else None,
                "source_rates":     source_rates_initial,  # Informativo no modo FLUX
                "use_substeps":     use_substeps,
                "n_substeps":       dep_params.n_substeps,
                "auto_tune_band":   dep_params.band,
                "auto_tuned":       dep_params.auto_tuned,
                "dt_depletion_h":   dt_depletion_h,
            },

            "database_info": {
                "active_library":      lib_name,
                "xs_path":             str(xs_path),
                "available_libraries": [n for n, _ in lib_hier.available],
            },

            # LEGADO: 'timesteps' mantido mas agora contém durações em segundos
            # (não mais pontos temporais em horas). Maestro deve migrar para
            # depletion_params['timesteps_s'].
            "timesteps":    timesteps_s,

            "data_manager": ChainDataProxy(chain_path=chain_path, xs_path=xs_path),

            "source_params": {
                "strength":           source_rate_initial,  # Informativo no modo FLUX
                "source_rates":       source_rates_initial,  # Informativo no modo FLUX
                "energy_ev":          energy_ev   if _OPENMC_OK else None,
                "energy_type":        energy_type if _OPENMC_OK else None,
                "flux_n_cm2_s":       flux,
                "wafer_area_cm2":     area,
                "calibration_required": False,  # V242: modo FLUX não usa calibração
            },

            "simulation_mode": sim_mode,
            "error": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada público — dupla assinatura
# ─────────────────────────────────────────────────────────────────────────────

def create_settings(
    geometry_result:   Optional[Dict] = None,
    materials_list:    Optional[list] = None,
    simulation_params: Optional[Dict] = None,
    energy_source:     Any            = None,
    debug:             bool           = False,
    **kwargs,
) -> dict:
    """
    Wrapper de compatibilidade dupla — chama SettingsBuilder.build().

    Assinatura A (Maestro): create_settings(geometry_result, materials_list, simulation_params, ...)
    Assinatura B (legado):  create_settings(parser_data, geometry_result)
    """
    _is_sig_b = (
        isinstance(geometry_result, dict)
        and isinstance(materials_list, dict)
        and "openmc_geometry"  in materials_list
        and "openmc_materials" in materials_list
    )

    if _is_sig_b:
        return SettingsBuilder(debug=debug).build(geometry_result, materials_list)

    _geometry_result = geometry_result or {}
    _sim_p = simulation_params or {}
    _parser_data = {
        "simulation_parameters": _sim_p,
        "chain_file":          _sim_p.get("chain_file",        _sim_p.get("chainfile", "")),
        "flux":                _sim_p.get("flux",               _sim_p.get("fluxo",    1e13)),
        "dt_h":                _sim_p.get("dt_h",               _sim_p.get("dth",      _SD.DT_H_OUTPUT)),
        "dt_h_depletion":      _sim_p.get("dt_h_depletion",     _SD.DT_H_DEPLETION),
        "total_time_h":        _sim_p.get("total_time_h",       _sim_p.get("totaltimeh", _SD.TOTAL_TIME_H)),
        "cooling_time_h":      _sim_p.get("cooling_time_h",     _SD.COOLING_TIME_H),
        "nparticles":          _sim_p.get("nparticles",         _sim_p.get("neutrons_por_passo", _SD.NPARTICLES)),
        "nbatches":            _sim_p.get("nbatches",           _sim_p.get("batches", _SD.NBATCHES)),
        "ninactivebatches":    _sim_p.get("ninactivebatches",   _sim_p.get("inactive_batches", _SD.NINACTIVE)),
        "source_temp_k":       _sim_p.get("source_temp_k",      _SD.SOURCE_TEMP_K),
        "simulation_mode":     _sim_p.get("simulation_mode",    SimulationModes.ACTIVATION),
        "depletion_integrator": _sim_p.get("depletion_integrator", _SD.DEPLETION_INTEGRATOR),
        "depletion_normalization": _sim_p.get("depletion_normalization", _SD.DEPLETION_NORMALIZATION),
        "energy_ev": (
            (lambda es: es.get("energy_ev") or es.get("value_ev") or
             (es.get("data") or {}).get("energy_ev"))(energy_source)
            if isinstance(energy_source, dict) else None
        ),
    }
    return SettingsBuilder(debug=debug).build(_parser_data, _geometry_result)
