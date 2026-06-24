#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maestro.py V235 — Orquestrador do pipeline OpenMC (Phases A→G).

CHANGELOG V235 vs V234:
  FIX 1 — _build_sim_kwargs: agora injeta 'output_times_h' (pontos em horas)
           no campo 'timesteps' do settings_result repassado ao simulation.py.
           simulation._safe_timesteps usa np.diff(timesteps_h)*3600 → espera
           PONTOS temporais em horas, não durações em segundos. V234 passava
           settings_result["timesteps"] que em V224 virou durações em segundos
           → np.diff retornava zeros → zero passos válidos → fallback incorreto.

  FIX 2 — _build_sim_kwargs: injeta depletion_params completo no settings_result
           para que simulation.py futuro possa usar diretamente se atualizado.

  FIX 3 — validate_settings_output: adiciona verificação de 'depletion_params'
           e loga alerta (não falha) quando ausente — backward compat com V223.

  FIX 4 — run_pipeline: log de Phase C expandido com integrador, normalization
           e DT_H_DEPLETION para rastreabilidade dos parâmetros de depleção.

  FIX 5 — validate_settings_output: aceita 'depletion_params' OU 'timesteps'
           legado — não quebra com settings.py V223 anterior.

  FIX 6 — Phase C log: reporta source_rate, integrador e chain file para
           diagnóstico imediato sem precisar ler o log do settings.

  FIX 7 — _build_sim_kwargs: garante que source_rates (lista) e integrador
           são passados em system_params para uso por simulation.py V237+.
  
  FIX V237 — Atualização para compatibilidade com simulation.py V237.0:
           - Contrato temporal unificado: output_times_h como pontos em horas
           - depletion_dt_h separado de output_times_h para sub-stepping
  
  MELHORIA 1 — PipelineAudit.version atualizada para "V237".
  MELHORIA 2 — Alias MaestroV230 e MaestroV226 mantidos para compat.
  MELHORIA 3 — Log de Phase D expande info de source_rate e n_passos.
  MELHORIA 4 — finalize_pipeline loga também chain file e integrador usados.
"""

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MODULE_DIR = str(Path(__file__).parent.resolve())
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from config import ValidationLimits, TNLoopConfig, CoolingConfig, SimulationDefaults

try:
    from pyne_bridge import BRIDGE
except ImportError as _e:
    print(f"\n[MAESTRO] ERRO CRÍTICO: {_e}")
    sys.exit(1)

_LOG_FILE = Path(_MODULE_DIR) / "maestro_v235.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-14s - %(levelname)-8s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), mode="a", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def _inject_run_separator() -> None:
    sep = "=" * 90
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(str(_LOG_FILE), "a", encoding="utf-8") as fh:
            fh.write(f"\n{sep}\n  NOVO RUN — {ts}\n{sep}\n")
    except Exception:
        pass


_inject_run_separator()


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses de auditoria
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseAudit:
    phase_letter:       str
    phase_name:         str
    started_at:         str  = field(default_factory=lambda: datetime.now().isoformat())
    completed_at:       Optional[str] = None
    duration_seconds:   float = 0.0
    success:            bool  = False
    error_message:      str   = ""
    module_version:     str   = "unknown"
    contract_validated: bool  = False
    contract_message:   str   = ""

    def complete(self, success: bool = True, error_msg: str = "", version: str = "") -> None:
        self.completed_at   = datetime.now().isoformat()
        self.success        = success
        self.error_message  = error_msg
        self.module_version = version
        try:
            s = datetime.fromisoformat(self.started_at)
            e = datetime.fromisoformat(self.completed_at)
            self.duration_seconds = (e - s).total_seconds()
        except Exception:
            pass


@dataclass
class PipelineAudit:
    version:                 str  = "V238"
    started_at:              str  = field(default_factory=lambda: datetime.now().isoformat())
    ended_at:                Optional[str] = None
    total_duration_seconds:  float = 0.0
    phase_a:  Optional[PhaseAudit] = None
    phase_b:  Optional[PhaseAudit] = None
    phase_c:  Optional[PhaseAudit] = None
    phase_d:  Optional[PhaseAudit] = None
    phase_e:  Optional[PhaseAudit] = None
    phase_f:  Optional[PhaseAudit] = None
    tn_loop_history: List[Dict]    = field(default_factory=list)
    phase_g:  Optional[PhaseAudit] = None
    phases_completed:  int   = 0
    success:           bool  = False
    error_phase:       str   = ""
    error_message:     str   = ""
    contract_violations: List[str] = field(default_factory=list)
    warnings:            List[str] = field(default_factory=list)
    # V235: campos de rastreabilidade de depleção
    depletion_integrator:    str = ""
    depletion_normalization: str = ""
    chain_file:              str = ""
    dt_depletion_h:          float = 0.0
    # V238/V242: campos de rastreabilidade de fluxo e modo de operação
    flux_nominal:              float = 0.0
    n_dep_materials:           int   = 0
    source_rate_initial:       float = 0.0
    calibration_required:      bool  = False  # V242: False no modo FLUX

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# ContractValidator
# ─────────────────────────────────────────────────────────────────────────────

class ContractValidator:

    @staticmethod
    def validate_parser_output(result: Dict) -> Tuple[bool, str]:
        if not isinstance(result, dict):
            return False, "Parser não retornou dict"
        if not result.get("success"):
            return False, f"Parser falhou: {result.get('error', 'sem mensagem')}"
        for f in ("wafer_geometry", "layers", "simulation_parameters", "metadata"):
            if f not in result:
                return False, f"Falta campo obrigatório: {f}"
        return True, "Contrato Parser OK"

    @staticmethod
    def validate_geometry_output(result: Dict) -> Tuple[bool, str]:
        if not isinstance(result, dict):
            return False, "Geometry não retornou dict"
        if not result.get("success"):
            return False, f"Geometry falhou: {result.get('error', 'sem mensagem')}"
        for f in ("openmc_geometry", "openmc_materials", "metadata", "wafer_geometry",
                  "layers", "cells_dict", "materials_dict"):
            if f not in result:
                return False, f"Falta campo obrigatório: {f}"
        return True, "Contrato Geometry OK"

    @staticmethod
    def validate_settings_output(result: Dict) -> Tuple[bool, str]:
        """
        FIX V235: aceita tanto V224 (com 'depletion_params') quanto V223 legado.
        Falha apenas se campos absolutamente críticos estiverem ausentes.
        """
        if not isinstance(result, dict):
            return False, "Settings não retornou dict"
        if not result.get("success"):
            return False, f"Settings falhou: {result.get('error', 'sem mensagem')}"
        for f in ("openmc_settings", "temporal_params"):
            if f not in result:
                return False, f"Falta campo obrigatório: {f}"

        # Verificação não-fatal: depletion_params (novo V224)
        if "depletion_params" not in result:
            logger.warning(
                "validate_settings_output: 'depletion_params' ausente — "
                "settings.py pode ser V223. Integrador e normalization serão padrão."
            )
        else:
            dep = result["depletion_params"]
            if not dep.get("timesteps_s"):
                logger.warning(
                    "validate_settings_output: depletion_params['timesteps_s'] vazio"
                )
            if not dep.get("source_rates"):
                logger.warning(
                    "validate_settings_output: depletion_params['source_rates'] vazio"
                )

        # Verificação não-fatal: timesteps legado ou output_times_h
        tp = result.get("temporal_params", {})
        has_output_times = bool(tp.get("output_times_h"))
        has_legacy_ts    = bool(result.get("timesteps"))
        if not has_output_times and not has_legacy_ts:
            logger.warning(
                "validate_settings_output: nem 'output_times_h' nem 'timesteps' "
                "encontrados — simulation.py vai usar fallback de dt_h e total_h"
            )

        return True, "Contrato Settings OK"

    @staticmethod
    def validate_simulation_output(result: Dict) -> Tuple[bool, str]:
        if not isinstance(result, dict):
            return False, "Simulation não retornou dict"
        if not result.get("success"):
            return False, f"Simulation falhou: {result.get('error', 'sem mensagem')}"
        return True, "Contrato Simulation OK"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sp_get(sp: Dict, *keys, default="?"):
    for k in keys:
        v = sp.get(k)
        if v is not None:
            return v
    return default


def _layer_get(ldata: Dict, *keys, default="?"):
    for k in keys:
        v = ldata.get(k)
        if v is not None:
            return v
    return default


def _settings_get_timesteps_for_simulation(settings_result: Dict) -> List[float]:
    """
    FIX V235: retorna PONTOS temporais em horas para simulation.py.

    simulation._safe_timesteps faz np.diff(timesteps_h)*3600 → precisa de
    pontos [0, 12, 24, ..., 168], não durações [43200, 43200, ...].

    Hierarquia de busca:
    1. temporal_params['output_times_h'] — V224, formato correto
    2. settings_result['timesteps'] — V223 legado (se já forem pontos em horas)
    3. Reconstrução a partir de dt_output_h e total_time_h
    """
    tp = settings_result.get("temporal_params", {})

    # Opção 1: V224 — output_times_h são pontos em horas [0, dt, 2dt, ..., T]
    output_times = tp.get("output_times_h")
    if output_times and len(output_times) >= 2:
        logger.debug("timesteps para simulation: output_times_h (%d pontos)", len(output_times))
        return list(output_times)

    # Opção 2: legado — 'timesteps' no resultado (V223: pontos em horas)
    legacy = settings_result.get("timesteps")
    if legacy and len(legacy) >= 2:
        # Detectar se são pontos em horas ou durações em segundos
        # Heurística: se todos os valores > 1000, provavelmente estão em segundos
        first_val = float(legacy[0]) if legacy else 0.0
        if first_val > 1000:
            # São durações em segundos (V224 legado) → reconstruir pontos em horas
            total_s = sum(float(x) for x in legacy)
            n = len(legacy)
            dt_s = total_s / n
            points_h = [round(i * dt_s / 3600.0, 10) for i in range(n + 1)]
            logger.debug(
                "timesteps legado detectado em segundos → convertido para %d pontos em horas",
                len(points_h),
            )
            return points_h
        else:
            logger.debug("timesteps legado: %d pontos em horas", len(legacy))
            return [float(x) for x in legacy]

    # Opção 3: reconstrução
    dt_h    = float(tp.get("dt_output_h", tp.get("dt_h", SimulationDefaults.DT_H_OUTPUT)))
    total_h = float(tp.get("total_time_h", SimulationDefaults.TOTAL_TIME_H))
    n       = max(1, round(total_h / dt_h))
    points  = [round(i * dt_h, 10) for i in range(n + 1)]
    logger.warning(
        "timesteps reconstruídos do fallback: %d pontos, dt=%.1fh, total=%.1fh",
        len(points), dt_h, total_h,
    )
    return points


def _settings_patch_for_simulation(settings_result: Dict) -> Dict:
    """
    FIX V235: retorna cópia de settings_result com 'timesteps' garantidamente
    no formato que simulation.py espera (pontos em horas).

    Também injeta depletion_params em system_params para acesso em simulation.py V237+.
    Não modifica o dict original.
    """
    patched = dict(settings_result)  # cópia rasa

    # Substituir 'timesteps' pelo formato correto para simulation.py
    correct_ts = _settings_get_timesteps_for_simulation(settings_result)
    patched["timesteps"] = correct_ts

    return patched


# ─────────────────────────────────────────────────────────────────────────────
# MaestroV237
# ─────────────────────────────────────────────────────────────────────────────

class MaestroV237:
    """
    Orquestrador de pipeline OpenMC — fases A→B→C→D→E[→F][→G].
    
    FIX V237: Compatibilidade com simulation.py V237.0
      - Contrato temporal unificado (output_times_h como pontos em horas)
      - depletion_dt_h separado para sub-stepping
      - Integração com IndependentOperator corrigido
    
    A  Parser         lê Input-simulador.txt
    B  Geometry       constrói openmc.Geometry
    C  Settings       cross-sections, source, tallies, timesteps
    D  Simulation     depleção OpenMC
    E  Output         relatórios e CSVs  [não-crítica]
    F  T-N Loop       acoplamento Térmico-Neutrônico (se ENABLE_TN_COUPLING)
    G  PostProcessor  decaimento + dose + isótopos notáveis (se cooling_time_h > 0)
    
    Compatibilidade: aceita settings.py V223 e V224+.
    """

    VERSION = "V237"

    def __init__(self, output_dir: str = "pipeline_results", debug: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.audit = PipelineAudit()
        if debug:
            logger.setLevel(logging.DEBUG)

    # ── run_pipeline ──────────────────────────────────────────────────────────

    def run_pipeline(self, input_file: str = "Input-simulador.txt") -> Dict[str, Any]:
        t0  = time.time()
        sep = "=" * 90
        logger.info(sep)
        logger.info("MAESTRO %s — PIPELINE OPENMC  A→B→C→D→E[→F][→G]", self.VERSION)
        logger.info(sep)

        context: Dict[str, Any] = {}
        try:
            # ── A: Parser ────────────────────────────────────────────────────
            logger.info("PHASE A — PARSER")
            r = self.phase_a_parser(input_file)
            if not r.get("success"):
                return self.finalize_pipeline(t0, success=False,
                                              error=r.get("error", "Phase A falhou"))
            context["parser_data"] = r["data"]
            pd_ = r["data"]
            sp  = pd_.get("simulation_parameters", {})
            logger.info("  OK versão=%s  camadas=%d  modo=%s  partículas=%s",
                        pd_.get("version", "?"), len(pd_.get("layers", [])),
                        _sp_get(sp, "simulation_mode"), _sp_get(sp, "nparticles"))

            # ── Configurar acoplamento T-N a partir do input ─────────────────
            # THERMAL_COUPLING do Input-simulador.txt controla ENABLE_TN_COUPLING.
            # Deve ser feito ANTES de Phase B para que todas as fases subsequentes
            # já enxerguem o flag correto. Quando False:
            #   • Phase F (T-N loop + ThermalModel + GeometryUpdater) não executa
            #   • PyNE não é chamado no ciclo de irradiação (só no cooling se ativo)
            #   • SimulationRunner._tn_enabled() retorna False (já implementado)
            TNLoopConfig.from_parser(context["parser_data"])
            _tc = TNLoopConfig.ENABLE_TN_COUPLING
            logger.info(
                "  THERMAL_COUPLING=%s → Phase F %s | PyNE no ciclo de irradiação %s",
                _tc,
                "ATIVA" if _tc else "DESATIVADA (simulação OpenMC pura)",
                "ATIVO" if _tc else "DESACOPLADO",
            )

            # ── B: Geometry ──────────────────────────────────────────────────
            logger.info("PHASE B — GEOMETRY")
            r = self.phase_b_geometry(context["parser_data"])
            if not r.get("success"):
                return self.finalize_pipeline(t0, success=False,
                                              error=r.get("error", "Phase B falhou"))
            context["geometry_result"] = r
            wg = r.get("wafer_geometry", {})
            logger.info("  OK materiais=%d  células=%d  wafer=(%.3f × %.3f)cm",
                        len(r.get("openmc_materials", [])), len(r.get("cells_dict", {})),
                        wg.get("x_cm", 0), wg.get("y_cm", 0))

            # ── C: Settings ──────────────────────────────────────────────────
            logger.info("PHASE C — SETTINGS")
            r = self.phase_c_settings(context["parser_data"], context["geometry_result"])
            if not r.get("success"):
                return self.finalize_pipeline(t0, success=False,
                                              error=r.get("error", "Phase C falhou"))
            context["settings_result"] = r
            tp  = r.get("temporal_params", {})
            db  = r.get("database_info", {})
            dep = r.get("depletion_params", {})
            src = r.get("source_params", {})

            # FIX V235: log expandido com todos os parâmetros de depleção
            logger.info(
                "  OK v=%s  n_passos=%s  dt_output=%.1fh  dt_depletion=%.1fh  "
                "integrador=%s  normalization=%s",
                r.get("version", "?"),
                tp.get("n_timesteps", tp.get("n_timesteps_legacy", "?")),
                float(tp.get("dt_output_h", tp.get("dt_h", 0))),
                float(tp.get("dt_depletion_h", SimulationDefaults.DT_H_DEPLETION)),
                dep.get("integrator", "padrão"),
                dep.get("normalization", "padrão"),
            )
            logger.info(
                "  biblioteca=%s  chain=%s  source_rate=%.4e n/s",
                db.get("active_library", "?"),
                Path(tp.get("chain_file", "?")).name,
                float(src.get("strength", 0)),
            )

            # Registrar na auditoria para rastreabilidade
            self.audit.depletion_integrator    = dep.get("integrator", "")
            self.audit.depletion_normalization = dep.get("normalization", "")
            self.audit.chain_file              = tp.get("chain_file", "")
            self.audit.dt_depletion_h          = float(
                tp.get("dt_depletion_h", SimulationDefaults.DT_H_DEPLETION)
            )
            
            # V242: Registrar parâmetros de fluxo nominal (modo FLUX)
            self.audit.flux_nominal              = float(src.get("flux_n_cm2_s", 0))
            self.audit.source_rate_initial       = float(src.get("strength", 0))
            self.audit.calibration_required      = src.get("calibration_required", False)
            
            # MODO FLUX: não há calibração, fluxo prescrito diretamente
            logger.info(
                "  MODO FLUX: flux_nominal=%.4e n/cm²/s, calibration_required=%s",
                self.audit.flux_nominal,
                "NÃO" if not self.audit.calibration_required else "SIM",
            )

            # ── D: Simulation ────────────────────────────────────────────────
            logger.info("PHASE D — SIMULATION")
            r = self.phase_d_simulation(
                context["geometry_result"],
                context["settings_result"],
                context["parser_data"],
            )
            if not r.get("success"):
                return self.finalize_pipeline(t0, success=False,
                                              error=r.get("error", "Phase D falhou"))
            context["simulation_result"] = r
            
            # V242: Modo FLUX não usa calibração - remover bloco de extração
            # O fluxo nominal é prescrito diretamente no IndependentOperator
            
            pw = r.get("power_distribution", {})
            logger.info(
                "  OK h5=%s  camadas_com_potência=%d  timesteps_integrados=%d",
                Path(r.get("h5_depletion_path") or r.get("depletion_h5") or "?").name,
                sum(1 for v in pw.values() if v > 0),
                len(r.get("timestep_results", [])),
            )

            # ── E: Output ────────────────────────────────────────────────────
            logger.info("PHASE E — OUTPUT")
            r = self.phase_e_output(context["simulation_result"])
            if r.get("success"):
                context["output_result"] = r
                files = r.get("files_written", r.get("csv_files", []))
                logger.info("  OK arquivos=%d  dir=%s", len(files), r.get("output_dir", "?"))
            else:
                logger.warning("  AVISO (não-crítico): %s", r.get("error", "?"))
                self.audit.warnings.append(f"Phase E: {r.get('error', '?')}")

            # ── F: T-N Loop ──────────────────────────────────────────────────
            # TNLoopConfig.ENABLE_TN_COUPLING foi definido por from_parser() acima.
            # Quando THERMAL_COUPLING = false no input, Phase F é completamente
            # ignorada — nem ThermalModel nem GeometryUpdater nem PyNE são chamados.
            if TNLoopConfig.ENABLE_TN_COUPLING:
                logger.info("PHASE F — T-N LOOP (Picard)")
                r = self.phase_f_tn_loop(
                    geometry_result=context["geometry_result"],
                    settings_result=context["settings_result"],
                    parser_data=context["parser_data"],
                    initial_sim=context["simulation_result"],
                )
                context["simulation_result"] = r
                hist      = self.audit.tn_loop_history
                converged = hist[-1].get("converged", False) if hist else False
                logger.info("  OK iterações=%d  convergiu=%s",
                            len(hist), "SIM" if converged else "NÃO")
            else:
                logger.info(
                    "PHASE F — IGNORADA (THERMAL_COUPLING=false). "
                    "Resultado da Phase D usado diretamente. "
                    "PyNE desacoplado do ciclo de irradiação."
                )

            # ── G: Posprocessamento ──────────────────────────────────────────
            cooling_cfg = CoolingConfig.from_simulation_params(
                context["parser_data"].get("simulation_parameters", {})
            )
            logger.info("PHASE G — POSPROCESSAMENTO (cooling=%.1fh  ativo=%s)",
                        cooling_cfg.COOLING_TIME_H, cooling_cfg.phase_g_active)
            if cooling_cfg.phase_g_active:
                r = self.phase_g_postprocessing(
                    simulation_result=context["simulation_result"],
                    parser_data=context["parser_data"],
                    geometry_result=context["geometry_result"],
                    output_result=context.get("output_result"),
                    input_file=input_file,
                )
                if r.get("success"):
                    context["postprocessing_result"] = r
                    logger.info("  OK cooling=%.1fh  csvs=%d",
                                r.get("cooling_time_h", 0), len(r.get("csv_files", [])))
                else:
                    errs    = r.get("errors", [r.get("error", "?")])
                    msg_g   = f"Phase G falhou — {errs[0] if errs else '?'}"
                    logger.warning("  AVISO (não-crítico): %s", msg_g)
                    self.audit.warnings.append(f"Phase G: {msg_g}")
            else:
                logger.info("  Phase G inativa (cooling_time_h=0)")

            return self.finalize_pipeline(t0, success=True, context=context)

        except Exception as e:
            logger.error("ERRO CRÍTICO: %s", e)
            if self.debug:
                logger.debug(traceback.format_exc())
            return self.finalize_pipeline(t0, success=False, error=str(e))

    # ── Phase A ───────────────────────────────────────────────────────────────

    def phase_a_parser(self, input_file: str) -> Dict[str, Any]:
        try:
            self.audit.phase_a = PhaseAudit("A", "Parser")
            from parser import InputParser
            result = InputParser(debug=self.debug).parse_simulation_input(input_file)
            ok, msg = ContractValidator.validate_parser_output(result)
            if not ok:
                self.audit.phase_a.complete(False, msg, result.get("version", "?"))
                self.audit.contract_violations.append(msg)
                return {"success": False, "error": msg}
            self.audit.phase_a.contract_validated = True
            self.audit.phase_a.complete(True, "", result.get("version", "?"))
            self.audit.phases_completed += 1
            return {"success": True, "data": result}
        except Exception as e:
            msg = f"Erro em Phase A: {e}"
            logger.error("  %s", msg)
            if self.debug:
                logger.debug(traceback.format_exc())
            if self.audit.phase_a:
                self.audit.phase_a.complete(False, msg)
            return {"success": False, "error": msg}

    # ── Phase B ───────────────────────────────────────────────────────────────

    def phase_b_geometry(self, parser_data: Dict) -> Dict[str, Any]:
        try:
            self.audit.phase_b = PhaseAudit("B", "Geometry")
            from geometry import GeometryBuilder
            result = GeometryBuilder(debug=self.debug).build(parser_data)
            ok, msg = ContractValidator.validate_geometry_output(result)
            if not ok:
                self.audit.phase_b.complete(False, msg, result.get("version", "?"))
                self.audit.contract_violations.append(msg)
                return {"success": False, "error": msg}
            self.audit.phase_b.contract_validated = True
            self.audit.phase_b.complete(True, "", result.get("version", "?"))
            self.audit.phases_completed += 1
            return result
        except Exception as e:
            msg = f"Erro em Phase B: {e}"
            logger.error("  %s", msg)
            if self.debug:
                logger.debug(traceback.format_exc())
            if self.audit.phase_b:
                self.audit.phase_b.complete(False, msg)
            return {"success": False, "error": msg}

    # ── Phase C ───────────────────────────────────────────────────────────────

    def phase_c_settings(self, parser_data: Dict, geo_result: Dict) -> Dict[str, Any]:
        try:
            self.audit.phase_c = PhaseAudit("C", "Settings")
            from settings import create_settings
            result = create_settings(
                geometry_result=geo_result,
                materials_list=geo_result.get("openmc_materials", []),
                simulation_params=parser_data.get("simulation_parameters", {}),
                energy_source=parser_data.get("energy_source"),
                debug=self.debug,
            )
            ok, msg = ContractValidator.validate_settings_output(result)
            if not ok:
                self.audit.phase_c.complete(False, msg, result.get("version", "?"))
                self.audit.contract_violations.append(msg)
                return {"success": False, "error": msg}
            self.audit.phase_c.contract_validated = True
            self.audit.phase_c.complete(True, "", result.get("version", "?"))
            self.audit.phases_completed += 1
            return result
        except Exception as e:
            msg = f"Erro em Phase C: {e}"
            logger.error("  %s", msg)
            if self.debug:
                logger.debug(traceback.format_exc())
            if self.audit.phase_c:
                self.audit.phase_c.complete(False, msg)
            return {"success": False, "error": msg}

    # ── _build_sim_kwargs ─────────────────────────────────────────────────────

    def _build_sim_kwargs(self, geo_result: Dict, settings_result: Dict,
                          parser_data: Dict, sig_params: set) -> Dict[str, Any]:
        """
        FIX V235: patches settings_result para garantir compatibilidade com
        simulation.py (que usa np.diff(timesteps_h)*3600 → espera pontos em horas).

        Também injeta depletion_params em system_params para uso por versões
        futuras do simulation.py.
        """
        # FIX V235: settings_result patchado com timesteps no formato correto
        settings_patched = _settings_patch_for_simulation(settings_result)

        _wg  = geo_result.get("wafer_geometry", {})
        _spar = dict(parser_data.get("simulation_parameters", {}))
        if "wafer_x_cm" not in _spar:
            _spar["wafer_x_cm"] = _wg.get("x_cm", 1.69)
        if "wafer_y_cm" not in _spar:
            _spar["wafer_y_cm"] = _wg.get("y_cm", 1.69)

        # FIX V235/V236: injetar depletion_params e source_rate em system_params
        dep = settings_result.get("depletion_params", {})
        src_params = settings_result.get("source_params", {})
        
        if dep:
            _spar["_depletion_integrator"]    = dep.get("integrator", "")
            _spar["_depletion_normalization"] = dep.get("normalization", "")
            _spar["_source_rates"]            = dep.get("source_rates", [])
            _spar["_timesteps_s"]             = dep.get("timesteps_s", [])
        
        # V242: MODO FLUX - fluxo prescrito diretamente, sem calibração
        # O IndependentOperator usa fluxes_list = [flux] * n_materiais
        if src_params and "strength" in src_params:
            _spar["source_rate"] = src_params["strength"]
            self.logger.info(
                "MODO FLUX: source_rate=%.4e n/s (informativo), flux prescrito por material",
                src_params["strength"],
            )
        elif dep and "source_rates" in dep and dep["source_rates"]:
            # Se não tem 'strength', usa o primeiro da lista source_rates
            _spar["source_rate"] = dep["source_rates"][0]
        
        # FIX FASE 2: Injetar geometry_result em system_params para acesso em simulation.py
        _spar["_geometry_result"] = geo_result

        _es = parser_data.get("energy_source")
        if _es is not None:
            _es_dict = (
                _es.to_dict() if hasattr(_es, "to_dict")
                else (_es if isinstance(_es, dict) else None)
            )
            if _es_dict:
                _spar["energy_source"] = _es_dict
                if _es_dict.get("type") == "single" and "source_energy_ev" not in _spar:
                    _data = _es_dict.get("data") or _es_dict.get("parsed_data") or {}
                    _ev   = _data.get("energy_ev") or _es_dict.get("energy_ev")
                    if _ev is not None:
                        _spar["source_energy_ev"] = float(_ev)

        layers_val = parser_data.get("layers") or geo_result.get("layers") or {}

        # Detecção de assinatura de run_simulation
        if "geometry_result" in sig_params:
            kw: Dict[str, Any] = {
                "geometry_result": geo_result,
                "settings_result": settings_patched,
                "parser_result":   parser_data,
                "output_dir":      str(self.output_dir),
                "system_params":   _spar,  # FIX V241: incluir system_params com _geometry_result
            }
        elif "geometry" in sig_params and "settings_result" in sig_params:
            kw = {
                "geometry":        geo_result.get("openmc_geometry"),
                "layers":          layers_val,
                "materials":       geo_result.get("openmc_materials", []),
                "settings_result": settings_patched,
                "system_params":   _spar,
                "output_dir":      str(self.output_dir),
                "tn_loop_config":  TNLoopConfig,
                "geometry_result": geo_result,  # FIX V240: passar geometry_result completo para SimulationRunner
            }
        elif "geometry" in sig_params and "settings" in sig_params:
            kw = {
                "geometry":         geo_result.get("openmc_geometry"),
                "settings":         settings_patched.get("openmc_settings"),
                "temporal_params":  settings_patched.get("temporal_params", {}),
                "simulation_params": _spar,
                "output_dir":       str(self.output_dir),
            }
        else:
            # Assinatura mais completa — cobre simulation.py V236
            kw = {
                "geometry":         geo_result.get("openmc_geometry"),
                "geometry_result":  geo_result,
                "materials":        geo_result.get("openmc_materials", []),
                "layers":           layers_val,
                "settings":         settings_patched.get("openmc_settings"),
                "settings_result":  settings_patched,
                "temporal_params":  settings_patched.get("temporal_params", {}),
                "simulation_params": _spar,
                "system_params":    _spar,
                "parser_result":    parser_data,
                "tn_loop_config":   TNLoopConfig,
                "output_dir":       str(self.output_dir),
            }

        return {k: v for k, v in kw.items() if k in sig_params}

    # ── Phase D ───────────────────────────────────────────────────────────────

    def phase_d_simulation(self, geo_result: Dict, settings_result: Dict,
                            parser_data: Dict) -> Dict[str, Any]:
        try:
            self.audit.phase_d = PhaseAudit("D", "Simulation")
            from simulation import run_simulation
            import inspect
            try:
                sig_params = set(inspect.signature(run_simulation).parameters.keys())
            except (TypeError, ValueError):
                sig_params = set()

            kwargs = self._build_sim_kwargs(geo_result, settings_result,
                                             parser_data, sig_params)

            # Log dos timesteps que serão passados ao simulation
            ts_passados = kwargs.get("settings_result", {}).get("timesteps", [])
            if ts_passados:
                logger.debug(
                    "Phase D: timesteps para simulation: %d pontos  "
                    "[0]=%.2fh  [-1]=%.2fh",
                    len(ts_passados), float(ts_passados[0]), float(ts_passados[-1]),
                )

            result = run_simulation(**kwargs)

            ok, msg = ContractValidator.validate_simulation_output(result)
            if not ok:
                self.audit.phase_d.complete(False, msg, result.get("version", "?"))
                self.audit.contract_violations.append(msg)
                return {"success": False, "error": msg}
            self.audit.phase_d.contract_validated = True
            self.audit.phase_d.complete(True, "", result.get("version", "?"))
            self.audit.phases_completed += 1
            return result
        except Exception as e:
            msg = f"Erro em Phase D: {e}"
            logger.error("  %s", msg)
            if self.debug:
                logger.debug(traceback.format_exc())
            if self.audit.phase_d:
                self.audit.phase_d.complete(False, msg)
            return {"success": False, "error": msg}

    # ── Phase E ───────────────────────────────────────────────────────────────

    def phase_e_output(self, sim_result: Dict) -> Dict[str, Any]:
        try:
            self.audit.phase_e = PhaseAudit("E", "Output")
            from output import OutputProcessor
            result = OutputProcessor(debug=self.debug).process_results(sim_result)
            if not result.get("success"):
                self.audit.phase_e.complete(False, result.get("error", ""))
                return result
            self.audit.phase_e.complete(True, "", result.get("version", "?"))
            self.audit.phases_completed += 1
            return result
        except Exception as e:
            msg = f"Erro em Phase E (não-crítico): {e}"
            logger.warning("  %s", msg)
            if self.audit.phase_e:
                self.audit.phase_e.complete(False, msg)
            return {"success": False, "error": msg}

    # ── Phase F ───────────────────────────────────────────────────────────────

    def phase_f_tn_loop(self, geometry_result: Dict, settings_result: Dict,
                         parser_data: Dict, initial_sim: Dict) -> Dict[str, Any]:
        self.audit.phase_f = PhaseAudit("F", "T-N Loop")
        try:
            from thermal import ThermalModel
            from geometry import GeometryUpdater
            from simulation import run_simulation
        except ImportError as e:
            msg = f"Dependência Phase F não encontrada: {e}"
            logger.error("  %s", msg)
            self.audit.phase_f.complete(False, msg)
            return initial_sim

        # Enriquecer layers com material_name via thermal_materials
        thermal_mats = parser_data.get("thermal_materials", {})
        if thermal_mats and isinstance(thermal_mats, dict):
            layers_raw  = geometry_result.get("layers", [])
            layers_list = (list(layers_raw.values())
                           if isinstance(layers_raw, dict) else list(layers_raw))
            for ldata in layers_list:
                if not isinstance(ldata, dict):
                    continue
                lname    = ldata.get("name") or ldata.get("layer_name", "")
                tm_entry = thermal_mats.get(lname)
                if not tm_entry:
                    continue
                if isinstance(tm_entry, list) and tm_entry:
                    dominant = max(tm_entry, key=lambda c: float(c.get("mass_g", 0.0)))
                    mat = dominant.get("material") or dominant.get("material_name")
                elif isinstance(tm_entry, dict):
                    mat = tm_entry.get("material") or tm_entry.get("material_name")
                else:
                    mat = None
                if mat and not ldata.get("material_name"):
                    ldata["material_name"] = str(mat)

        thermal = ThermalModel(parser_data, geometry_result)
        updater = GeometryUpdater(
            materials_dict=geometry_result["materials_dict"],
            cells_dict=geometry_result["cells_dict"],
        )

        import inspect as _ins
        try:
            _sig_params = set(_ins.signature(run_simulation).parameters.keys())
        except (TypeError, ValueError):
            _sig_params = set()

        sim_result = initial_sim

        for it in range(TNLoopConfig.MAX_TN_ITERATIONS):
            power_dist   = sim_result.get("power_distribution", {})
            temp_map_raw = thermal.compute_temperature_profile(power_dist)

            if it == 0:
                temp_map = temp_map_raw
            else:
                temp_map = {
                    k: BRIDGE.apply_underrelaxation(v, thermal.last_temp_map.get(k, v))
                    for k, v in temp_map_raw.items()
                }

            sim_conv_flag = sim_result.get("tn_converged")
            if sim_conv_flag is not None:
                conv     = bool(sim_conv_flag)
                L2       = float(sim_result.get("tn_L2_K", 0.0))
                Linf     = float(sim_result.get("tn_Linf_K", L2))
                max_cell = sim_result.get("tn_max_diff_cell", "?")
            else:
                cr       = thermal.check_convergence(temp_map)
                conv     = bool(cr.converged)
                L2       = float(getattr(cr, "l2_norm", 0.0))
                Linf     = float(getattr(cr, "linf_norm", L2))
                max_cell = getattr(cr, "max_diff_cell", "?")

            entry: Dict[str, Any] = {
                "iter": it + 1, "L2_K": L2, "Linf_K": Linf,
                "max_diff_cell": max_cell, "converged": conv,
                "temp_map_K": {k: round(v, 3) for k, v in temp_map.items()},
            }
            self.audit.tn_loop_history.append(entry)
            logger.info(
                "  [TN iter %d/%d]  L2=%.4fK  Linf=%.4fK  max_diff@%s  %s",
                it + 1, TNLoopConfig.MAX_TN_ITERATIONS, L2, Linf, max_cell,
                "*** CONVERGED ***" if conv else "",
            )

            if conv:
                break

            updater.update_temperatures(temp_map)
            kwargs     = self._build_sim_kwargs(geometry_result, settings_result,
                                                parser_data, _sig_params)
            sim_result = run_simulation(**kwargs)

            if not sim_result.get("success"):
                logger.error("  Simulação falhou na iteração T-N %d — abortando", it + 1)
                break

        self.audit.phase_f.complete(True, "", f"{self.VERSION}-TNLoop")
        return sim_result

    # ── Phase G ───────────────────────────────────────────────────────────────

    def phase_g_postprocessing(
        self,
        simulation_result: Dict,
        parser_data: Dict,
        geometry_result: Dict,
        output_result: Optional[Dict] = None,
        input_file: str = "Input-simulador.txt",
    ) -> Dict[str, Any]:
        self.audit.phase_g = PhaseAudit("G", "PostProcessor")
        try:
            from posprocessamento import PostProcessor
            pp     = PostProcessor(output_dir=str(self.output_dir), debug=self.debug)
            result = pp.process(
                sim_result=simulation_result,
                parser_result=parser_data,
                geometry_result=geometry_result,
                output_result=output_result,
                input_file=input_file,
            )
            version = result.get("version", "V1.1")
            errors  = result.get("errors", [])
            if result.get("success"):
                self.audit.phase_g.complete(True, "", version)
                self.audit.phases_completed += 1
            else:
                warn_msg = f"Phase G com erros: {errors[0] if errors else '?'}"
                logger.warning("  %s", warn_msg)
                self.audit.phase_g.complete(False, warn_msg, version)
                self.audit.warnings.append(warn_msg)
            return result
        except Exception as e:
            msg = f"Erro em Phase G (não-crítico): {e}"
            logger.warning("  %s", msg)
            if self.debug:
                logger.debug(traceback.format_exc())
            if self.audit.phase_g:
                self.audit.phase_g.complete(False, msg)
            return {"success": False, "error": msg, "errors": [msg]}

    # ── finalize_pipeline ─────────────────────────────────────────────────────

    def finalize_pipeline(
        self,
        t0: float,
        success: bool,
        context: Optional[Dict] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        self.audit.ended_at               = datetime.now().isoformat()
        self.audit.total_duration_seconds = time.time() - t0
        self.audit.success                = success
        if not success:
            self.audit.error_message = error

        audit_path = self.output_dir / "pipeline_audit.json"
        self.audit.to_json(str(audit_path))

        sep = "=" * 90
        logger.info(sep)
        if success:
            logger.info(
                "SUCESSO  v=%s  fases=%d  duração=%.2fs  dir=%s  "
                "integrador=%s  chain=%s",
                self.VERSION,
                self.audit.phases_completed,
                self.audit.total_duration_seconds,
                self.output_dir,
                self.audit.depletion_integrator or "padrão",
                Path(self.audit.chain_file).name if self.audit.chain_file else "?",
            )
        else:
            logger.info("ERRO no pipeline%s", f": {error}" if error else "")
        logger.info(sep)

        return {
            "success":    success,
            "context":    context or {},
            "audit":      asdict(self.audit),
            "output_dir": str(self.output_dir),
        }


# ── Aliases de retrocompatibilidade ──────────────────────────────────────────
MaestroV235 = MaestroV237
MaestroV230 = MaestroV237
MaestroV226 = MaestroV237
__all__ = ["MaestroV237", "MaestroV235", "MaestroV230", "MaestroV226"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Maestro V235 — Pipeline OpenMC")
    ap.add_argument("input_file", nargs="?", default="Input-simulador.txt",
                    help="Arquivo de input (padrão: Input-simulador.txt)")
    ap.add_argument("--output-dir", default="pipeline_results",
                    help="Diretório de saída (padrão: pipeline_results)")
    ap.add_argument("--debug", action="store_true",
                    help="Ativa logging DEBUG")
    args = ap.parse_args()

    maestro = MaestroV235(output_dir=args.output_dir, debug=args.debug)
    result  = maestro.run_pipeline(input_file=args.input_file)
    sys.exit(0 if result.get("success") else 1)
