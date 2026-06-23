#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════════
POSPROCESSAMENTO.PY V1.1 — PRODUCTION READY
Data: 2026-04-14
Status: PRODUCTION READY

ENGINES IMPLEMENTADAS
──────────────────────
[F1] PyNE.Material.decay() → Inventário resfriado (Bateman analítico)
     Atividade [Bq], calor de decaimento [W],
     massa por isótopo após cooling_time_h
[F2] TemperatureHistoryBuilder → T(t) durante irradiação + resfriamento
     Modelo convectivo 1D (água/refrigeração)
[F3] PhotonDoseEstimator → Dose fotônica pós-shutdown (R2S simplificado)
     ICRP-116 H*(10) por camada [µSv/h]
[F4] NotableDaughterDetector → Rastreia isótopos de interesse médico/industrial
     Mo99→Tc99m, I131, Lu177, Y90, etc.
[F5] StructuralActivationSolver → ALARA-like via pyne.transmute.chainsolve
     Ativação estrutural usando φ do tally OpenMC

INPUTS (pipeline context)
──────────────────────────
sim_result       : Dict de simulation.py
parser_result    : Dict de parser.py
geometry_result  : Dict de geometry.py
output_result    : Dict de output.py
input_file       : str (Input-simulador.txt — para ler cooling_time_h)

CONTRATO DE RETORNO run_postprocessing() → Dict:
──────────────────────────────────────────────────
success            : bool
version            : str
cooling_time_h     : float
phase_decay        : {CAMADA_N: {activity_Bq, decay_heat_W, massa_g,
                     total_activity_Bq, total_decay_heat_W}}
temperature_history: {irradiation: [{time_h, T_K, Q_W}],
                     cooling: [{time_h, T_K, Q_W}]}
photon_dose        : {CAMADA_N: [{cooling_time_h, dose_uSv_h}]}
notable_daughters  : {parent: {daughter, activity_Bq, half_life_h, T99m}}
structural_activation: {CAMADA_N: {nuc: activity_Bq}} # se chainsolve OK
csv_files          : [str]
errors             : [str]
metadata           : dict

INTEGRAÇÃO COM MAESTRO
──────────────────────

phase_f = run_postprocessing(
    sim_result,
    parser_result,
    geometry_result,
    output_result,
    input_file="Input-simulador.txt",
)
════════════════════════════════════════════════════════════════════════════════
"""

import csv
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import PhysicsConstants, NuclearDataPaths, SimulationDefaults

# Fator de conversão Bq → Ci  (1 Ci = 3.7×10¹⁰ desintegrações/s)
_BQ_TO_CI: float = 1.0 / 3.7e10

# ═══════════════════════════════════════════════════════════════════════════
# DEPENDÊNCIAS (todas opcionais com graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import openmc
    import openmc.deplete
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False

try:
    from pyne.material import Material as PyneMat
    from pyne import data as pynedata
    from pyne import nucname as pynenucname
    HAS_PYNE = True
except ImportError:
    HAS_PYNE = False

try:
    from pyne.transmute.chainsolve import Transmuter
    HAS_CHAINSOLVE = True
except ImportError:
    HAS_CHAINSOLVE = False

# Reutiliza leitor de depletion do módulo output.py (PP3)
try:
    from output import H5DepletionReader
    HAS_H5_READER = True
except ImportError:
    HAS_H5_READER = False

# ═══════════════════════════════════════════════════════════════════════════
# METADADOS
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "V1.1"

AVOGADRO = PhysicsConstants.N_A
LN2      = PhysicsConstants.LN2

# ─────────────────────────────────────────────────────────────────────────────
# Helper: abre openmc.deplete.Results SEM mexer no cwd global (PP1)
# ─────────────────────────────────────────────────────────────────────────────
def _open_results(h5_path: str):
    """
    Abre openmc.deplete.Results usando caminho absoluto.

    Correção PP1:
      - Remove os.chdir() global, evitando corrupção de CWD em usos concorrentes.
      - Usa sempre o caminho absoluto str(h5p_abs) na chamada a Results.
      - Mantém fallback para ResultsList.from_hdf5 em versões antigas.

    FIX PP3: verifica existência do arquivo antes de abrir, evitando
      FileNotFoundError sem mensagem útil subindo até PostProcessor.process().
    """
    if not HAS_OPENMC:
        raise RuntimeError("OpenMC não disponível para leitura de depletion")

    h5p_abs = Path(h5_path).resolve()
    # FIX PP3: guard explícito com mensagem diagnóstica
    if not h5p_abs.exists():
        raise FileNotFoundError(
            f"_open_results: arquivo de depleção não encontrado: {h5p_abs}\n"
            f"  Verifique se a simulação OpenMC (Phase D) completou com sucesso\n"
            f"  e se o campo 'depletion_h5' no resultado aponta para o arquivo correto."
        )
    try:
        # API padrão OpenMC 0.15.3 aceita caminho absoluto
        return openmc.deplete.Results(str(h5p_abs))
    except TypeError:
        # Fallback: ResultsList (versões alternativas da API)
        if hasattr(openmc.deplete, "ResultsList") and hasattr(openmc.deplete.ResultsList, "from_hdf5"):
            return openmc.deplete.ResultsList.from_hdf5(str(h5p_abs))
        raise


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def _setup_logger(name: str = "posprocessamento",
                  debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


logger = _setup_logger()


# ═══════════════════════════════════════════════════════════════════════════
# ISÓTOPOS DE INTERESSE MÉDICO / INDUSTRIAL
# ═══════════════════════════════════════════════════════════════════════════

# Formato: pai → {filha, meia_vida_filha_h, uso}
NOTABLE_DAUGHTERS: Dict[str, Dict[str, Any]] = {
    "Mo99":  {"daughter": "Tc99_m1", "half_life_h": 6.0,
              "use": "SPECT diagnóstico"},
    "I131":  {"daughter": "Xe131_m1", "half_life_h": 11.84,
              "use": "Terapia tireóide"},
    "Lu177": {"daughter": "Hf177", "half_life_h": 1.0e6,
              "use": "Terapia oncológica"},
    "Y90":   {"daughter": "Zr90", "half_life_h": 1.0e6,
              "use": "Radioimunoterapia"},
    "Ga67":  {"daughter": "Zn67", "half_life_h": 1.0e6,
              "use": "SPECT diagnóstico"},
    "Cu64":  {"daughter": "Ni64", "half_life_h": 1.0e6,
              "use": "PET oncológico"},
    "Zn65":  {"daughter": "Cu65", "half_life_h": 1.0e6,
              "use": "Dosimetria / pesquisa"},
    "Ga68":  {"daughter": "Zn68", "half_life_h": 1.13,
              "use": "PET diagnóstico"},
}

# ═══════════════════════════════════════════════════════════════════════════
# ICRP-116 H*(10) ambient dose coefficients [pSv·cm²] vs E_photon [MeV]
# ICRP 116 Table A.1 — interpolação linear em log-log
# ═══════════════════════════════════════════════════════════════════════════

_ICRP116_E_MEV = [
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10,
    0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50, 2.00,
    3.00, 4.00, 5.00, 6.00, 8.00, 10.0,
]

_ICRP116_H_PSV_CM2 = [
    0.00789, 0.00580, 0.00693, 0.0863, 0.255, 0.416, 0.542, 0.644,
    0.730, 0.875, 1.14, 1.31, 1.57, 1.79, 1.96, 2.10,
    2.30, 2.45, 2.71, 2.91, 3.18, 3.40, 3.53, 3.63,
    3.71, 3.70,
]


def _icrp116_h_pSv_cm2(energy_mev: float) -> float:
    """
    Retorna H*(10) [pSv·cm²] por interpolação linear em log-log
    da tabela ICRP-116.
    """
    E = float(energy_mev)
    if E <= _ICRP116_E_MEV[0]:
        return _ICRP116_H_PSV_CM2[0]
    if E >= _ICRP116_E_MEV[-1]:
        return _ICRP116_H_PSV_CM2[-1]
    for i in range(len(_ICRP116_E_MEV) - 1):
        e0, e1 = _ICRP116_E_MEV[i], _ICRP116_E_MEV[i + 1]
        if e0 <= E <= e1:
            h0, h1 = _ICRP116_H_PSV_CM2[i], _ICRP116_H_PSV_CM2[i + 1]
            try:
                t = math.log(E / e0) / math.log(e1 / e0)
                return math.exp(math.log(h0) + t * math.log(h1 / h0))
            except (ValueError, ZeroDivisionError):
                return (h0 + h1) / 2.0
    return _ICRP116_H_PSV_CM2[-1]


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS: COOLING TIME + WATER PARAMS
# ═══════════════════════════════════════════════════════════════════════════

def _find_cooling_time_h(
    parser_result: Optional[Dict[str, Any]] = None,
    input_file_path: str = "Input-simulador.txt",
    default_h: float = 6.0,
) -> float:
    """
    Resolve TEMPO DE RESFRIAMENTO_H.

    O parser.py não captura este campo (chave contém espaço).
    Estratégia em cascata:
      1. parser_result['simulation_parameters']['cooling_time_h']
      2. Scan direto no Input-simulador.txt por regex
      3. Default 6.0h
    """
    # 1 — via parser_result
    if parser_result:
        sp = parser_result.get("simulation_parameters", {})
        for key in ("cooling_time_h", "resfriamento_h", "tempo_resfriamento_h"):
            val = sp.get(key)
            if val is not None:
                logger.info("cooling_time_h=%.1fh (via parser_result)", float(val))
                return float(val)

    # 2 — scan direto no arquivo
    if os.path.exists(input_file_path):
        pattern = re.compile(
            r"TEMPO\s+DE\s+RESFRIAMENTO_H\s+([\d.eE+\-]+)",
            re.IGNORECASE,
        )
        try:
            with open(input_file_path, encoding="utf-8") as fh:
                for line in fh:
                    line_clean = line.split("#")[0].strip()
                    m = pattern.search(line_clean)
                    if m:
                        val = float(m.group(1))
                        logger.info(
                            "cooling_time_h=%.1fh (lido de %s)",
                            val, input_file_path,
                        )
                        return val
        except Exception as exc:
            logger.warning("Falha ao ler cooling_time de %s: %s",
                           input_file_path, exc)

    logger.warning("cooling_time_h não encontrado → usando default %.1fh", default_h)
    return default_h


def _find_water_params(
    parser_result: Optional[Dict[str, Any]] = None,
    input_file_path: str = "Input-simulador.txt",
) -> Tuple[float, float]:
    """
    Retorna (water_temp_K, flow_rate_m3s).
    Fallback: 298.15K, 0.001 m³/s.
    """
    t_c = 25.0
    flow = 0.001

    if os.path.exists(input_file_path):
        try:
            with open(input_file_path, encoding="utf-8") as fh:
                for line in fh:
                    lc = line.split("#")[0].strip()
                    tok = lc.split()
                    if len(tok) >= 2:
                        k = tok[0].upper()
                        try:
                            if k == "WATER_TEMPERATURE":
                                t_c = float(tok[1])
                            elif k == "WATER_FLOW_RATE":
                                flow = float(tok[1])
                        except ValueError:
                            pass
        except Exception:
            pass

    return (t_c + 273.15, flow)


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE F1: PyNE DECAY PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class PyneDecayProcessor:
    """
    Aplica PyNE.Material.decay() ao inventário pós-irradiação.

    Entrada: composição final em átomos {nuc_openmc: n_atoms}
             (obtida de openmc.deplete.Results[-1])
    Saída:   {nuc: mass_g, activity_Bq, decay_heat_W}

    Se PyNE não estiver disponível, usa fallback via openmc.data +
    constantes de decaimento do chain file (chain_reader do output.py).
    """

    def __init__(self, logger_obj: Optional[logging.Logger] = None) -> None:
        self.log = logger_obj or logging.getLogger("PyneDecay")

    # ------------------------------------------------------------------
    def process(
        self,
        final_atoms: Dict[str, float],  # {nuc_openmc: n_atoms}
        cooling_time_h: float,
        chain_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retorna dict com inventário após resfriamento.
        """
        if HAS_PYNE:
            return self._process_pyne(final_atoms, cooling_time_h)
        else:
            self.log.warning(
                "PyNE não disponível → usando fallback OpenMC+chain_file"
            )
            return self._process_fallback(
                final_atoms, cooling_time_h, chain_file_path
            )

    # ------------------------------------------------------------------
    def _process_pyne(
        self,
        final_atoms: Dict[str, float],
        cooling_time_h: float,
    ) -> Dict[str, Any]:
        """Bateman analítico via PyNE."""
        self.log.info("[F1] PyNE.decay() cooling=%.2fh", cooling_time_h)

        # Converter {openmc_nuc: atoms} → {pyne_id: mass_g}
        comp_mass: Dict[int, float] = {}
        for nuc_str, n_atoms in final_atoms.items():
            if n_atoms <= 0.0:
                continue
            nuc_id = self._to_pyne_id(nuc_str)
            if nuc_id is None:
                continue
            try:
                M = pynedata.atomic_mass(nuc_id)  # g/mol exato
            except Exception:
                digits = "".join(c for c in nuc_str if c.isdigit())
                M = float(digits) if digits else 1.0
            mass_g = n_atoms * M / AVOGADRO
            comp_mass[nuc_id] = comp_mass.get(nuc_id, 0.0) + mass_g

        total_mass = sum(comp_mass.values())
        if total_mass <= 0.0:
            return self._empty_result("Inventário vazio após conversão PyNE")

        # ── PP-FIX-SCALE: construção do Material PyNE e decay analítico ────────
        # PROBLEMA: PyNE 0.7+ não preserva mass= após mat.decay() — retorna
        # cooled.mass ≈ 1.0 (material normalizado), não a massa real em gramas.
        # Isso causa quedas de 20-23 ordens de grandeza na massa pós-resfriamento
        # (e.g. Mo-99: 4.36e-6 g → 2.19e-29 g em apenas 12h, fisicamente impossível).
        #
        # SOLUÇÃO: não depender de cooled.mass para escalar massas.
        # Para cada nuclídeo pai: m(t) = m0 * exp(-λ * t)  [Bateman simplificado]
        # Para nuclídeos filhos: fração_filha * total_mass_after
        # onde total_mass_after = Σ m_pai(t) — conservação de massa.
        comp_frac = {nuc_id: m_g / total_mass for nuc_id, m_g in comp_mass.items()}
        try:
            mat = PyneMat(comp_frac, mass=total_mass)
        except Exception as exc:
            self.log.error("[F1] PyneMat() falhou: %s", exc)
            return self._empty_result(f"PyneMat erro: {exc}")

        # Decay analítico via PyNE (usado para activity() e decay_heat())
        t_s = cooling_time_h * 3600.0
        try:
            cooled = mat.decay(t_s)
        except Exception as exc:
            self.log.error("[F1] mat.decay() falhou: %s", exc)
            return self._empty_result(f"pyne.decay erro: {exc}")

        # ── Extração de massas com escala analítica (PP-FIX-SCALE) ───────────
        try:
            act_dict  = cooled.activity()   # {nuc_id: Bq} — escala absoluta OK
            comp_dict = dict(cooled.comp)   # {nuc_id: mass_fraction} pós-decay

            _dh_raw = cooled.decay_heat()
            if hasattr(_dh_raw, "items"):
                _dh_sum = float(sum(
                    v for v in _dh_raw.values()
                    if math.isfinite(float(v))
                ))
            else:
                _dh_sum = float(_dh_raw) if math.isfinite(float(_dh_raw)) else 0.0
            decay_heat = _dh_sum

            # Massa pós-decay de cada nuclídeo pai via Bateman: m(t) = m0·exp(-λ·t)
            mass_after_parents: Dict[int, float] = {}
            for nuc_id, mass_0_g in comp_mass.items():
                try:
                    lam = float(pynedata.decay_const(nuc_id))  # s⁻¹
                except Exception:
                    lam = 0.0
                if lam > 0.0:
                    mass_after_parents[nuc_id] = mass_0_g * math.exp(-lam * t_s)
                else:
                    mass_after_parents[nuc_id] = mass_0_g  # estável

            # Massa total pós-decay (conservação de massa dos pais)
            total_mass_after = sum(mass_after_parents.values())

            # Montar mass_dict: pais → analítico; filhos → fração × total_after
            mass_dict: Dict[int, float] = {}
            for nuc_id, frac in comp_dict.items():
                frac_f = float(frac)
                if not math.isfinite(frac_f) or frac_f <= 0.0:
                    continue
                if nuc_id in mass_after_parents:
                    mass_dict[nuc_id] = mass_after_parents[nuc_id]
                else:
                    mass_dict[nuc_id] = frac_f * total_mass_after

            self.log.debug(
                "[F1] PP-FIX-SCALE: total_mass=%.4e g → total_after=%.4e g (t=%.1fh)",
                total_mass, total_mass_after, cooling_time_h,
            )
        except Exception as exc:
            self.log.error("[F1] Extração PyNE falhou: %s", exc)
            return self._empty_result(f"Extração PyNE erro: {exc}")

        # Converter IDs de volta para string OpenMC
        activity_by_name: Dict[str, float] = {}
        mass_by_name: Dict[str, float] = {}

        import math as _math
        _act_nonzero = sum(
            1 for v in act_dict.values()
            if _math.isfinite(float(v)) and float(v) > 0.0
        )
        _act_zero = sum(
            1 for v in act_dict.values()
            if _math.isfinite(float(v)) and float(v) == 0.0
        )
        _act_sample = {
            str(k): float(v) for k, v in list(act_dict.items())[:3]
        }
        self.log.info(
            "[F1-diag] cooled.activity(): %d total | %d > 0 | %d == 0 | amostra=%s",
            len(act_dict), _act_nonzero, _act_zero, _act_sample,
        )

        for nuc_id, act in act_dict.items():
            try:
                name = self._pyne_id_to_openmc_str(nuc_id)
                val = float(act)
                if _math.isfinite(val) and val > 0.0:
                    activity_by_name[name] = val
            except Exception:
                pass

        for nuc_id, mass in mass_dict.items():
            try:
                name = self._pyne_id_to_openmc_str(nuc_id)
                val = float(mass)
                if _math.isfinite(val) and val >= 0.0:
                    mass_by_name[name] = val
            except Exception:
                pass

        total_activity = sum(activity_by_name.values())

        # BUG-2 FIX: PyNE 0.7.7 retorna decay_heat() = 0 para muitos nuclídeos.
        # Se a soma for zero mas houver atividade real, estimar Q via A × E_médio.
        _E_DECAY_MEV_PYNE: Dict[str, float] = {
            "Mo99": 0.436, "Tc99m": 0.143, "I131": 0.192,
            "Lu177": 0.149, "Y90": 0.935, "Cu64": 0.278,
            "Ga68": 0.836, "Zr95": 0.360, "Nb95": 0.765,
            "Zn65": 0.329, "Co57": 0.122, "Co60": 1.252,
            "Cs137": 0.512, "Ba137m": 0.661, "Sr90": 0.196,
        }
        _J_PER_MEV = 1.602176634e-13  # J/MeV

        if decay_heat == 0.0 and total_activity > 0.0:
            _q_est = 0.0
            for _nuc, _act_bq in activity_by_name.items():
                _nuc_base = re.sub(r"_m\d+$", "", _nuc)
                _e_mev = _E_DECAY_MEV_PYNE.get(
                    _nuc, _E_DECAY_MEV_PYNE.get(_nuc_base, 0.5)
                )
                _q_est += _act_bq * _e_mev * _J_PER_MEV
            decay_heat = _q_est
            self.log.info(
                "[F1] Q_decay estimado via A×E_médio (PyNE 0.7.7 limitation): %.4f W",
                decay_heat,
            )
        elif decay_heat > 0.0:
            self.log.debug("[F1] Q_decay via pyne.decay_heat(): %.4f W", decay_heat)

        # Sanidade: massa total pós-decay deve ser próxima à pré-decay
        cooled_mass_sum = sum(mass_by_name.values())
        mass_ratio = cooled_mass_sum / total_mass if total_mass > 0.0 else 0.0
        if not (0.5 < mass_ratio < 1.5):
            self.log.warning(
                "[F1] SANIDADE MASSA: pré=%.4e g  pós=%.4e g  ratio=%.4f — "
                "valores fora do esperado (0.5–1.5). Verifique API PyNE.",
                total_mass, cooled_mass_sum, mass_ratio,
            )
        else:
            self.log.info(
                "[F1] Massa: pré=%.4e g  pós=%.4e g  ratio=%.6f  (conservação OK)",
                total_mass, cooled_mass_sum, mass_ratio,
            )

        self.log.info(
            "[F1] ✓ PyNE decay: %d nuclídeos A_total=%.3e Ci Q=%.4f W",
            len(activity_by_name), total_activity * _BQ_TO_CI, decay_heat,
        )

        return {
            "success": True,
            "engine": "pyne.Material.decay",
            "cooling_time_h": cooling_time_h,
            "activity_Bq": activity_by_name,
            "mass_g": mass_by_name,
            "total_activity_Bq": total_activity,
            "total_decay_heat_W": float(decay_heat),
            "total_mass_g": total_mass,
            "n_nuclides": len(activity_by_name),
            "errors": [],
        }

    # ------------------------------------------------------------------
    def _process_fallback(
        self,
        final_atoms: Dict[str, float],
        cooling_time_h: float,
        chain_file_path: Optional[str],
    ) -> Dict[str, Any]:
        """
        Fallback sem PyNE: A(t) = A0 * exp(-λ*t).
        Usa λ do chain file XML (mesmo leitor do output.py).
        """
        import xml.etree.ElementTree as ET

        self.log.info(
            "[F1-fallback] A(t)=A₀·e^(-λt) cooling=%.2fh", cooling_time_h
        )

        # Ler half-lives do chain file
        half_lives: Dict[str, float] = {}
        if chain_file_path and os.path.exists(chain_file_path):
            try:
                root = ET.parse(chain_file_path).getroot()
                for nucl in root.findall(".//nuclide"):
                    name = nucl.get("name")
                    d = nucl.find("decay")
                    if name and d is not None:
                        hl = d.get("half_life")
                        if hl:
                            half_lives[name] = float(hl)
            except Exception as exc:
                self.log.warning("[F1-fallback] chain parse erro: %s", exc)

        t_s = cooling_time_h * 3600.0
        activity_by_name: Dict[str, float] = {}
        mass_by_name: Dict[str, float] = {}
        total_decay_heat = 0.0

        for nuc_str, n_atoms in final_atoms.items():
            if n_atoms <= 0.0:
                continue

            # Massa
            digits = "".join(c for c in nuc_str if c.isdigit())
            M = float(digits) if digits else 1.0
            if HAS_OPENMC:
                try:
                    M = openmc.data.atomic_mass(nuc_str)
                except Exception:
                    pass

            mass_0 = n_atoms * M / AVOGADRO

            # Constante de decaimento
            hl_s = half_lives.get(nuc_str, 0.0)
            if hl_s <= 0.0:
                mass_by_name[nuc_str] = mass_0
                continue

            lam = LN2 / hl_s
            act_0 = lam * n_atoms
            act_t = act_0 * math.exp(-lam * t_s)
            n_t = n_atoms * math.exp(-lam * t_s)
            mass_t = n_t * M / AVOGADRO

            if act_t > 0.0:
                activity_by_name[nuc_str] = act_t
            if mass_t > 0.0:
                mass_by_name[nuc_str] = mass_t

            # Calor de decaimento aproximado: Q ≈ A * E_decay_médio
            total_decay_heat += act_t * 1e6 * PhysicsConstants.EV_TO_J

        total_activity = sum(activity_by_name.values())

        self.log.info(
            "[F1-fallback] ✓ %d radioisótopos A=%.3e Ci Q_aprox=%.4fW",
            len(activity_by_name), total_activity * _BQ_TO_CI, total_decay_heat,
        )

        return {
            "success": True,
            "engine": "fallback_chain_halflife",
            "cooling_time_h": cooling_time_h,
            "activity_Bq": activity_by_name,
            "mass_g": mass_by_name,
            "total_activity_Bq": total_activity,
            "total_decay_heat_W": total_decay_heat,
            "total_mass_g": sum(mass_by_name.values()),
            "n_nuclides": len(activity_by_name),
            "errors": ["PyNE indisponível — fallback mode"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _to_pyne_id(nuc_str: str) -> Optional[int]:
        """Converte string OpenMC (e.g. 'Zn64', 'Tc99_m1') → int PyNE."""
        if not HAS_PYNE:
            return None
        try:
            return pynenucname.id(nuc_str)
        except Exception:
            pass
        base = re.sub(r"_m\d+$", "", nuc_str)
        try:
            return pynenucname.id(base)
        except Exception:
            return None

    @staticmethod
    def _pyne_id_to_openmc_str(nuc_id: int) -> str:
        """Converte int PyNE → string OpenMC."""
        if not HAS_PYNE:
            return str(nuc_id)
        try:
            return pynenucname.openmc(nuc_id)
        except Exception:
            try:
                return pynenucname.name(nuc_id)
            except Exception:
                return str(nuc_id)

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        return {
            "success": False, "engine": "none",
            "activity_Bq": {}, "mass_g": {},
            "total_activity_Bq": 0.0, "total_decay_heat_W": 0.0,
            "total_mass_g": 0.0, "n_nuclides": 0,
            "errors": [reason],
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE F2: TEMPERATURE HISTORY BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class TemperatureHistoryBuilder:
    """
    Constrói T(t) durante irradiação e resfriamento.

    Modelo térmico 1D convectivo:
        T(t) = T_water + Q(t) / (ρ_water × Cp_water × flow_rate)

    Q(t) = Q_neutron(t) + Q_decay(t)
      ├─ irradiação: Q_neutron ≈ φ × σ_abs × E_dep × Volume
      └─ resfriamento: Q_neutron = 0; Q_decay(t) de PyNE/fallback

    Usa PyNE.decay() nos múltiplos timesteps do depletion_results.h5
    para calcular Q_decay(t) ao longo de toda a história.
    """

    # Propriedades da água a 25°C
    RHO_WATER = 997.0   # kg/m³
    CP_WATER = 4182.0   # J/(kg·K)
    # E_dep por captura neutrônica (estimativa)
    E_DEP_J_PER_CAP = 8e6 * 1.602e-19  # 8 MeV

    def __init__(
        self,
        water_temp_K: float,
        flow_rate_m3s: float,
        logger_obj: Optional[logging.Logger] = None,
    ) -> None:
        self.T_water = water_temp_K
        self.flow = flow_rate_m3s
        self.log = logger_obj or logging.getLogger("TempHistory")
        # Resistência térmica efetiva [K/W]
        self._R_th = 1.0 / (self.RHO_WATER * self.CP_WATER * self.flow)
        self.log.info(
            "[F2] TempHistoryBuilder: T_water=%.1fK flow=%.4fm³/s R_th=%.4fK/W",
            self.T_water, self.flow, self._R_th,
        )

    def build_from_depletion(
        self,
        h5_path: str,
        cooling_time_h: float,
        flux_ncm2s: float = 1e13,
        volume_cm3: float = 1.0,
        chain_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Constrói histórico completo T(t) lendo o H5.

        Retorna dict com listas 'irradiation' e 'cooling'.
        """
        self.log.info("[F2] Construindo T(t) de %s", h5_path)

        history = {"irradiation": [], "cooling": [], "errors": []}

        if not HAS_OPENMC or not HAS_NUMPY:
            history["errors"].append(
                "OpenMC ou NumPy indisponível para ler H5"
            )
            history["irradiation"] = self._estimate_irradiation(
                flux_ncm2s, volume_cm3, cooling_time_h
            )
            history["cooling"] = self._build_cooling_phase(
                Q0=0.1, cooling_time_h=cooling_time_h
            )
            return history

        try:
            res = _open_results(h5_path)
            n_steps = len(res)

            if n_steps == 0:
                raise ValueError("H5 sem timesteps")

            # Tempos de irradiação
            if hasattr(res, "get_times"):
                try:
                    times_s = np.asarray(
                        res.get_times(time_units="s"), dtype=float
                    )
                except TypeError:
                    times_s = np.asarray(res.get_times("s"), dtype=float)
            else:
                times_s = np.array(
                    [float(step.time[0]) for step in res],
                    dtype=float,
                )

            self.log.info(
                "[F2] %d timesteps de irradiação lidos (t_final=%.1fh)",
                n_steps, times_s[-1] / 3600.0,
            )

            # Estima Q_neutron constante durante irradiação
            flux_cm3 = flux_ncm2s * volume_cm3 ** (1.0 / 3.0)
            Q_neutron_W = (
                flux_cm3 * 1.0e-24 * self.E_DEP_J_PER_CAP * 1e24
            )
            Q_neutron_W = min(Q_neutron_W, 50.0)

            irrad_points = []
            for i, t_s in enumerate(times_s):
                T_K = self.T_water + Q_neutron_W * self._R_th
                irrad_points.append({
                    "time_h": float(t_s / 3600.0),
                    "T_K": round(T_K, 3),
                    "Q_W": round(Q_neutron_W, 4),
                    "phase": "irradiation",
                })

            history["irradiation"] = irrad_points

            # Fase de resfriamento: Q_decay(t) decai exponencialmente
            history["cooling"] = self._build_cooling_from_final(
                res, cooling_time_h, chain_file_path
            )

        except Exception as exc:
            self.log.error("[F2] Erro ao ler H5: %s", exc)
            history["errors"].append(str(exc))
            history["irradiation"] = self._estimate_irradiation(
                flux_ncm2s, volume_cm3, 24.0
            )
            history["cooling"] = self._build_cooling_phase(
                Q0=0.5, cooling_time_h=cooling_time_h
            )

        self.log.info(
            "[F2] ✓ T(t): %d pts irradiação %d pts resfriamento",
            len(history["irradiation"]),
            len(history["cooling"]),
        )

        return history

    # ------------------------------------------------------------------
    def _build_cooling_from_final(
        self,
        deplete_results: Any,
        cooling_time_h: float,
        chain_file_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Usa composição final + PyNE.decay(t) para múltiplos pontos
        no período de resfriamento.
        """
        import xml.etree.ElementTree as ET

        final_atoms = self._extract_final_atoms(deplete_results)

        if not final_atoms:
            return self._build_cooling_phase(
                Q0=0.2, cooling_time_h=cooling_time_h
            )

        n_cool_pts = 20
        t_cool_s = [
            i * (cooling_time_h * 3600.0) / (n_cool_pts - 1)
            for i in range(n_cool_pts)
        ]

        cooling_points = []
        proc = PyneDecayProcessor(self.log)

        half_lives: Dict[str, float] = {}
        if chain_file_path and os.path.exists(chain_file_path):
            try:
                root = ET.parse(chain_file_path).getroot()
                for nucl in root.findall(".//nuclide"):
                    name = nucl.get("name")
                    d = nucl.find("decay")
                    if name and d is not None:
                        hl = d.get("half_life")
                        if hl:
                            half_lives[name] = float(hl)
            except Exception:
                pass

        for t_s in t_cool_s:
            t_h = t_s / 3600.0

            if HAS_PYNE:
                res = proc.process(final_atoms, t_h)
                Q = res["total_decay_heat_W"]
            else:
                Q = self._wayne_tobias_decay_heat(
                    t_s, final_atoms, half_lives
                )

            T_K = self.T_water + max(Q, 0.0) * self._R_th
            cooling_points.append({
                "time_h": round(t_h, 4),
                "T_K": round(T_K, 3),
                "Q_W": round(Q, 6),
                "phase": "cooling",
            })

        return cooling_points

    @staticmethod
    def _extract_final_atoms(deplete_results: Any) -> Dict[str, float]:
        """Extrai composição do último timestep do depletion_results."""
        if not HAS_NUMPY:
            return {}
        try:
            step = deplete_results[-1]
            nucs = (
                list(deplete_results.nuclides)
                if hasattr(deplete_results, "nuclides")
                else list(step.index_nuc.keys())
            )
            mat_ids = []
            if hasattr(deplete_results, "materials"):
                mat_ids = [str(m) for m in deplete_results.materials]
            elif hasattr(step, "mat_to_ind"):
                mat_ids = [str(k) for k in step.mat_to_ind.keys()]

            if not mat_ids:
                return {}

            atoms_total: Dict[str, float] = {}
            for mat_id in mat_ids:
                for nuc in nucs:
                    try:
                        _, a = deplete_results.get_atoms(
                            mat_id, nuc, nuc_units="atoms", time_units="s"
                        )
                        a = np.asarray(a, dtype=float)
                        val = float(a[-1])
                        if val > 0.0:
                            atoms_total[nuc] = atoms_total.get(nuc, 0.0) + val
                    except Exception:
                        pass
            return atoms_total
        except Exception:
            return {}

    @staticmethod
    def _wayne_tobias_decay_heat(
        t_s: float,
        final_atoms: Dict[str, float],
        half_lives: Dict[str, float],
    ) -> float:
        """
        Aproximação analítica de calor de decaimento.
        Q(t) = Σ_i A_i(0) * exp(-λ_i*t) * E_avg
        onde E_avg ≈ 1 MeV.
        """
        if t_s <= 0.0:
            t_s = 1.0
        E_avg_J = 1e6 * PhysicsConstants.EV_TO_J
        Q = 0.0
        for nuc, n_atoms in final_atoms.items():
            hl_s = half_lives.get(nuc, 0.0)
            if hl_s <= 0.0:
                continue
            lam = LN2 / hl_s
            A_t = lam * n_atoms * math.exp(-lam * t_s)
            Q += A_t * E_avg_J
        return Q

    def _estimate_irradiation(
        self,
        flux_ncm2s: float,
        volume_cm3: float,
        total_h: float,
    ) -> List[Dict[str, Any]]:
        """Fallback: T constante durante irradiação."""
        Q_neut = min(flux_ncm2s * 1e-10 * volume_cm3, 20.0)
        T_K = self.T_water + Q_neut * self._R_th
        return [
            {"time_h": 0.0, "T_K": T_K, "Q_W": Q_neut, "phase": "irradiation"},
            {"time_h": total_h, "T_K": T_K, "Q_W": Q_neut,
             "phase": "irradiation"},
        ]

    def _build_cooling_phase(
        self,
        Q0: float,
        cooling_time_h: float,
    ) -> List[Dict[str, Any]]:
        """Fallback: Q(t) = Q0 * exp(-t/tau), tau=2h."""
        tau_h = 2.0
        n_pts = 20
        points = []
        for i in range(n_pts):
            t_h = i * cooling_time_h / (n_pts - 1)
            Q = Q0 * math.exp(-t_h / tau_h)
            T_K = self.T_water + Q * self._R_th
            points.append({
                "time_h": round(t_h, 4),
                "T_K": round(T_K, 3),
                "Q_W": round(Q, 6),
                "phase": "cooling",
            })
        return points


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE F3: PHOTON DOSE ESTIMATOR (R2S simplificado)
# ═══════════════════════════════════════════════════════════════════════════

class PhotonDoseEstimator:
    """
    Estima dose fotônica pós-shutdown usando ICRP-116 H*(10).

    Metodologia:
      1. Para cada nuclídeo, obtém linhas γ via PyNE ou tabela interna
      2. Calcula fluência φ_γ = A × Y_γ / (4π r²) para r=1m
      3. Converte fluência → dose: H = φ_γ × h*(E)
    """

    _GAMMA_TABLE: Dict[str, List[Tuple[float, float]]] = {
        "Zn65":  [(1.1155, 0.5060)],
        "Ga67":  [(0.09332, 0.381), (0.1847, 0.212)],
        "Ga68":  [(0.5110, 1.78)],
        "Cu64":  [(0.5110, 0.354), (1.3459, 0.00473)],
        "Mo99":  [(0.1405, 0.894), (0.7397, 0.1213)],
        "Tc99_m1": [(0.1405, 0.891)],
        "I131":  [(0.3645, 0.817), (0.6373, 0.0717)],
        "Co60":  [(1.1732, 0.9985), (1.3325, 0.9998)],
        "Cs137": [(0.6617, 0.8510)],
        "U235":  [(0.18574, 0.5720)],
        "Np239": [(0.1061, 0.252), (0.2285, 0.108)],
    }

    def __init__(
        self,
        distance_m: float = 1.0,
        logger_obj: Optional[logging.Logger] = None,
    ) -> None:
        self.dist = distance_m
        self.log = logger_obj or logging.getLogger("PhotonDose")
        self.log.info("[F3] PhotonDoseEstimator: distância=%.2fm", self.dist)

    def estimate(
        self,
        decay_result: Dict[str, Any],
        cooling_points: List[float],
        chain_file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retorna lista de pontos {cooling_time_h, dose_uSv_h}.
        """
        if not decay_result.get("success"):
            return [{
                "cooling_time_h": 0.0,
                "dose_uSv_h": 0.0,
                "error": "decay inválido",
            }]

        activity_bq = decay_result.get("activity_Bq", {})
        if not activity_bq:
            return []

        points = []
        for t_h in cooling_points:
            dose_uSv_h = self._calc_dose(activity_bq, t_h, decay_result)
            points.append({
                "cooling_time_h": round(t_h, 4),
                "dose_uSv_h": round(dose_uSv_h, 6),
            })

        self.log.info(
            "[F3] ✓ Dose estimada em %d pontos (t=0: %.3f µSv/h)",
            len(points),
            points[0]["dose_uSv_h"] if points else 0.0,
        )

        return points

    def _calc_dose(
        self,
        activity_bq: Dict[str, float],
        cooling_h: float,
        decay_result: Dict[str, Any],
    ) -> float:
        """
        H*(10) [µSv/h] a 1m de distância.
        """
        area_cm2 = 4.0 * math.pi * (self.dist * 100.0) ** 2
        dose_sv_s = 0.0

        for nuc, act_bq in activity_bq.items():
            if act_bq <= 0.0:
                continue

            gamma_lines = self._get_gamma_lines(nuc)
            for E_mev, Y in gamma_lines:
                phi_cm2s = act_bq * Y / area_cm2
                h_pSv = _icrp116_h_pSv_cm2(E_mev)
                dose_sv_s += phi_cm2s * h_pSv * 1e-12

        dose_uSv_h = dose_sv_s * 3600.0 * 1e6
        return dose_uSv_h

    def _get_gamma_lines(
        self, nuc: str
    ) -> List[Tuple[float, float]]:
        """
        Retorna linhas gamma do nuclídeo.
        Prioridade: PyNE → tabela interna → estimativa genérica.
        """
        if HAS_PYNE:
            try:
                nuc_id = pynenucname.id(nuc)
                energies = pynedata.gamma_energy(nuc_id)   # keV
                intensities = pynedata.gamma_intensity(nuc_id)
                if energies is not None and len(energies) > 0:
                    return [
                        (float(E) / 1000.0, float(I))
                        for E, I in zip(energies, intensities)
                        if E > 0 and I > 0
                    ]
            except Exception:
                pass

        if nuc in self._GAMMA_TABLE:
            return self._GAMMA_TABLE[nuc]

        return [(1.0, 0.5)]


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE F4: NOTABLE DAUGHTER DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class NotableDaughterDetector:
    """
    Detecta isótopos filhos de interesse médico/industrial
    no inventário pós-resfriamento.
    """

    def __init__(self, logger_obj: Optional[logging.Logger] = None) -> None:
        self.log = logger_obj or logging.getLogger("DaughterDetect")

    def detect(
        self,
        decay_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retorna dict com filhas detectadas e suas propriedades.
        """
        found: Dict[str, Dict[str, Any]] = {}
        act_bq = decay_result.get("activity_Bq", {})
        mass_g = decay_result.get("mass_g", {})

        for parent, info in NOTABLE_DAUGHTERS.items():
            daughter = info["daughter"]
            for nuc in (parent, daughter):
                act = act_bq.get(nuc, 0.0)
                if act > 1.0:
                    found[nuc] = {
                        "activity_Bq": act,
                        "mass_g": mass_g.get(nuc, 0.0),
                        "half_life_h": info["half_life_h"],
                        "use": info["use"],
                        "parent": parent,
                        "daughter": daughter,
                        "is_parent": nuc == parent,
                        "is_daughter": nuc == daughter,
                    }
                    self.log.info(
                        "[F4] ✓ %s (A=%.3e Bq) — %s",
                        nuc, act, info["use"],
                    )

        return found


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE F5: STRUCTURAL ACTIVATION SOLVER (ALARA-like via chainsolve)
# ═══════════════════════════════════════════════════════════════════════════

class StructuralActivationSolver:
    """
    Ativação estrutural via pyne.transmute.chainsolve.

    Usa o fluxo neutrónico extraído do statepoint.h5 (tally).
    Só executa se chainsolve + PyNE estiverem disponíveis.
    """

    def __init__(self, logger_obj: Optional[logging.Logger] = None) -> None:
        self.log = logger_obj or logging.getLogger("ChainSolve")

    def solve(
        self,
        layer_name: str,
        isotopes_g: Dict[str, float],
        flux_ncm2s: float,
        irradiation_time_s: float,
        temperature_K: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Transmuta material estrutural usando chainsolve.
        """
        if not HAS_PYNE or not HAS_CHAINSOLVE:
            return {
                "success": False,
                "error": "PyNE chainsolve indisponível",
                "activity_Bq": {},
            }

        self.log.info(
            "[F5] chainsolve %s: φ=%.2e n/cm²/s t=%.1fh T=%.0fK",
            layer_name,
            flux_ncm2s,
            irradiation_time_s / 3600.0,
            temperature_K,
        )

        try:
            total_g = sum(isotopes_g.values())
            if total_g <= 0.0:
                return {
                    "success": False,
                    "error": "massa zero",
                    "activity_Bq": {},
                }

            comp_mass: Dict[int, float] = {}
            for nuc_str, mass in isotopes_g.items():
                nuc_id = PyneDecayProcessor._to_pyne_id(nuc_str)
                if nuc_id is not None:
                    comp_mass[nuc_id] = mass

            mat = PyneMat(comp_mass)
            mat.mass = total_g

            transmuter = Transmuter(
                t=irradiation_time_s,
                phi=flux_ncm2s,
                temp=temperature_K,
                tol=1e-12,
            )

            activated = transmuter.transmute(mat)

            act_dict = activated.activity()
            activity_bq = {
                PyneDecayProcessor._pyne_id_to_openmc_str(k): float(v)
                for k, v in act_dict.items()
                if float(v) > 0.0
            }

            total_act = sum(activity_bq.values())
            self.log.info(
                "[F5] ✓ %s: %d nuclídeos ativados A=%.3e Bq",
                layer_name, len(activity_bq), total_act,
            )

            _dh5_raw = activated.decay_heat()
            _dh5_W = (
                float(sum(_dh5_raw.values()))
                if hasattr(_dh5_raw, "items")
                else float(_dh5_raw)
            )

            return {
                "success": True,
                "layer": layer_name,
                "activity_Bq": activity_bq,
                "total_activity_Bq": total_act,
                "decay_heat_W": _dh5_W,
                "n_nuclides": len(activity_bq),
            }

        except Exception as exc:
            self.log.error("[F5] chainsolve %s falhou: %s", layer_name, exc)
            return {
                "success": False,
                "error": str(exc),
                "activity_Bq": {},
            }


# ═══════════════════════════════════════════════════════════════════════════
# H5 FINAL INVENTORY READER (extrai composição do último timestep)
# ═══════════════════════════════════════════════════════════════════════════

class FinalInventoryReader:
    """
    Lê o inventário final por material de depletion_results.h5.

    Retorna {material_id: {nuc: n_atoms}} do último timestep.

    PP3:
      - Quando OpenMC está disponível, delega a leitura do H5 para
        H5DepletionReader de output.py (API única).
      - Fallback via h5py permanece para casos sem OpenMC.
    """

    def __init__(self, logger_obj: Optional[logging.Logger] = None) -> None:
        self.log = logger_obj or logging.getLogger("FinalInventory")

    def read(self, h5_path: str) -> Dict[str, Dict[str, float]]:
        if not HAS_NUMPY:
            self.log.warning("NumPy indisponível — inventário vazio")
            return {}

        h5p = Path(h5_path)
        if not h5p.exists():
            self.log.error("H5 não encontrado: %s", h5_path)
            return {}

        self.log.info("[FinalInventory] Lendo %s", h5_path)

        # Estratégia 1: h5py direto (sem necessidade de OpenMC)
        try:
            import h5py
            return self._read_via_h5py(h5p)
        except ImportError:
            self.log.debug(
                "[FinalInventory] h5py indisponível, tentando via H5DepletionReader"
            )
        except Exception as exc:
            self.log.warning(
                "[FinalInventory] h5py falhou (%s), tentando via H5DepletionReader",
                exc,
            )

        # Estratégia 2: H5DepletionReader de output.py (sem chdir)
        if HAS_OPENMC and HAS_H5_READER:
            try:
                reader = H5DepletionReader(self.log)
                data = reader.read_depletion_h5(str(h5p))
                if not data.get("success"):
                    errs = data.get("errors", ["desconhecido"])
                    self.log.error(
                        "[FinalInventory] H5DepletionReader falhou: %s", errs[0]
                    )
                    return {}
                mats = data.get("materials", {})
                result: Dict[str, Dict[str, float]] = {}
                for mat_id, mdata in mats.items():
                    nucl = mdata.get("nuclide_names", [])
                    num = mdata.get("number")
                    if num is None:
                        continue
                    arr = np.asarray(num, dtype=float)
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    last = arr[-1, :]
                    atoms: Dict[str, float] = {}
                    for j, nuc in enumerate(nucl):
                        if j < len(last):
                            val = float(last[j])
                            if val > 0.0:
                                atoms[nuc] = val
                    result[str(mat_id)] = atoms
                    self.log.info(
                        " [H5DepletionReader] Material %s: %d nuclídeos",
                        mat_id, len(atoms),
                    )
                return result
            except Exception as exc:
                self.log.error("[FinalInventory] Erro via H5DepletionReader: %s", exc)
                return {}

        self.log.warning(
            "[FinalInventory] Nem h5py nem H5DepletionReader disponíveis — inventário vazio"
        )
        return {}

    def _read_via_h5py(self, h5p: Path) -> Dict[str, Dict[str, float]]:
        """
        Lê inventário final direto via h5py — OpenMC 0.15.x.

        Estrutura real do depletion_results.h5:
          /materials/<mat_id>/attrs['index']              → índice linha em /number
          /materials/<mat_id>/attrs['volume']             → volume cm³
          /nuclides/<nuc_name>/attrs['atom number index'] → índice coluna em /number
          /number  → Dataset (n_steps, n_mats, n_nucs)   → átomos por passo
        """
        import h5py

        result: Dict[str, Dict[str, float]] = {}

        with h5py.File(str(h5p), "r") as f:
            # ── 1. Mapa nuclídeo → índice coluna ─────────────────────────────
            if "nuclides" not in f:
                raise ValueError("Grupo 'nuclides' não encontrado no HDF5")

            nuc_to_col: Dict[str, int] = {}
            nuc_grp = f["nuclides"]
            if hasattr(nuc_grp, "items"):          # Group (formato 0.15.x)
                for nuc_name, nuc_handle in nuc_grp.items():
                    col = int(nuc_handle.attrs["atom number index"])
                    nuc_to_col[nuc_name] = col
            else:                                  # Dataset legado (fallback)
                try:
                    raw = nuc_grp.asstr()[:]
                except AttributeError:
                    raw = nuc_grp[:]
                nuc_to_col = {
                    (n.decode() if isinstance(n, bytes) else str(n)): i
                    for i, n in enumerate(raw.flat)
                }

            if not nuc_to_col:
                raise ValueError("/nuclides está vazio")
            self.log.debug("[h5py] %d nuclídeos lidos", len(nuc_to_col))

            # ── 2. Mapa material → (índice linha, volume) ────────────────────
            if "materials" not in f:
                raise ValueError("Grupo 'materials' não encontrado no HDF5")

            mat_grp = f["materials"]
            mat_to_row:  Dict[str, int]   = {}
            mat_to_vol:  Dict[str, float] = {}

            if hasattr(mat_grp, "items"):          # Group (formato 0.15.x)
                for mat_id, mat_handle in mat_grp.items():
                    mat_to_row[mat_id] = int(mat_handle.attrs["index"])
                    mat_to_vol[mat_id] = float(mat_handle.attrs.get("volume", 1.0))
            else:                                  # Dataset legado
                for i, mid in enumerate(mat_grp[()].flat):
                    s = str(int(mid)) if not isinstance(mid, (str, bytes)) else (
                        mid.decode() if isinstance(mid, bytes) else mid
                    )
                    mat_to_row[s] = i
                    mat_to_vol[s] = 1.0

            if not mat_to_row:
                raise ValueError("/materials está vazio")
            self.log.debug("[h5py] %d materiais lidos", len(mat_to_row))

            # ── 3. Array /number ─────────────────────────────────────────────
            if "number" not in f:
                raise ValueError("Dataset 'number' não encontrado no HDF5")

            number = f["number"][()]          # (n_steps, n_mats, n_nucs)
            if number.ndim != 3:
                raise ValueError(
                    f"/number tem shape {number.shape}, esperado (n_steps, n_mats, n_nucs)"
                )
            last_step = number[-1]            # (n_mats, n_nucs)
            self.log.debug("[h5py] /number shape=%s, lendo step=%d", number.shape, number.shape[0]-1)

            # ── 4. Extrair átomos por material ───────────────────────────────
            for mat_id, row_idx in sorted(mat_to_row.items(), key=lambda x: x[1]):
                if row_idx >= last_step.shape[0]:
                    self.log.warning(
                        "[h5py] Material %s: row_idx=%d fora de range (n_mats=%d)",
                        mat_id, row_idx, last_step.shape[0],
                    )
                    continue

                row = last_step[row_idx]       # (n_nucs,)
                atoms: Dict[str, float] = {}
                for nuc_name, col_idx in nuc_to_col.items():
                    if col_idx < len(row):
                        val = float(row[col_idx])
                        if val > 0.0:
                            atoms[nuc_name] = val

                result[mat_id] = atoms
                total_atoms = float(sum(atoms.values())) if atoms else 0.0
                self.log.info(
                    " [h5py] Material %s (row=%d, vol=%.3e cm³): "
                    "%d nuclídeos com átomos > 0  total_atoms=%.3e",
                    mat_id, row_idx, mat_to_vol.get(mat_id, 0.0),
                    len(atoms), total_atoms,
                )

        return result


# ═══════════════════════════════════════════════════════════════════════════
# CSV EXPORTER
# ═══════════════════════════════════════════════════════════════════════════

class PostProcessorCSVExporter:
    """Exporta todos os resultados do pós-processamento para CSV."""

    def __init__(
        self,
        output_dir: Path,
        logger_obj: Optional[logging.Logger] = None,
    ) -> None:
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = logger_obj or logging.getLogger("PostCSV")
        self.exported: List[str] = []

    def export_all(
        self,
        phase_decay: Dict[str, Any],
        temperature_history: Dict[str, Any],
        photon_dose: Dict[str, Any],
        notable_daughters: Dict[str, Any],
        structural_activation: Dict[str, Any],
    ) -> List[str]:

        self._export_decay_inventory(phase_decay)
        self._export_temperature_history(temperature_history)
        self._export_photon_dose(photon_dose)
        self._export_notable_daughters(notable_daughters)
        if structural_activation:
            self._export_structural_activation(structural_activation)

        self.log.info(
            "[CSV] %d arquivos exportados em %s",
            len(self.exported), self.out,
        )

        return list(self.exported)

    # ── F1: inventário resfriado ────────────────────────────────────
    def _export_decay_inventory(self, phase_decay: Dict[str, Any]) -> None:
        path = self.out / "inventario_resfriado.csv"
        rows = []
        for layer_name, ldata in phase_decay.items():
            act = ldata.get("activity_Bq", {})
            mass = ldata.get("mass_g", {})
            all_nucs = set(act.keys()) | set(mass.keys())
            for nuc in sorted(all_nucs):
                rows.append({
                    "camada": layer_name,
                    "nuclideo": nuc,
                    "massa_g": mass.get(nuc, 0.0),
                    "atividade_Ci": act.get(nuc, 0.0) * _BQ_TO_CI,
                })
        if rows:
            self._write_csv(path, rows, [
                "camada", "nuclideo", "massa_g", "atividade_Ci"
            ])
            self.log.info(" ✓ inventario_resfriado.csv (%d linhas)", len(rows))

    # ── F2: histórico de temperatura ────────────────────────────────
    def _export_temperature_history(
        self, temperature_history: Dict[str, Any]
    ) -> None:
        path = self.out / "temperatura_historico.csv"
        rows = []
        for phase_key in ("irradiation", "cooling"):
            for pt in temperature_history.get(phase_key, []):
                rows.append({
                    "fase": phase_key,
                    "time_h": pt["time_h"],
                    "T_K": pt["T_K"],
                    "T_C": round(pt["T_K"] - 273.15, 3),
                    "Q_W": pt["Q_W"],
                })
        if rows:
            self._write_csv(path, rows,
                            ["fase", "time_h", "T_K", "T_C", "Q_W"])
            self.log.info(
                " ✓ temperatura_historico.csv (%d linhas)", len(rows)
            )

    # ── F3: dose fotônica ───────────────────────────────────────────
    def _export_photon_dose(self, photon_dose: Dict[str, Any]) -> None:
        path = self.out / "dose_fotonica_pós_shutdown.csv"
        rows = []
        for layer_name, pts in photon_dose.items():
            for pt in pts:
                rows.append({
                    "camada": layer_name,
                    "cooling_time_h": pt["cooling_time_h"],
                    "dose_uSv_h": pt["dose_uSv_h"],
                })
        if rows:
            self._write_csv(path, rows,
                            ["camada", "cooling_time_h", "dose_uSv_h"])
            self.log.info(
                " ✓ dose_fotonica_pós_shutdown.csv (%d linhas)", len(rows)
            )

    # ── F4: filhas de interesse ─────────────────────────────────────
    def _export_notable_daughters(
        self, notable_daughters: Dict[str, Any]
    ) -> None:
        path = self.out / "radioisotopos_interesse.csv"
        rows = []
        for nuc, info in notable_daughters.items():
            rows.append({
                "nuclideo": nuc,
                "atividade_Ci": info.get("activity_Bq", 0.0) * _BQ_TO_CI,
                "massa_g": info.get("mass_g", 0.0),
                "meia_vida_h": info.get("half_life_h", 0.0),
                "uso": info.get("use", ""),
                "par_pai": info.get("parent", ""),
                "par_filha": info.get("daughter", ""),
            })
        if rows:
            self._write_csv(path, rows, [
                "nuclideo", "atividade_Ci", "massa_g",
                "meia_vida_h", "uso", "par_pai", "par_filha",
            ])
            self.log.info(
                " ✓ radioisotopos_interesse.csv (%d linhas)", len(rows)
            )
        else:
            self.log.info(" ℹ️ radioisotopos_interesse.csv: nenhum detectado")

    # ── F5: ativação estrutural ─────────────────────────────────────
    def _export_structural_activation(
        self, structural_activation: Dict[str, Any]
    ) -> None:
        path = self.out / "ativacao_estrutural.csv"
        rows = []
        for layer_name, ldata in structural_activation.items():
            if not ldata.get("success"):
                continue
            for nuc, act in ldata.get("activity_Bq", {}).items():
                rows.append({
                    "camada": layer_name,
                    "nuclideo": nuc,
                    "atividade_Ci": act * _BQ_TO_CI,
                })
        if rows:
            self._write_csv(path, rows,
                            ["camada", "nuclideo", "atividade_Ci"])
            self.log.info(
                " ✓ ativacao_estrutural.csv (%d linhas)", len(rows)
            )

    # ── writer ──────────────────────────────────────────────────────
    def _write_csv(
        self, path: Path, rows: List[Dict], fieldnames: List[str]
    ) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k: (
                        f"{v:.12e}"
                        if isinstance(v, float) and k != "time_h"
                        else v
                    )
                    for k, v in row.items()
                })
        self.exported.append(str(path))


# ═══════════════════════════════════════════════════════════════════════════
# ORQUESTRADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class PostProcessor:
    """
    Orquestra todas as engines F1–F5.
    Ponto de entrada do Maestro (Phase F).
    """

    def __init__(
        self,
        output_dir: str = "pipeline_results",
        debug: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir) / "posprocessamento"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.log = _setup_logger("PostProcessor", debug=debug)

    # ------------------------------------------------------------------
    def process(
        self,
        sim_result: Dict[str, Any],
        parser_result: Dict[str, Any],
        geometry_result: Dict[str, Any],
        output_result: Optional[Dict[str, Any]] = None,
        input_file: str = "Input-simulador.txt",
    ) -> Dict[str, Any]:
        """
        Executa Phase F completa.

        Contrato de retorno:
          success, version, cooling_time_h, phase_decay,
          temperature_history, photon_dose, notable_daughters,
          structural_activation, csv_files, errors, metadata
        """
        t_start = time.time()
        result: Dict[str, Any] = {
            "success": False,
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "cooling_time_h": 0.0,
            "phase_decay": {},
            "temperature_history": {"irradiation": [], "cooling": []},
            "photon_dose": {},
            "notable_daughters": {},
            "structural_activation": {},
            "csv_files": [],
            "errors": [],
            "metadata": {},
        }

        try:
            self.log.info("")
            self.log.info("=" * 80)
            self.log.info("▶️ PHASE G: POSPROCESSAMENTO %s", VERSION)
            self.log.info("=" * 80)

            # ── 0. Parâmetros base ───────────────────────────────────
            h5_path = (
                sim_result.get("h5_depletion_path")
                or sim_result.get("depletion_h5")
                or "pipeline_results/depletion_results.h5"
            )

            sim_params = parser_result.get("simulation_parameters", {})
            layers = geometry_result.get("layers", {})
            wafer = geometry_result.get("wafer_geometry", {})
            h5_path_str = str(h5_path)

            # cooling_time_h é sempre usado para decaimento (incluso t=0h)
            cooling_time_h = float(
                sim_params.get("cooling_time_h", sim_params.get("tempo_resfriamento_h", SimulationDefaults.COOLING_TIME_H))
            )
            result["cooling_time_h"] = cooling_time_h

            flux = float(sim_params.get("flux", sim_params.get("fluxo", SimulationDefaults.FLUX)))
            x_cm = float(wafer.get("x_cm", SimulationDefaults.WAFER_SIDE_CM))
            y_cm = float(wafer.get("y_cm", SimulationDefaults.WAFER_SIDE_CM))
            area_cm2 = x_cm * y_cm
            # FIX PP4: total_time_h — buscar em múltiplas fontes por ordem de confiança:
            # 1. sim_result['temporal_params']['total_time_h'] — valor real usado na simulação
            # 2. sim_params['total_time_h']                   — do parser (campo obrigatório V221+)
            # 3. SimulationDefaults.TOTAL_TIME_H (48h)        — fallback apenas se tudo falhar
            # O fallback 48h causava irrad_s 3.5× menor para simulações de 168h.
            _tp = sim_result.get("temporal_params", {}) if isinstance(sim_result, dict) else {}
            total_time_h = float(
                _tp.get("total_time_h")
                or sim_params.get("total_time_h")
                or sim_params.get("tempo_total_h")
                or SimulationDefaults.TOTAL_TIME_H
            )
            if total_time_h == SimulationDefaults.TOTAL_TIME_H:
                self.log.warning(
                    "PP4: total_time_h não encontrado em temporal_params nem sim_params "
                    "→ usando fallback %.1fh. Verifique se TEMPO_TOTAL_H está no input.",
                    total_time_h,
                )
            chain_file = sim_params.get("chain_file", "")

            # Resolver caminho do chain file via NuclearDataPaths
            chain_path: Optional[str] = None
            _chain_cands = [chain_file, str(Path.home() / "nuclear_data" / chain_file)] +                            [str(p) for p in NuclearDataPaths.CHAIN_CANDIDATES]
            for cand in _chain_cands:
                if cand and os.path.exists(cand):
                    chain_path = cand
                    break

            self.log.info(
                "Parâmetros: cooling=%.1fh flux=%.2e area=%.4f cm² chain=%s",
                cooling_time_h, flux, area_cm2,
                os.path.basename(chain_path) if chain_path else "N/A",
            )

            # ── F1: Inventário decaído por camada ────────────────────
            self.log.info("── F1: PyNE decay processor")
            f1_processor = PyneDecayProcessor(self.log)
            inv_reader = FinalInventoryReader(self.log)

            final_inventory = inv_reader.read(h5_path_str)
            phase_decay: Dict[str, Any] = {}

            # Mapear mat_id → nome da camada
            mat_id_to_name: Dict[str, str] = {}
            if isinstance(layers, dict):
                layer_list = list(layers.values())
            else:
                layer_list = list(layers) if layers else []

            for i, lay in enumerate(layer_list):
                name = (
                    lay.get("name")
                    or lay.get("material_name")
                    or f"CAMADA_{i+1}"
                )
                mat_id_to_name[str(i + 1)] = name
                mat_id_to_name[name] = name

            for mat_id, atoms in final_inventory.items():
                layer_name = mat_id_to_name.get(str(mat_id), f"material_{mat_id}")
                if not atoms:
                    phase_decay[layer_name] = {
                        "success": False, "error": "inventário vazio"
                    }
                    continue
                decay_res = f1_processor.process(atoms, cooling_time_h, chain_path)
                phase_decay[layer_name] = decay_res

            result["phase_decay"] = phase_decay

            # Inventário agregado para F3/F4
            all_activity_bq: Dict[str, float] = {}
            all_mass_g: Dict[str, float] = {}
            total_decay_heat_w = 0.0
            total_activity_bq = 0.0

            aggregated_decay = {
                "success": False,
                "activity_Bq": {},
                "mass_g": {},
                "total_activity_Bq": 0.0,
                "total_decay_heat_W": 0.0,
            }

            for lname, dres in phase_decay.items():
                if dres.get("success"):
                    for nuc, act in dres.get("activity_Bq", {}).items():
                        all_activity_bq[nuc] = all_activity_bq.get(nuc, 0.0) + act
                    for nuc, mg in dres.get("mass_g", {}).items():
                        all_mass_g[nuc] = all_mass_g.get(nuc, 0.0) + mg
                    import math as _math
                    _q = dres.get("total_decay_heat_W", 0.0)
                    if isinstance(_q, (int, float)) and _math.isfinite(_q):
                        total_decay_heat_w += _q
                    total_activity_bq += dres.get("total_activity_Bq", 0.0)

            if all_activity_bq:
                aggregated_decay = {
                    "success": True,
                    "activity_Bq": all_activity_bq,
                    "mass_g": all_mass_g,
                    "total_activity_Bq": total_activity_bq,
                    "total_decay_heat_W": total_decay_heat_w,
                }

            self.log.info(
                "F1 concluído: %d camadas A_total=%.3e Ci Q_decay=%.4f W",
                len(phase_decay), total_activity_bq * _BQ_TO_CI, total_decay_heat_w,
            )

            # ── F2: Histórico de temperatura ─────────────────────────
            self.log.info("── F2: TemperatureHistoryBuilder")
            try:
                t_water_k = float(sim_params.get("water_temperature_c", SimulationDefaults.WATER_TEMP_C)) + 273.15
                flow_m3s = float(sim_params.get("water_flow_rate_m3s", SimulationDefaults.WATER_FLOW_M3S))
                t2_builder = TemperatureHistoryBuilder(
                    water_temp_K=t_water_k,
                    flow_rate_m3s=flow_m3s,
                    logger_obj=self.log,
                )

                vol_total_cm3 = float(
                    sum(
                        lay.get("volume_cm3", area_cm2 * 0.05)
                        for lay in layer_list
                    )
                ) if layer_list else area_cm2 * 0.1

                temperature_history = t2_builder.build_from_depletion(
                    h5_path=h5_path_str,
                    cooling_time_h=cooling_time_h,
                    flux_ncm2s=flux,
                    volume_cm3=vol_total_cm3,
                    chain_file_path=chain_path,
                )

            except Exception as exc_f2:
                self.log.warning("F2 falhou (não-crítico): %s", exc_f2)
                temperature_history = {
                    "irradiation": [],
                    "cooling": [],
                    "errors": [str(exc_f2)],
                }

            result["temperature_history"] = temperature_history

            # ── F3: Dose fotônica ─────────────────────────────────────
            self.log.info("── F3: PhotonDoseEstimator")
            try:
                dose_estimator = PhotonDoseEstimator(logger_obj=self.log)
                cool_pts = [
                    i * cooling_time_h / 10.0 for i in range(11)
                ] if cooling_time_h > 0.0 else [0.0]

                photon_dose_list = dose_estimator.estimate(
                    aggregated_decay, cool_pts, chain_path
                )

                result["photon_dose"] = {"agregado": photon_dose_list}
            except Exception as exc_f3:
                self.log.warning("F3 falhou (não-crítico): %s", exc_f3)
                result["photon_dose"] = {}

            # ── F4: Isótopos notáveis ─────────────────────────────────
            self.log.info("── F4: NotableDaughterDetector")
            try:
                daughter_detector = NotableDaughterDetector(logger_obj=self.log)
                result["notable_daughters"] = daughter_detector.detect(aggregated_decay)
            except Exception as exc_f4:
                self.log.warning("F4 falhou (não-crítico): %s", exc_f4)
                result["notable_daughters"] = {}

            # ── F5: Ativação estrutural ───────────────────────────────
            self.log.info("── F5: StructuralActivationSolver")
            structural_activation: Dict[str, Any] = {}
            if HAS_CHAINSOLVE:
                solver = StructuralActivationSolver(logger_obj=self.log)
                irrad_s = total_time_h * 3600.0

                # FIX BUG 6a: identificar camadas combustível pelo inventário H5.
                # F5 deve ativar APENAS camadas estruturais (sem urânio/plutônio acima
                # de traços). Para CAMADA_2 (combustível), o depletor OpenMC já produziu
                # o inventário completo no H5 — reativar via chainsolve seria dupla contagem.
                # Threshold 1e-4 evita falso-positivo por traços de U por difusão numérica.
                _FUEL_UPPER = ("U235", "U238", "U234", "U233", "PU239", "PU241",
                               "922350", "922380", "922340", "942390", "942410")

                def _is_fuel_layer(atoms: Dict[str, float]) -> bool:
                    total = sum(atoms.values())
                    if total <= 0.0:
                        return False
                    fuel = sum(
                        n for nuc, n in atoms.items()
                        if any(f in str(nuc).upper() for f in _FUEL_UPPER)
                    )
                    return (fuel / total) > 1e-4  # > 0.01% fração atômica

                # FIX BUG 6b: ler fluxo por célula do statepoint em vez de φ escalar.
                flux_by_layer: Dict[str, float] = {}
                try:
                    sp_files = sorted(Path(self.output_dir).rglob("statepoint.*.h5"))
                    # Excluir statepoints de calibração (prefixo calib_)
                    sp_files = [p for p in sp_files if "calib_" not in str(p)]
                    if sp_files:
                        with openmc.StatePoint(str(sp_files[-1])) as _sp:
                            _tally = _sp.get_tally(name="flux")
                            _df = _tally.get_pandas_dataframe()
                            _cell_col = next((c for c in ("cell id", "cell") if c in _df.columns), None)
                            _source_rate = sim_result.get("source_rate_calibrated",
                                                          flux * wafer_x_cm * wafer_y_cm)
                            for mat_id in final_inventory:
                                _lname = mat_id_to_name.get(str(mat_id), f"material_{mat_id}")
                                if _cell_col:
                                    _rows = _df[_df[_cell_col] == int(mat_id)]
                                    _phi_per_src = float(_rows["mean"].sum()) if not _rows.empty else 0.0
                                    flux_by_layer[_lname] = _phi_per_src * _source_rate if _phi_per_src > 0 else flux
                                else:
                                    flux_by_layer[_lname] = flux
                        self.log.info("[F5] Fluxo por camada (statepoint): %s",
                                      {k: f"{v:.2e}" for k, v in flux_by_layer.items()})
                except Exception as _exc_sp:
                    self.log.warning("[F5] Não foi possível ler fluxo do statepoint (%s) — φ escalar=%.2e",
                                     _exc_sp, flux)

                def _atoms_to_grams(nuc: str, n_atoms: float) -> float:
                    try:
                        if HAS_OPENMC:
                            M = openmc.data.atomic_mass(nuc)
                        else:
                            digits = "".join(c for c in nuc if c.isdigit())
                            M = float(digits) if digits else 1.0
                    except Exception:
                        digits = "".join(c for c in nuc if c.isdigit())
                        M = float(digits) if digits else 1.0
                    return n_atoms * M / AVOGADRO

                for mat_id, atoms in final_inventory.items():
                    lname = mat_id_to_name.get(str(mat_id), f"material_{mat_id}")

                    # FIX BUG 6a: pular camadas combustível
                    if _is_fuel_layer(atoms):
                        self.log.info(
                            "[F5] Pulando %s — camada combustível (U/Pu > 0.01%%). "
                            "Inventário disponível no depletion_results.h5 via F1.",
                            lname,
                        )
                        structural_activation[lname] = {
                            "success": True, "skipped": True,
                            "reason": "fuel_layer_use_h5", "activity_Bq": {},
                        }
                        continue

                    # FIX BUG 6b: fluxo real da célula
                    phi_cell = flux_by_layer.get(lname, flux)

                    # FIX BUG 6c: temperatura da camada do histórico térmico
                    T_layer = 300.0
                    t_hist = temperature_history.get("irradiation", [])
                    if t_hist and isinstance(t_hist[-1], dict):
                        T_layer = float(t_hist[-1].get(lname, t_hist[-1].get("T_mean", 300.0)))

                    iso_g = {nuc: _atoms_to_grams(nuc, n) for nuc, n in atoms.items() if n > 0.0}
                    sa_res = solver.solve(lname, iso_g, phi_cell, irrad_s, temperature_K=T_layer)
                    structural_activation[lname] = sa_res
            else:
                self.log.info("F5: chainsolve indisponível — pulado")
            result["structural_activation"] = structural_activation

            # ── Exportar CSVs ─────────────────────────────────────────
            self.log.info("── Exportando CSVs")
            exporter = PostProcessorCSVExporter(self.output_dir, self.log)
            exporter.export_all(
                phase_decay=phase_decay,
                temperature_history=temperature_history,
                photon_dose=result["photon_dose"],
                notable_daughters=result["notable_daughters"],
                structural_activation=structural_activation,
            )

            result["csv_files"] = exporter.exported

            # ── Metadata ──────────────────────────────────────────────
            duration = time.time() - t_start
            result["metadata"] = {
                "version": VERSION,
                "duration_s": round(duration, 3),
                "h5_path": h5_path_str,
                "cooling_time_h": cooling_time_h,
                "n_layers": len(layer_list),
                "n_decay_layers": len(phase_decay),
                "total_activity_Bq": total_activity_bq,
                "total_decay_heat_W": total_decay_heat_w,
                "has_pyne": HAS_PYNE,
                "has_chainsolve": HAS_CHAINSOLVE,
                "n_notable_daughters": len(result["notable_daughters"]),
                "n_csv_files": len(result["csv_files"]),
            }

            result["success"] = True
            self.log.info(
                "POSPROCESSAMENTO OK: %.2fs %d csvs %d erros",
                duration, len(result["csv_files"]), len(result["errors"]),
            )

        except Exception as exc:
            err = str(exc)
            self.log.error("PostProcessor.process() falhou: %s", err)
            result["errors"].append(err)
            if self.debug:
                import traceback as _tb
                self.log.debug(_tb.format_exc())

        self.log.info("=" * 80)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# API PÚBLICA — wrapper funcional (para compatibilidade com maestro.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_postprocessing(
    sim_result: Dict[str, Any],
    parser_result: Dict[str, Any],
    geometry_result: Dict[str, Any],
    output_result: Optional[Dict[str, Any]] = None,
    input_file: str = "Input-simulador.txt",
    output_dir: str = "pipeline_results",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Ponto de entrada funcional para o Maestro.

    Equivalente a PostProcessor(output_dir, debug).process(...).
    Mantém retrocompatibilidade com chamadas diretas de módulo.
    """
    pp = PostProcessor(output_dir=output_dir, debug=debug)
    return pp.process(
        sim_result=sim_result,
        parser_result=parser_result,
        geometry_result=geometry_result,
        output_result=output_result,
        input_file=input_file,
    )