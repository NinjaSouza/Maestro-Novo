#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parser.py V220 — Lê Input-simulador.txt e retorna contrato padronizado.

Contrato de retorno parse_simulation_input() → Dict:
  success, version, timestamp, wafer_geometry, layers (List[dict]),
  thermal_materials, simulation_parameters, energy_source, metadata,
  errors, warnings
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EnergySpectrum
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergySpectrum:
    spectrum_type: str
    raw_value:     str
    parsed_data:   Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.spectrum_type, "raw_value": self.raw_value, "data": self.parsed_data}


# ─────────────────────────────────────────────────────────────────────────────
# InputParser V220
# ─────────────────────────────────────────────────────────────────────────────

class InputParserV217:
    """
    Parser V220 — Production Ready.
    Alias InputParserV217 mantido para compatibilidade com importações existentes.
    """

    VERSION = "V220"

    PYNE_ISOTOPES_BASIC = {
        "U235","U238","U234","Pu239","Pu240","Pu241","Th232",
        "Zn64","Zn66","Zn67","Zn68","Zn70",
        "Al27","Mg24","Mg25","Mg26",
        "Fe54","Fe56","Fe57","Fe58",
        "Ni58","Ni60","Ni61","Ni62","Ni64",
        "Cr50","Cr52","Cr53","Cr54",
        "Mo92","Mo94","Mo95","Mo96","Mo97","Mo98","Mo99","Mo100",
        "Si28","Si29","Si30","Cu63","Cu65",
        "H1","H2","O16","O17","O18","C12","C13","N14","N15",
        "Kr83","Kr85","Xe131","Xe133","Xe135",
        "Cs133","Cs137","Ba138","Ba140","I131","I133",
        "Tc99","Sr90","Zr90","Zr91","Zr92","Zr94","Zr96",
        "Mn55","Co59","Nb93",
    }

    _MULTIWORD_NORMS = [
        (re.compile(r"TEMPO\s+DE\s+RESFRIAMENTO_H",  re.IGNORECASE), "TEMPO_RESFRIAMENTO_H"),
        (re.compile(r"TEMPO\s+RESFRIAMENTO\s+H",     re.IGNORECASE), "TEMPO_RESFRIAMENTO_H"),
        (re.compile(r"COOLING\s+TIME\s+H",           re.IGNORECASE), "COOLING_TIME_H"),
        (re.compile(r"SIMULATION\s+MODE",            re.IGNORECASE), "SIMULATION_MODE"),
        (re.compile(r"THERMAL\s+COUPLING",           re.IGNORECASE), "THERMAL_COUPLING"),
        (re.compile(r"USE\s+GPU",                    re.IGNORECASE), "USE_GPU"),
        (re.compile(r"CHAIN\s+FILE",                 re.IGNORECASE), "CHAIN_FILE"),
        (re.compile(r"WATER\s+TEMPERATURE",          re.IGNORECASE), "WATER_TEMPERATURE"),
        (re.compile(r"WATER\s+FLOW\s+RATE",          re.IGNORECASE), "WATER_FLOW_RATE"),
    ]

    _PARAMMAP = {
        "SIMULATION_MODE":         "simulation_mode",
        "NEUTRONS_POR_PASSO":      "nparticles",
        "BATCHES":                 "nbatches",
        "INACTIVE_BATCHES":        "ninactivebatches",
        "TEMPO_TOTAL_H":           "total_time_h",
        "DT_H":                    "dt_h",
        "DT_H_OUTPUT":             "dt_h",           # alias — saída de inventário
        "DT_H_DEPLETION":          "dt_h_depletion", # FIX P1: passo interno do integrador
        "TIMESTEP_ADAPTATIVO":     "timestep_adaptive",
        "COOLING_TIME_H":          "cooling_time_h",
        "TEMPO_RESFRIAMENTO_H":    "cooling_time_h",
        "TEMPO_DE_RESFRIAMENTO_H": "cooling_time_h",
        "FLUXO":                   "flux",
        "FONTE_TEMPERATURA_K":     "source_temp_k",
        "USE_GPU":                 "use_gpu",
        "THERMAL_COUPLING":        "thermal_coupling",
        "WATER_TEMPERATURE":       "water_temperature_c",
        "WATER_FLOW_RATE":         "water_flow_rate_m3s",
        "CHAIN_FILE":              "chain_file",
        "DEPLETION_INTEGRATOR":    "depletion_integrator",   # FIX P1: ex: celi, cecm
        "NORMALIZACAO":            "depletion_normalization", # FIX P1: source-rate / fission-q
    }

    _BOOL_FIELDS          = {"use_gpu", "thermal_coupling", "timestep_adaptive"}
    _INT_FIELDS           = {"nparticles", "nbatches", "ninactivebatches"}
    _STR_FIELDS           = {"simulation_mode", "chain_file", "depletion_integrator", "depletion_normalization"}
    _NUMERIC_FLOAT_FIELDS = {
        "total_time_h", "dt_h", "dt_h_depletion", "flux", "source_temp_k",
        "cooling_time_h", "water_temperature_c", "water_flow_rate_m3s", "source_rate",
    }
    _NUMERIC_INT_FIELDS   = {"nparticles", "nbatches", "ninactivebatches"}

    def __init__(self, debug: bool = False, pyne_enabled: bool = False) -> None:
        self.debug        = debug
        self.pyne_enabled = pyne_enabled
        self.errors:   List[str] = []
        self.warnings: List[str] = []

    # ── API pública ───────────────────────────────────────────────────────────

    def parse_simulation_input(self, filepath: str) -> Dict:
        self.errors = []
        self.warnings = []

        if not os.path.exists(filepath):
            return self._failure("Arquivo não encontrado: " + filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            wafer_geo = self._parse_wafer_geometry(lines)
            x_cm = wafer_geo["x_cm"] if wafer_geo else 1.0
            y_cm = wafer_geo["y_cm"] if wafer_geo else 1.0

            layers_map = self._parse_layers(lines, x_cm=x_cm, y_cm=y_cm) or {}
            sim_params = self._coerce_simulation_parameters(
                self._parse_simulation_parameters(lines)
            )

            # FIX V236: parser NÃO calcula source_rate — isso é responsabilidade
            # exclusiva do settings.py (única fonte de verdade). Parser apenas
            # repassa flux e geometria; settings.py fará flux × area uma única vez.
            # Mantido source_rate=None para backward compat, mas será sobrescrito
            # pelo valor calculado em Phase C.
            flux     = sim_params.get("flux") or 0.0
            area_cm2 = x_cm * y_cm
            sim_params["wafer_x_cm"] = x_cm   # para settings.py usar
            sim_params["wafer_y_cm"] = y_cm   # para settings.py usar
            # sim_params["source_rate"] será definido em settings.py

            energy_spec = self._parse_energy_source(lines)

            if not self._validate_required_fields(wafer_geo, layers_map, sim_params):
                return self._failure("Validação de campos obrigatórios falhou")

            thermal_materials: Dict = {}
            if sim_params.get("thermal_coupling"):
                thermal_materials = self._parse_thermal_fases()

            layers = self._normalize_layers_output(layers_map)
            layers = self._merge_thermal_into_layers(layers, thermal_materials)

            pyne_valid = True
            if self.pyne_enabled:
                pyne_valid = self._validate_isotopes_pyne_list(layers)

            total_mass  = sum(lay.get("total_mass_g", 0.0) for lay in layers)
            cooling_h   = sim_params.get("cooling_time_h") or 0.0
            # FIX V236: source_rate será definido em settings.py (única fonte de verdade).
            # Parser retorna None aqui; maestro preencherá após Phase C.
            source_rate = sim_params.get("source_rate")  # pode ser None nesta fase

            return {
                "success":   True,
                "version":   self.VERSION,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "wafer_geometry":        wafer_geo,
                "layers":                layers,
                "thermal_materials":     thermal_materials,
                "simulation_parameters": sim_params,
                "energy_source": energy_spec.to_dict() if energy_spec else None,
                "metadata": {
                    "n_layers":            len(layers),
                    "parser_version":      self.VERSION,
                    "filepath":            filepath,
                    "has_thermal_data":    bool(thermal_materials),
                    "has_energy_source":   energy_spec is not None,
                    "total_mass_g":        total_mass,
                    "cooling_time_h":      cooling_h,
                    "source_rate":         source_rate,
                    "pyne_support":        self.pyne_enabled,
                    "pyne_isotopes_valid": pyne_valid,
                },
                "errors":   self.errors,
                "warnings": self.warnings,
            }
        except Exception as exc:
            logger.error("Erro ao fazer parse: %s", exc, exc_info=True)
            return self._failure("Erro: " + str(exc))

    # ── Wafer geometry ────────────────────────────────────────────────────────

    def _parse_wafer_geometry(self, lines: List[str]) -> Optional[Dict]:
        x_cm = y_cm = None
        for line in lines:
            lc  = self._clean_line(line)
            tok = lc.split()
            if not tok:
                continue
            key = tok[0].lower()
            if key == "x" and len(tok) >= 2:
                try:
                    x_cm = float(tok[1])
                except ValueError as e:
                    self.errors.append("Erro ao ler 'x': " + str(e))
                    return None
            elif key == "y" and len(tok) >= 2:
                try:
                    y_cm = float(tok[1])
                except ValueError as e:
                    self.errors.append("Erro ao ler 'y': " + str(e))
                    return None
        if x_cm is None:
            self.errors.append("Campo obrigatório 'x' não encontrado")
            return None
        if y_cm is None:
            self.errors.append("Campo obrigatório 'y' não encontrado")
            return None
        return {"x_cm": x_cm, "y_cm": y_cm}

    # ── Layers ────────────────────────────────────────────────────────────────

    def _parse_layers(self, lines: List[str], x_cm: float = 1.0, y_cm: float = 1.0) -> Optional[Dict]:
        try:
            layers: Dict = {}
            cur_layer = cur_layer_num = None
            cur_thickness = None
            cur_isotopes: Dict[str, float] = {}

            def _flush() -> None:
                if cur_layer is not None and cur_thickness is not None:
                    layers[cur_layer] = self._make_layer(
                        number=cur_layer_num, name=cur_layer,
                        thickness_mm=cur_thickness, isotopes=cur_isotopes,
                        x_cm=x_cm, y_cm=y_cm,
                    )

            for i, line in enumerate(lines):
                lc = self._clean_line(line)
                lc_upper = lc.upper()
                if not lc:
                    continue

                if lc_upper.startswith("CAMADA"):
                    _flush()
                    try:
                        tok = lc.split()
                        if len(tok) >= 2:
                            cur_layer_num = int(tok[1])
                        else:
                            m = re.search(r"\d+", lc)
                            cur_layer_num = int(m.group()) if m else 0
                        cur_layer     = "CAMADA_" + str(cur_layer_num)
                        cur_thickness = None
                        cur_isotopes  = {}
                    except (IndexError, ValueError) as e:
                        self.warnings.append("Parse CAMADA linha %d: %s" % (i + 1, e))

                elif lc_upper.startswith("ESPESSURA"):
                    if cur_layer is None:
                        self.errors.append(
                            "ESPESSURA na linha %d fora de bloco CAMADA" % (i + 1)
                        )
                        continue
                    tok = lc.split()
                    if len(tok) >= 2:
                        try:
                            cur_thickness = float(tok[1])
                        except ValueError as e:
                            self.warnings.append("Erro ESPESSURA linha %d: %s" % (i + 1, e))

                elif cur_layer is not None:
                    tok = lc.split()
                    if len(tok) >= 2 and tok[0][0].isalpha():
                        try:
                            mass = float(tok[1])
                            if mass > 0.0:
                                cur_isotopes[tok[0]] = mass
                        except ValueError:
                            pass

            _flush()

            if not layers:
                self.errors.append("Nenhuma camada encontrada no arquivo")
                return None
            return layers
        except Exception as e:
            self.errors.append("Erro ao parsear camadas: " + str(e))
            return None

    def _make_layer(self, number: int, name: str, thickness_mm: float,
                    isotopes: Dict[str, float], x_cm: float = 1.0, y_cm: float = 1.0) -> Dict:
        from config import ValidationLimits
        thickness_cm = thickness_mm / 10.0
        area_cm2     = x_cm * y_cm
        volume_cm3   = area_cm2 * thickness_cm
        total_mass_g = sum(isotopes.values())
        density_gcm3 = total_mass_g / volume_cm3 if volume_cm3 > 0.0 else 0.0
        if density_gcm3 > ValidationLimits().RHO_MAX_GCM3:
            logger.warning("%s rho=%.2f g/cm³ acima do limite físico", name, density_gcm3)
        return {
            "number": number, "name": name,
            "thickness_mm": thickness_mm, "thickness_cm": thickness_cm,
            "area_cm2": area_cm2, "volume_cm3": volume_cm3,
            "density_gcm3": density_gcm3, "total_mass_g": total_mass_g,
            "isotopes": dict(isotopes),
            "fractions_by_mass": self._calculate_fractions(isotopes),
            "thermal_components": [], "material_name": None,
            "initial_temperature_k": 300.0,
        }

    # ── Normalização e fusão ──────────────────────────────────────────────────

    def _normalize_layers_output(self, layers_map: Dict) -> List[Dict]:
        out = []
        for layer_key, layer_data in layers_map.items():
            item = dict(layer_data)
            if not item.get("name"):
                item["name"] = layer_key
            out.append(item)
        out.sort(key=lambda d: int(d.get("number", 10**9)))
        return out

    def _merge_thermal_into_layers(self, layers: List[Dict], thermal_materials: Dict) -> List[Dict]:
        if not thermal_materials:
            return layers
        merged = []
        for layer in layers:
            item     = dict(layer)
            name_key = item.get("name") or "CAMADA_%s" % item.get("number")
            comps    = list(thermal_materials.get(name_key, []))
            item["thermal_components"] = comps
            if len(comps) == 1:
                item["material_name"]         = comps[0].get("material")
                item["initial_temperature_k"] = float(comps[0].get("temperature_k", 300.0))
            elif len(comps) > 1:
                # Componente de maior massa como material térmico representativo
                dominant = max(comps, key=lambda c: float(c.get("mass_g", 0.0)))
                item["material_name"]         = dominant.get("material")
                item["initial_temperature_k"] = float(comps[0].get("temperature_k", 300.0))
            merged.append(item)
        return merged

    # ── Simulation parameters ─────────────────────────────────────────────────

    def _normalize_multiword_keys(self, line: str) -> str:
        for pattern, replacement in self._MULTIWORD_NORMS:
            normalized, n = pattern.subn(replacement, line, count=1)
            if n:
                return normalized
        return line

    def _parse_simulation_parameters(self, lines: List[str]) -> Dict:
        params: Dict[str, Any] = {
            "simulation_mode": None, "nparticles": None, "nbatches": None,
            "ninactivebatches": None, "total_time_h": None, "dt_h": None,
            "dt_h_depletion": None,           # FIX P1: passo interno do integrador [h]
            "depletion_integrator": None,     # FIX P1: ex: 'celi', 'cecm', 'cf4'
            "depletion_normalization": None,  # FIX P1: 'source-rate' | 'fission-q'
            "flux": None, "source_temp_k": None, "use_gpu": False,
            "thermal_coupling": False, "timestep_adaptive": False,
            "cooling_time_h": 0.0, "water_temperature_c": 20.0,
            "water_flow_rate_m3s": 0.0, "chain_file": "",  # FIX P2: sem default hardcoded; settings.py resolve via NuclearDataPaths.CHAIN_CANDIDATES
            "source_rate": None,
        }
        for line in lines:
            lc = self._clean_line(line)
            if not lc:
                continue
            lc = self._normalize_multiword_keys(lc)
            parts = lc.split(None, 1)
            if len(parts) < 2:
                continue
            key_upper = parts[0].upper()
            if key_upper not in self._PARAMMAP:
                continue
            param_key = self._PARAMMAP[key_upper]
            value_str = parts[1].strip()
            try:
                if param_key in self._BOOL_FIELDS:
                    params[param_key] = value_str.lower() in ("true", "yes", "1", "sim", "s")
                elif param_key in self._STR_FIELDS:
                    params[param_key] = value_str.upper() if param_key == "simulation_mode" else value_str
                elif param_key in self._INT_FIELDS:
                    params[param_key] = int(float(value_str))
                else:
                    params[param_key] = float(value_str)
            except ValueError as e:
                self.errors.append("Erro ao converter %s = '%s': %s" % (key_upper, value_str, e))
        return params

    def _coerce_simulation_parameters(self, sim_params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(sim_params)
        for key in self._NUMERIC_FLOAT_FIELDS:
            val = out.get(key)
            if val is not None:
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    out[key] = None
        for key in self._NUMERIC_INT_FIELDS:
            val = out.get(key)
            if val is not None:
                try:
                    out[key] = int(float(val))
                except (TypeError, ValueError):
                    out[key] = None
        for key in self._BOOL_FIELDS:
            val = out.get(key)
            if val is not None and not isinstance(val, bool):
                out[key] = str(val).strip().lower() in {"true", "yes", "1", "sim", "s"}
        return out

    # ── Energia da fonte ──────────────────────────────────────────────────────

    def _parse_energy_source(self, lines: List[str]) -> Optional[EnergySpectrum]:
        for line in lines:
            lc = self._clean_line(line)
            if not lc or not lc.upper().startswith("ENERGIA_DA_FONTE"):
                continue
            parts = lc.split(None, 1)
            if len(parts) < 2:
                self.warnings.append("ENERGIA_DA_FONTE sem valor")
                return None
            raw_value = parts[1].strip()
            spectrum = self._parse_energy_value(raw_value)
            if spectrum:
                return spectrum
            self.errors.append("Erro ao parsear ENERGIA_DA_FONTE: " + raw_value)
            return None
        return None

    def _parse_energy_value(self, value_str: str) -> Optional[EnergySpectrum]:
        v = value_str.strip()
        # Espectro tabulado 252g ORIGEN JEFF-3.0/A
        if v.upper().startswith("ORIGEN252"):
            return self._parse_origen252_spectrum(v)
        # Espectro tabular genérico: TABULAR [E1:p1, E2:p2, ...]
        if v.upper().startswith("TABULAR"):
            return self._parse_tabular_spectrum(v)
        if "+" in v or re.search(r"\be\b", v, re.IGNORECASE):
            return self._parse_hybrid_spectrum(v)
        if v.startswith("[") and v.endswith("]"):
            return self._parse_discrete_spectrum(v)
        func_match = re.match(r"(\w+)\((.+)\)", v)
        if func_match:
            return self._parse_named_distribution(v, func_match)
        _kw = v.upper()
        if _kw in ("MAXWELL", "MAXWELL-BOLTZMANN", "MAXWELLIANO"):
            return EnergySpectrum("maxwell", v, {"distribution": "maxwell", "parameters": {}, "use_source_temp": True})
        if _kw in ("WATT", "FISSAO", "FISSÃO", "FISSION"):
            return EnergySpectrum("watt", v, {"distribution": "watt", "parameters": {"a": 0.988e6, "b": 2.249e-6}, "use_source_temp": False})
        try:
            return EnergySpectrum("single", v, {"energy_ev": float(v)})
        except ValueError:
            pass
        return None

    def _parse_origen252_spectrum(self, value_str: str) -> "EnergySpectrum":
        """
        Espectro tabulado 252 grupos JEFF-3.0/A do ORIGEN.

        Sintaxe:
          ORIGEN252                          → pesos padrão canal de irradiação IEA-R1
          ORIGEN252 w_th=0.80 w_epi=0.15 w_fast=0.05  → pesos customizados
          ORIGEN252 pwr                      → espectro de núcleo PWR (w_th=0.167...)

        PESOS PADRÃO — canal de irradiação de reator de pesquisa (IEA-R1 típico):
          w_th=0.80  w_epi=0.15  w_fast=0.05
          O espectro original do ORIGEN/JEFF-3.0A (w_th=0.167, w_epi=0.212, w_fast=0.621)
          representa núcleo de PWR (espectro rígido), inadequado para canal de irradiação.
          Para o IEA-R1, φ_th/φ_total ≈ 0.8-0.9 (espectro mole/térmico).

        NOTA: as probabilidades geradas são densidades de probabilidade [1/eV] tal que
        ∫PDF·dE = 1, adequadas para openmc.stats.Tabular(interpolation='histogram').
        """
        import math

        # Pesos padrão: canal de irradiação de reator de pesquisa (IEA-R1)
        # Use ORIGEN252 pwr para espectro de núcleo PWR
        W_TH_DEFAULT, W_EPI_DEFAULT, W_FAST_DEFAULT = 0.80, 0.15, 0.05
        W_TH_PWR, W_EPI_PWR, W_FAST_PWR             = 0.1673, 0.2121, 0.6206

        # Detectar preset "pwr"
        is_pwr = bool(re.search(r"\bpwr\b", value_str, re.IGNORECASE))
        if is_pwr:
            W_TH, W_EPI, W_FAST = W_TH_PWR, W_EPI_PWR, W_FAST_PWR
        else:
            W_TH, W_EPI, W_FAST = W_TH_DEFAULT, W_EPI_DEFAULT, W_FAST_DEFAULT

        params: Dict[str, float] = {}
        for m in re.finditer(r"(\w+)\s*=\s*([0-9eE.+\-]+)", value_str):
            params[m.group(1).lower()] = float(m.group(2))
        w_th   = params.get("w_th",   W_TH)
        w_epi  = params.get("w_epi",  W_EPI)
        w_fast = params.get("w_fast", W_FAST)
        total  = w_th + w_epi + w_fast
        w_th, w_epi, w_fast = w_th/total, w_epi/total, w_fast/total

        logger.info(
            "ORIGEN252: w_th=%.4f w_epi=%.4f w_fast=%.4f (%s)",
            w_th, w_epi, w_fast,
            "PWR core" if is_pwr else "canal irradiação IEA-R1"
        )

        # Valores de fluxo por grupo (forma espectral relativa)
        THERMAL_VAL    = 9.815e11
        EPITHERMAL_VAL = 1.806e11
        RAPID_VALS = [
            6.655e4,  6.179e5,  4.108e6,  2.706e7,  1.762e8,  3.208e8,  8.100e8,
            2.035e9,  5.087e9,  1.236e10, 3.112e10, 7.595e10, 1.833e11, 4.362e11,
            1.019e12, 2.323e12, 5.115e12, 3.396e12, 4.519e12, 5.929e12, 7.640e12,
            9.609e12, 1.168e13, 1.349e13, 6.715e12, 2.017e13, 2.388e12, 1.879e12,
            5.802e11, 4.685e11, 1.893e11, 1.575e11, 1.245e11, 5.823e10, 3.177e10,
            3.163e10, 2.247e10, 1.644e10, 3.221e8,
        ]
        N_TH, N_EPI, N_RAP = 27, 186, 39

        def _logmid(e_lo, e_hi, n):
            return [math.exp(math.log(e_lo) + (math.log(e_hi) - math.log(e_lo)) *
                             (i + 0.5) / n) for i in range(n)]

        all_E = _logmid(1e-5, 0.625, N_TH) + _logmid(0.625, 1e5, N_EPI) + _logmid(1e5, 2e7, N_RAP)

        # Reescalar amplitudes por região para respeitar os pesos w_th/w_epi/w_fast
        # phi_região ∝ w_região (independente do número de grupos)
        phi_th_raw   = THERMAL_VAL * N_TH
        phi_epi_raw  = EPITHERMAL_VAL * N_EPI
        phi_fast_raw = sum(RAPID_VALS)
        phi_raw_tot  = phi_th_raw + phi_epi_raw + phi_fast_raw

        scale_th   = (w_th   * phi_raw_tot) / phi_th_raw
        scale_epi  = (w_epi  * phi_raw_tot) / phi_epi_raw
        scale_fast = (w_fast * phi_raw_tot) / phi_fast_raw

        all_phi = (
            [THERMAL_VAL * scale_th]    * N_TH  +
            [EPITHERMAL_VAL * scale_epi] * N_EPI +
            [v * scale_fast for v in RAPID_VALS]
        )

        phi_tot  = sum(all_phi)
        all_prob = [p / phi_tot for p in all_phi]

        return EnergySpectrum("tabular", value_str, {
            "energies_ev":   all_E,
            "probabilities": all_prob,
            "n_points":      len(all_E),
            "source":        "ORIGEN252_JEFF30A",
            "weights":       {"thermal": w_th, "epithermal": w_epi, "fast": w_fast},
            "interpolation": "histogram",
            "preset":        "pwr" if is_pwr else "irradiation_channel",
        })

    def _parse_tabular_spectrum(self, value_str: str) -> "Optional[EnergySpectrum]":
        """
        Espectro tabular genérico: TABULAR [E1:p1, E2:p2, ...]
        Exemplo: TABULAR [0.025:0.5, 1e6:0.3, 2e6:0.2]
        """
        m = re.search(r"\[(.+)\]", value_str)
        if not m:
            logger.error("TABULAR requer lista [E:p, ...]: %s", value_str)
            return None
        result = self._parse_discrete_spectrum("[" + m.group(1) + "]")
        if result is None:
            return None
        return EnergySpectrum("tabular", value_str, {
            "energies_ev":   result.parsed_data["energies_ev"],
            "probabilities": result.parsed_data["probabilities"],
            "n_points":      result.parsed_data["n_points"],
            "source":        "user_tabular",
            "interpolation": "histogram",
        })

    def _parse_discrete_spectrum(self, value_str: str) -> Optional[EnergySpectrum]:
        try:
            inner = value_str[1:-1]
            energies, probs, has_weights = [], [], False
            for token in inner.split(","):
                token = token.strip()
                if ":" in token:
                    has_weights = True
                    e_str, p_str = token.split(":", 1)
                    energies.append(float(e_str.strip()))
                    probs.append(float(p_str.strip()))
                else:
                    energies.append(float(token))
                    probs.append(1.0)
            n = len(energies)
            if not has_weights:
                probs = [1.0 / n] * n
            else:
                total_p = sum(probs)
                if abs(total_p - 1.0) > 1e-6:
                    probs = [p / total_p for p in probs]
            return EnergySpectrum("discrete", value_str, {
                "energies_ev": energies, "n_points": n,
                "probabilities": probs, "has_weights": has_weights,
            })
        except Exception as e:
            logger.error("Erro ao parsear lista discreta: %s", e)
            return None

    def _parse_named_distribution(self, value_str: str, match: re.Match) -> Optional[EnergySpectrum]:
        func_name  = match.group(1).lower()
        params_str = match.group(2)
        params: Dict[str, float] = {}
        try:
            for part in params_str.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    params[k] = float(v[:-1]) if v.endswith("K") else float(v)
        except Exception as e:
            logger.error("Erro ao parsear parâmetros de %s: %s", func_name, e)
            return None
        _TYPE_MAP = {"maxwell": "maxwell", "maxwellian": "maxwell", "watt": "watt", "wigner": "wigner"}
        stype = _TYPE_MAP.get(func_name, func_name + "_distribution")
        return EnergySpectrum(stype, value_str, {"distribution": func_name, "parameters": params})

    def _parse_hybrid_spectrum(self, value_str: str) -> Optional[EnergySpectrum]:
        parts = re.split(r"\s+e\s+", value_str, flags=re.IGNORECASE)
        if len(parts) == 1:
            parts = [p.strip() for p in value_str.split("+")]
        _BARE = {"maxwell": {"distribution": "maxwell", "parameters": {}},
                 "maxwellian": {"distribution": "maxwell", "parameters": {}},
                 "watt": {"distribution": "watt", "parameters": {}}}
        distributions = []
        for part in parts:
            part = part.strip()
            fm = re.match(r"(\w+)\((.+)\)", part)
            if fm:
                sub = self._parse_named_distribution(part, fm)
                if sub:
                    distributions.append(sub.parsed_data)
            elif part.lower() in _BARE:
                distributions.append(dict(_BARE[part.lower()]))
            else:
                try:
                    distributions.append({"energy_ev": float(part)})
                except ValueError:
                    pass
        if not distributions:
            return None
        return EnergySpectrum("hybrid", value_str, {"distributions": distributions, "n_distributions": len(distributions)})

    # ── Thermal fases ─────────────────────────────────────────────────────────

    def _parse_thermal_fases(self) -> Dict:
        thermal: Dict[str, List[Dict]] = {}
        for fases_file in ("Input-fases.txt", "Input-fases-2.txt"):
            if not os.path.exists(fases_file):
                self.warnings.append("Arquivo não encontrado: " + fases_file)
                continue
            try:
                with open(fases_file, "r", encoding="utf-8") as fh:
                    flines = fh.readlines()
                current_block_id = None
                for i, line in enumerate(flines):
                    lc    = self._clean_line(line)
                    parts = lc.split()
                    if not parts:
                        continue
                    p0_up = parts[0].upper()
                    # Formato A: CAMADA_N <mat_name com possíveis espaços> mass [temp]
                    # Tokeniza da direita para suportar nomes com espaços/vírgulas
                    # ex: "CAMADA_1  Aluminum, Alloy 6061-T6  3.31  300"
                    if p0_up.startswith("CAMADA") and p0_up != "CAMADA" and len(parts) >= 3:
                        layer_id  = self._normalize_layer_key(parts[0])
                        remaining = parts[1:]   # tudo após CAMADA_N
                        temp_k    = 300.0
                        mat_end   = len(remaining)
                        try:
                            # último token pode ser temp_k
                            temp_k  = float(remaining[-1])
                            mat_end = len(remaining) - 1
                        except (ValueError, IndexError):
                            pass
                        try:
                            # penúltimo (agora último) é mass_g
                            mass_g  = float(remaining[mat_end - 1])
                            mat_end -= 1
                        except (ValueError, IndexError) as e:
                            self.warnings.append("Erro linha %d em %s: %s" % (i+1, fases_file, e))
                            continue
                        if mat_end < 1:
                            self.warnings.append("Linha %d em %s: material ausente" % (i+1, fases_file))
                            continue
                        mat_name = " ".join(remaining[:mat_end])
                        thermal.setdefault(layer_id, []).append(
                            {"material": mat_name, "mass_g": mass_g, "temperature_k": temp_k}
                        )
                        continue                    # Formato B cabeçalho
                    if p0_up == "CAMADA" and len(parts) >= 2:
                        try:
                            current_block_id = "CAMADA_" + str(int(parts[1]))
                            thermal.setdefault(current_block_id, [])
                        except ValueError as e:
                            self.warnings.append("Parse CAMADA linha %d: %s" % (i+1, e))
                        continue
                    # Formato B conteúdo
                    if current_block_id is not None and len(parts) >= 2:
                        try:
                            mat_name = parts[0]
                            mass_g   = float(parts[1].strip("(g)"))
                            temp_k   = float(parts[2]) if len(parts) >= 3 else 300.0
                            thermal[current_block_id].append(
                                {"material": mat_name, "mass_g": mass_g, "temperature_k": temp_k}
                            )
                        except ValueError:
                            pass
            except Exception as exc:
                self.warnings.append("Erro ao ler " + fases_file + ": " + str(exc))
        return thermal

    # ── Validação ─────────────────────────────────────────────────────────────

    def _validate_required_fields(self, wafer_geo, layers_map, sim_params) -> bool:
        ok = True
        if wafer_geo is None:
            self.errors.append("Wafer geometry não extraído")
            ok = False
        if not layers_map:
            self.errors.append("Nenhuma camada extraída")
            ok = False
        required = ["simulation_mode", "nparticles", "nbatches",
                    "total_time_h", "dt_h", "flux", "source_temp_k"]
        # FIX P3: ninactivebatches removido de required — default 0 é válido para fixed source
        # e o campo pode estar ausente no input sem ser erro
        missing = [f for f in required if sim_params.get(f) is None]
        if missing:
            self.errors.append("Campos obrigatórios ausentes: " + ", ".join(missing))
            ok = False
        return ok

    def _validate_isotopes_pyne_list(self, layers: List[Dict]) -> bool:
        if not self.pyne_enabled:
            return True
        invalid: set = set()
        for layer in layers:
            for iso in layer.get("isotopes", {}).keys():
                iso_base = re.sub(r"\d+$", "", iso)
                if iso not in self.PYNE_ISOTOPES_BASIC and iso_base not in self.PYNE_ISOTOPES_BASIC:
                    invalid.add(iso)
                    self.warnings.append("Isótopo PyNE não reconhecido: %s (camada %s)" % (iso, layer.get("name", "?")))
        return len(invalid) == 0

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _normalize_layer_key(self, raw_key: str) -> str:
        m = re.search(r"\d+", raw_key)
        return "CAMADA_" + m.group() if m else raw_key.upper()

    def _calculate_fractions(self, isotopes: Dict[str, float]) -> Dict[str, float]:
        total = sum(isotopes.values())
        if total == 0.0:
            return {}
        return {k: v / total for k, v in isotopes.items()}

    def _clean_line(self, line: str) -> str:
        return line.split("#")[0].strip()

    def _failure(self, error_msg: str) -> Dict:
        return {
            "success":   False,
            "version":   self.VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error":     error_msg,
            "errors":    self.errors,
            "warnings":  self.warnings,
            "metadata":  {"parser_version": self.VERSION},
        }

    def export_to_json(self, parsed_result: Dict, output_filepath: str) -> bool:
        try:
            with open(output_filepath, "w", encoding="utf-8") as fh:
                json.dump(parsed_result, fh, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error("Erro ao exportar JSON: %s", e)
            return False


# ── Aliases ───────────────────────────────────────────────────────────────────

InputParserV216 = InputParserV217
InputParser     = InputParserV217

__all__ = ["InputParserV217", "InputParserV216", "InputParser", "EnergySpectrum"]
