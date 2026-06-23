#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""output.py V3.4 — Processamento de outputs: CSVs de inventário por material."""

import csv
import functools
import json
import logging
import re
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import openmc
    import openmc.deplete
    import openmc.data
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from config import NuclearDataPaths

AVOGADRO         = 6.022e23
SECONDS_PER_HOUR = 3600.0


def _setup_logger(name: str, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


def _find_chain_file() -> Path:
    """
    Retorna o primeiro chain file encontrado via NuclearDataPaths.

    FIX O1: se nenhum candidato existir, loga erro explícito em vez de
    retornar silenciosamente um path inexistente que causa CSV de atividade
    zerado. O caller (CSVGenerator) deve verificar path.exists().
    """
    for p in NuclearDataPaths.CHAIN_CANDIDATES:
        if p.exists():
            return p
    # Nenhum encontrado — loga e retorna o primeiro para que o caller detecte
    _no_chain_logger = logging.getLogger(__name__)
    _no_chain_logger.error(
        "_find_chain_file: nenhum chain file encontrado. "
        "Candidatos: %s. "
        "CSV de atividade ficará zerado. "
        "Defina CHAIN_FILE no input ou coloque o arquivo em ~/nuclear_data/.",
        [str(p) for p in NuclearDataPaths.CHAIN_CANDIDATES[:3]],
    )
    return NuclearDataPaths.CHAIN_CANDIDATES[0]


# ─────────────────────────────────────────────────────────────────────────────
# ChainFileReader
# ─────────────────────────────────────────────────────────────────────────────

class ChainFileReader:
    """
    Lê decay chain file XML. Use get_chain_reader(path) para instância singleton.
    """

    def __init__(self, chain_file_path: Path, logger: Optional[logging.Logger] = None):
        self.chain_file_path = chain_file_path
        self.logger  = logger or _setup_logger("ChainFileReader")
        self._cache: Dict[str, float] = {}
        self._load_chain()

    def _load_chain(self) -> None:
        if not self.chain_file_path.exists():
            self.logger.warning("Chain file não encontrado: %s", self.chain_file_path)
            return
        try:
            root   = ET.parse(self.chain_file_path).getroot()
            loaded = 0
            for el in root.findall("nuclide"):
                name = el.get("name", "")
                if not name:
                    continue
                hl = el.get("half_life", "")
                if not hl:
                    decay = el.find("decay")
                    if decay is not None:
                        hl = decay.get("half_life", "")
                if not hl:
                    hl_el = el.find("half_life")
                    if hl_el is not None:
                        hl = hl_el.get("value", hl_el.text or "")
                if hl:
                    try:
                        self._cache[name] = float(hl)
                        loaded += 1
                    except ValueError:
                        pass
            self.logger.info("%d isótopos carregados do chain file", loaded)
        except Exception as exc:
            self.logger.error("Erro ao ler chain file: %s", exc)

    def get_half_life(self, isotope: str) -> float:
        return self._cache.get(isotope, 1e99)

    def get_decay_constant(self, isotope: str) -> float:
        hl = self.get_half_life(isotope)
        return 0.0 if hl > 1e18 else float(np.log(2.0) / hl)


@functools.lru_cache(maxsize=8)
def get_chain_reader(chain_file_path: str) -> ChainFileReader:
    """Singleton de ChainFileReader por caminho de arquivo."""
    return ChainFileReader(Path(chain_file_path))


# ─────────────────────────────────────────────────────────────────────────────
# AtomicMassHelper
# ─────────────────────────────────────────────────────────────────────────────

class AtomicMassHelper:
    """Usa openmc.data.atomic_mass() com fallbacks para metaestáveis e numérico."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger          = logger or _setup_logger("AtomicMassHelper")
        self._cache: Dict[str, float] = {}
        self._warnings_count = 0

    def get_atomic_mass(self, isotope: str) -> float:
        if isotope in self._cache:
            return self._cache[isotope]
        norm = self._normalize(isotope)
        if HAS_OPENMC:
            try:
                mass = float(openmc.data.atomic_mass(norm))
                self._cache[isotope] = mass
                return mass
            except Exception:
                pass
            if re.search(r"m\d*$", norm, re.IGNORECASE):
                ground = re.sub(r"m\d*$", "", norm, flags=re.IGNORECASE)
                try:
                    mass = float(openmc.data.atomic_mass(ground))
                    self._cache[isotope] = mass
                    return mass
                except Exception:
                    pass
        m = re.search(r"\d+", norm)
        if m:
            mass = float(m.group())
            self._cache[isotope] = mass
            self._warnings_count += 1
            return mass
        self._cache[isotope] = 1.0
        return 1.0

    @staticmethod
    def _normalize(isotope: str) -> str:
        clean = isotope.replace("-", "").replace("_", "").replace(" ", "").strip()
        match = re.match(r"^([A-Za-z]{1,2})(\d+.*)$", clean)
        if match:
            sym  = match.group(1)
            rest = match.group(2)
            sym  = sym[0].upper() + (sym[1].lower() if len(sym) > 1 else "")
            return sym + rest
        return clean

    def get_warnings_count(self) -> int:
        return self._warnings_count


# ─────────────────────────────────────────────────────────────────────────────
# H5DepletionReader
# ─────────────────────────────────────────────────────────────────────────────

class H5DepletionReader:
    """Leitor de depletion_results.h5 via openmc.deplete.Results."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or _setup_logger("H5DepletionReader")

    def read_depletion_h5(self, h5_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"success": False, "materials": {}, "errors": [], "metadata": {}}
        try:
            h5p = Path(h5_path)
            if not h5p.exists():
                result["errors"].append(f"Arquivo H5 não encontrado: {h5_path}")
                return result
            if not HAS_OPENMC:
                result["errors"].append("openmc não disponível")
                return result
            return self._read_via_openmc_results(h5p.resolve(), result)
        except Exception as exc:
            result["errors"].append(f"Erro ao ler H5: {exc}")
            return result

    def _read_via_openmc_results(self, h5p_abs: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        res = self._try_open_results(str(h5p_abs))
        if res is None:
            res = self._try_open_results_with_xml_path(h5p_abs)
        if res is None:
            result["errors"].append("Não foi possível abrir Results")
            return result
        if len(res) == 0:
            raise ValueError("Arquivo de depletion não contém steps")

        if hasattr(res, "get_times"):
            try:
                times_s = np.asarray(res.get_times(time_units="s"), dtype=float)
            except TypeError:
                times_s = np.asarray(res.get_times("s"), dtype=float)
        else:
            times_s = np.asarray([float(step.time[0]) for step in res], dtype=float)

        if hasattr(res, "nuclides"):
            nuclide_names = list(res.nuclides)
        elif hasattr(res[0], "index_nuc"):
            nuclide_names = list(res[0].index_nuc)
        else:
            raise ValueError("Não foi possível determinar nuclídeos")

        mat_ids = []
        if hasattr(res, "materials"):
            try:
                mat_ids = [str(m) for m in res.materials]
            except Exception:
                pass
        if not mat_ids and hasattr(res[0], "mat_to_ind"):
            mat_ids = [str(k) for k in res[0].mat_to_ind.keys()]
        if not mat_ids and hasattr(res[0], "index_mat"):
            mat_ids = [str(m) for m in res[0].index_mat]
        if not mat_ids:
            raise ValueError("Não foi possível determinar materiais")

        n_t, n_n = int(len(times_s)), int(len(nuclide_names))
        for mat_id in mat_ids:
            mat_data: Dict[str, Any] = {
                "nuclide_names": nuclide_names, "time": times_s,
                "timesteps": n_t, "number": np.zeros((n_t, n_n), dtype=np.float64),
            }
            for j, nuc in enumerate(nuclide_names):
                try:
                    _t, atoms = res.get_atoms(mat_id, nuc, nuc_units="atoms", time_units="s")
                    atoms = np.asarray(atoms, dtype=float)
                    m = min(atoms.size, n_t)
                    mat_data["number"][:m, j] = atoms[:m]
                except Exception:
                    pass
            result["materials"][mat_id] = mat_data

        result["metadata"] = {
            "reader": "openmc.deplete.Results", "materials": mat_ids,
            "n_materials": len(mat_ids), "n_timesteps": n_t, "n_nuclides": n_n,
        }
        result["success"] = True
        return result

    @staticmethod
    def _try_open_results(path: str):
        try:
            return openmc.deplete.Results(path)
        except TypeError:
            pass
        except Exception:
            return None
        if hasattr(openmc.deplete, "ResultsList"):
            rl = openmc.deplete.ResultsList
            if hasattr(rl, "from_hdf5"):
                try:
                    return rl.from_hdf5(path)
                except Exception:
                    return None
        return None

    @staticmethod
    def _try_open_results_with_xml_path(h5p_abs: Path):
        # FIX O3: lógica de restauração clarificada — usar sentinel explícito
        _SENTINEL = object()
        original = _SENTINEL
        try:
            if hasattr(openmc, "config") and "cross_sections" in openmc.config:
                original = openmc.config.get("cross_sections", "")
                openmc.config["cross_sections"] = str(h5p_abs.parent / "cross_sections.xml")
            return openmc.deplete.Results(str(h5p_abs))
        except Exception:
            return None
        finally:
            if original is not _SENTINEL and hasattr(openmc, "config"):
                openmc.config["cross_sections"] = original


# ─────────────────────────────────────────────────────────────────────────────
# CSVGenerator
# ─────────────────────────────────────────────────────────────────────────────

class CSVGenerator:
    """Gerador de CSVs (massa + atividade) por material."""

    def __init__(self, output_dir: str, chain_file_path: Path,
                 top_n_isotopes: Optional[int] = None, logger: Optional[logging.Logger] = None):
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger         = logger or _setup_logger("CSVGenerator")
        self.top_n_isotopes = top_n_isotopes
        # FIX O1: avisar explicitamente se chain file não existe — CSVs de atividade ficarão zerados
        if not chain_file_path.exists():
            self.logger.error(
                "CSVGenerator: chain file não encontrado: %s — "
                "decay_const=0 para todos os nuclídeos → atividade CSV zerada",
                chain_file_path,
            )
        self.chain_reader   = get_chain_reader(str(chain_file_path))
        self.mass_helper    = AtomicMassHelper(logger=self.logger)

    def generate_all(self, h5_data: Dict[str, Any]) -> Dict[str, str]:
        csv_files: Dict[str, str] = {}
        if not h5_data.get("success"):
            return csv_files
        materials = h5_data.get("materials", {})
        if not materials:
            return csv_files
        try:
            for mat_id, mat_data in materials.items():
                path_mass = self._generate_mass_csv_for_material(mat_id, mat_data)
                if path_mass:
                    csv_files[f"mass_material_{mat_id}"] = path_mass
                path_act = self._generate_activity_csv_for_material(mat_id, mat_data)
                if path_act:
                    csv_files[f"activity_material_{mat_id}"] = path_act
        except Exception as exc:
            self.logger.error("Erro ao gerar CSVs: %s", exc)
        return csv_files

    def _select_columns(self, max_per_col: np.ndarray, threshold: float, top_n: Optional[int]) -> np.ndarray:
        above = np.where(max_per_col > threshold)[0]
        if top_n is not None and len(above) > top_n:
            ranked = above[np.argsort(max_per_col[above])[::-1]][:top_n]
            above  = np.sort(ranked)
        return above

    def _generate_mass_csv_for_material(self, mat_id: str, mat_data: Dict) -> Optional[str]:
        try:
            number_data   = mat_data.get("number")
            nuclide_names = mat_data.get("nuclide_names", [])
            time_data     = mat_data.get("time")
            timesteps     = int(mat_data.get("timesteps", 0))
            if number_data is None or not nuclide_names or timesteps <= 0:
                return None

            atomic_masses = np.array([self.mass_helper.get_atomic_mass(iso) for iso in nuclide_names], dtype=float)
            masses        = number_data * (atomic_masses / AVOGADRO)
            cols_to_keep  = self._select_columns(np.max(np.abs(masses), axis=0), 1e-15, self.top_n_isotopes)
            if len(cols_to_keep) == 0:
                return None

            csv_path   = self.output_dir / f"isotope_inventory_mass_material_{mat_id}.csv"
            header     = ["Time_h"] + [nuclide_names[i] for i in cols_to_keep]
            time_hours = np.asarray(time_data, dtype=float) / SECONDS_PER_HOUR if time_data is not None else None

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for t in range(timesteps):
                    row  = [f"{time_hours[t]:.6f}"] if time_hours is not None else [t]
                    row += [f"{masses[t, i]:.12e}" for i in cols_to_keep]
                    writer.writerow(row)
            return str(csv_path)
        except Exception as exc:
            self.logger.error("Erro mass CSV material %s: %s", mat_id, exc)
            return None

    def _generate_activity_csv_for_material(self, mat_id: str, mat_data: Dict) -> Optional[str]:
        try:
            number_data   = mat_data.get("number")
            nuclide_names = mat_data.get("nuclide_names", [])
            time_data     = mat_data.get("time")
            timesteps     = int(mat_data.get("timesteps", 0))
            if number_data is None or not nuclide_names or timesteps <= 0:
                return None

            decay_consts = np.array([self.chain_reader.get_decay_constant(iso) for iso in nuclide_names], dtype=float)
            activities   = number_data * decay_consts
            cols_to_keep = self._select_columns(np.max(np.abs(activities), axis=0), 1e-6, None)
            if len(cols_to_keep) == 0:
                return None

            csv_path   = self.output_dir / f"isotope_inventory_activity_material_{mat_id}.csv"
            header     = ["Time_h"] + [nuclide_names[i] for i in cols_to_keep]
            time_hours = np.asarray(time_data, dtype=float) / SECONDS_PER_HOUR if time_data is not None else None

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for t in range(timesteps):
                    row  = [f"{time_hours[t]:.6f}"] if time_hours is not None else [t]
                    row += [f"{activities[t, i]:.6e}" for i in cols_to_keep]
                    writer.writerow(row)
            return str(csv_path)
        except Exception as exc:
            self.logger.error("Erro activity CSV material %s: %s", mat_id, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class OutputProcessor:
    """
    Processador de outputs — contrato Phase E:
      success, files_written, output_dir, version, error
    """

    VERSION = "V3.4"

    def __init__(self, output_dir: str = "pipeline_results",
                 top_n_isotopes: Optional[int] = None, debug: bool = False):
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_n_isotopes = top_n_isotopes
        self.logger         = _setup_logger("OutputProcessor", debug=debug)
        self.h5_reader      = H5DepletionReader(logger=self.logger)

    def _resolve_chain_file(self, sim_result: Dict[str, Any]) -> Path:
        cf = sim_result.get("chain_file")
        if cf and Path(cf).exists():
            return Path(cf)
        tp = sim_result.get("temporal_params", {})
        cf = tp.get("chain_file", "")
        if cf and Path(cf).exists():
            return Path(cf)
        return _find_chain_file()

    def process_results(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        start  = time.time()
        result = {
            "success": False, "version": self.VERSION,
            "files_written": [], "csv_files": [],
            "output_files": {}, "output_dir": str(self.output_dir),
            "metadata": {}, "error": "", "timestamp": datetime.now().isoformat(),
        }
        try:
            if not isinstance(sim_result, dict):
                raise ValueError("sim_result deve ser Dict")
            if not sim_result.get("success"):
                raise ValueError("sim_result.success = False")

            h5_path = sim_result.get("h5_depletion_path") or sim_result.get("depletion_h5")
            if not h5_path:
                raise ValueError("sim_result não contém 'h5_depletion_path' nem 'depletion_h5'")

            chain_path = self._resolve_chain_file(sim_result)
            csv_gen    = CSVGenerator(
                output_dir=str(self.output_dir), chain_file_path=chain_path,
                top_n_isotopes=self.top_n_isotopes, logger=self.logger,
            )

            h5_data = self.h5_reader.read_depletion_h5(h5_path)
            if not h5_data.get("success"):
                errors = h5_data.get("errors", ["desconhecido"])
                raise ValueError(f"H5 read failed: {errors[0]}")

            csv_files_dict = csv_gen.generate_all(h5_data)
            n_irrad        = len(csv_files_dict)
            all_files      = list(csv_files_dict.values())

            n_cooling    = 0
            cooling_json = sim_result.get("cooling_json")
            if cooling_json and HAS_OPENMC:
                cooled_xml = Path(str(cooling_json)).parent / "materials_cooled.xml"
                if cooled_xml.exists():
                    try:
                        extra = self._generate_postcooling_csvs(
                            cooled_xml=cooled_xml, csv_gen=csv_gen,
                            cooling_time_h=float(sim_result.get("cooling_time_h", 6.0)),
                        )
                        n_cooling = len(extra)
                        for key, path in extra.items():
                            csv_files_dict[key] = path
                            all_files.append(path)
                    except Exception as exc_cool:
                        self.logger.warning("CSVs pós-cooling falhou: %s", exc_cool)

            result["output_files"]  = csv_files_dict
            result["csv_files"]     = all_files
            result["files_written"] = all_files
            result["metadata"]      = {
                "h5_depletion_path":     str(h5_path),
                "n_materials":           len(h5_data.get("materials", {})),
                "csv_files_count":       len(all_files),
                "chain_file_path":       str(chain_path),
                "atomic_mass_warnings":  csv_gen.mass_helper.get_warnings_count(),
                "top_n_isotopes":        self.top_n_isotopes,
                "duration_sec":          round(time.time() - start, 3),
                "version":               self.VERSION,
            }
            result["success"] = True
            return result

        except Exception as exc:
            result["error"] = f"Output falhou: {exc}"
            self.logger.error(result["error"])
            return result

    def _generate_postcooling_csvs(self, cooled_xml: Path, csv_gen: CSVGenerator,
                                    cooling_time_h: float = 6.0) -> Dict[str, str]:
        output: Dict[str, str] = {}
        cooled_mats = openmc.Materials.from_xml(str(cooled_xml))

        for mat in cooled_mats:
            mat_id = str(mat.id)
            vol    = getattr(mat, "volume", None)

            nucs_ad: Dict[str, float] = {}
            try:
                tree = ET.parse(str(cooled_xml))
                for mat_el in tree.getroot().findall("material"):
                    if str(mat_el.get("id", "")) == mat_id:
                        dens_el   = mat_el.find("density")
                        rho_total = 0.0
                        if dens_el is not None:
                            rho_total = float(dens_el.get("value", 0.0))
                            if "g" in dens_el.get("units", ""):
                                rho_total = float(mat.get_atom_density() or 0.0)
                        for nuc_el in mat_el.findall("nuclide"):
                            nname = nuc_el.get("name", "")
                            ao    = float(nuc_el.get("ao", 0.0))
                            if nname and ao > 0 and rho_total > 0:
                                nucs_ad[nname] = ao * rho_total
                        break
            except Exception:
                pass

            if not nucs_ad:
                nucs_ad = mat.get_nuclide_atom_densities()

            if not nucs_ad:
                continue

            nuclides  = sorted(nucs_ad.keys())
            densities = np.array([nucs_ad[n] for n in nuclides], dtype=float)

            if vol and vol > 0.0:
                atoms_total   = densities * float(vol)
                atomic_masses = np.array([csv_gen.mass_helper.get_atomic_mass(n) for n in nuclides], dtype=float)
                masses        = atoms_total * (atomic_masses / AVOGADRO)
                keep_m        = np.where(masses > 1e-30)[0]
                if len(keep_m) > 0:
                    path_mass = self.output_dir / f"isotope_postcooling_mass_material_{mat_id}.csv"
                    with open(path_mass, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["Time_h"] + [nuclides[i] for i in keep_m])
                        w.writerow([f"{cooling_time_h:.6f}"] + [f"{masses[i]:.12e}" for i in keep_m])
                    output[f"postcooling_mass_material_{mat_id}"] = str(path_mass)

            decay_consts = np.array([csv_gen.chain_reader.get_decay_constant(n) for n in nuclides], dtype=float)
            activities   = (densities * float(vol) if vol and vol > 0.0 else densities) * decay_consts
            keep_a       = np.where(activities > 1e-12)[0]
            if len(keep_a) > 0:
                path_act = self.output_dir / f"isotope_postcooling_activity_material_{mat_id}.csv"
                with open(path_act, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Time_h"] + [nuclides[i] for i in keep_a])
                    w.writerow([f"{cooling_time_h:.6f}"] + [f"{activities[i]:.6e}" for i in keep_a])
                output[f"postcooling_activity_material_{mat_id}"] = str(path_act)

        return output


def create_output_processor(output_dir: str = "pipeline_results",
                             top_n_isotopes: Optional[int] = None,
                             debug: bool = False) -> OutputProcessor:
    return OutputProcessor(output_dir=output_dir, top_n_isotopes=top_n_isotopes, debug=debug)
