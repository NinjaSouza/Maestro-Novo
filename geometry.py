#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py — Construção de geometria OpenMC para wafer multicamadas.

VERSÃO DE PRODUÇÃO
------------------
Principais decisões:
1. OpenMC permite 1 material -> N células, portanto NÃO exigimos mais
   bijeção estrita entre materials_dict e cells_dict.
2. cells_dict continua contendo apenas as células do wafer, para tallies,
   depleção e T-N loop; cells_dict_all inclui também as células de água.
3. O material de água ('water_reflector') permanece em openmc_materials,
   pois é usado por células reais da geometria.
4. Água leve recebe S(alpha,beta) opcional ('c_H_in_H2O') quando disponível.
5. Metadados e contratos foram enriquecidos para permitir uso robusto em
   maestro.py, simulation.py e pós-processamento.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmc

from config import ValidationLimits, GeometryLimits

_VL = ValidationLimits()
_GL = GeometryLimits()

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ContractValidator
# ─────────────────────────────────────────────────────────────────────────────

class ContractValidator:
    """Valida o contrato de saída de GeometryBuilder.build()."""

    _REQUIRED = (
        "success",
        "openmc_geometry",
        "openmc_materials",
        "materials_dict",
        "cells_dict",
        "layers",
        "wafer_geometry",
        "metadata",
        "updater",
    )

    @classmethod
    def validate(cls, result: dict) -> Tuple[bool, str]:
        for field_name in cls._REQUIRED:
            if field_name not in result:
                return False, f"Campo obrigatório ausente: '{field_name}'"

        if not isinstance(result.get("success"), bool):
            return False, "'success' deve ser bool"

        if result.get("success"):
            typed_fields = (
                ("materials_dict", dict),
                ("cells_dict", dict),
                ("layers", list),
                ("metadata", dict),
                ("wafer_geometry", dict),
            )
            for field_name, expected_type in typed_fields:
                value = result.get(field_name)
                if not isinstance(value, expected_type):
                    return False, (
                        f"'{field_name}' deve ser {expected_type.__name__}, "
                        f"obteve {type(value).__name__}"
                    )

            if result.get("updater") is None:
                return False, "'updater' é None com success=True"

            openmc_mats = result.get("openmc_materials")
            if openmc_mats is None:
                return False, "'openmc_materials' é None com success=True"

            wafer_cells = result.get("cells_dict", {})
            materials_dict = result.get("materials_dict", {})

            # Regra correta para produção:
            # toda célula do wafer deve possuir material homônimo em materials_dict.
            # Não exigimos o inverso, pois OpenMC permite 1 material -> N células
            # (caso da água/moderador/reflector compartilhado).
            missing_for_cells = set(wafer_cells) - set(materials_dict)
            if missing_for_cells:
                return False, (
                    "Células do wafer sem material correspondente em materials_dict: "
                    f"{missing_for_cells}"
                )

            # Checagem opcional de consistência expandida
            all_cells = result.get("cells_dict_all", {})
            if all_cells and not isinstance(all_cells, dict):
                return False, "'cells_dict_all' deve ser dict quando presente"

        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# SharedSurfaceManager
# ─────────────────────────────────────────────────────────────────────────────

class SharedSurfaceManager:
    """Cache de openmc.ZPlane indexado por posição z."""

    def __init__(self) -> None:
        self._cache: Dict[str, openmc.ZPlane] = {}

    def get_or_create(
        self,
        z: float,
        boundary_type: Optional[str] = None,
    ) -> openmc.ZPlane:
        key = f"{z:.8f}"
        if key not in self._cache:
            plane = openmc.ZPlane(z0=float(z))
            if boundary_type:
                plane.boundary_type = boundary_type
            self._cache[key] = plane
        elif boundary_type and self._cache[key].boundary_type != boundary_type:
            self._cache[key].boundary_type = boundary_type
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
# GeometryUpdater
# ─────────────────────────────────────────────────────────────────────────────

class GeometryUpdater:
    """
    Aplica feedback térmico T-N nos materiais OpenMC em memória.

    Estratégia:
    - Camadas do wafer: temperatura real guardada em _user_temps
    - Material OpenMC recebe temperatura snapped para ponto de biblioteca
    - Água é mantida fora da malha de atualização por padrão, salvo se
      explicitamente incluída no temp_map.
    """

    def __init__(
        self,
        materials_dict: Dict[str, openmc.Material],
        cells_dict: Dict[str, openmc.Cell],
    ) -> None:
        self._materials = materials_dict
        self._cells = cells_dict
        self._update_count = 0
        self._needs_export = False
        self._user_temps: Dict[str, float] = {
            name: getattr(mat, "_temperature_user", mat.temperature or 300.0)
            for name, mat in materials_dict.items()
        }

    def get_user_temperature(self, cell_name: str) -> Optional[float]:
        """Retorna a temperatura real (não snapped) da célula/material."""
        return self._user_temps.get(cell_name)

    def update_temperatures(self, temp_map: Dict[str, float]) -> None:
        self._update_count += 1
        updated = 0

        for cell_name, t_raw in temp_map.items():
            t_k = float(np.clip(float(t_raw), _VL.TEMP_MIN_K, _VL.TEMP_MAX_K))
            if t_k != float(t_raw):
                _log.warning(
                    "'%s': T=%.1f K clampado para [%.0f, %.0f] K",
                    cell_name, float(t_raw), _VL.TEMP_MIN_K, _VL.TEMP_MAX_K
                )

            if cell_name in self._materials:
                mat = self._materials[cell_name]
                self._user_temps[cell_name] = t_k
                mat.temperature = GeometryBuilder._snap_temperature(t_k)
                updated += 1
            else:
                _log.warning("'%s' não encontrado em materials_dict", cell_name)

        if updated > 0:
            self._needs_export = True

    def mark_exported(self) -> None:
        self._needs_export = False

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def needs_export(self) -> bool:
        return self._needs_export


# ─────────────────────────────────────────────────────────────────────────────
# GeometryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class GeometryBuilder:
    """Constrói geometria OpenMC completa para wafer multicamadas."""

    VERSION = "V222"

    WATER_AXIAL_CM: float = 5.0
    WATER_LATERAL_CM: float = 10.0

    _LIB_TEMPS_K: Tuple[float, ...] = (
        250.0, 294.0, 600.0, 900.0, 1200.0, 2500.0
    )

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._surface_mgr = SharedSurfaceManager()
        self._errors: List[str] = []

    def build(self, parser_result: dict) -> dict:
        self._errors = []

        if not parser_result.get("success"):
            return self._fail(
                "parser_result.success=False: "
                + str(parser_result.get("error", ""))
            )

        try:
            wafer_geom = self._extract_wafer_geometry(parser_result)
            layers = self._normalize_layers(parser_result["layers"])

            z = 0.0
            enriched = []
            for lay in layers:
                thick_cm = self._thick_cm(lay)
                lay_enriched = dict(lay)
                lay_enriched["zmin"] = round(z, 9)
                lay_enriched["zmax"] = round(z + thick_cm, 9)
                lay_enriched["area_cm2"] = wafer_geom["area_cm2"]
                enriched.append(lay_enriched)
                z += thick_cm
            layers = enriched

            self._validate_layers(layers)
            self._validate_nanoscale(layers)

            materials_dict, openmc_mats = self._build_materials(layers, wafer_geom)
            cells_dict_all, openmc_geometry = self._build_geometry(
                layers, materials_dict, openmc_mats, wafer_geom
            )

            water_cells = {
                k: v for k, v in cells_dict_all.items()
                if k.startswith("water_")
            }
            wafer_cells = {
                k: v for k, v in cells_dict_all.items()
                if not k.startswith("water_")
            }

            updater = GeometryUpdater(materials_dict, wafer_cells)

            result = {
                "success": len(self._errors) == 0,
                "openmc_geometry": openmc_geometry,
                "openmc_materials": openmc_mats,
                "materials_dict": materials_dict,
                "cells_dict": wafer_cells,
                "cells_dict_all": cells_dict_all,
                "water_cells": water_cells,
                "wafer_geometry": wafer_geom,
                "layers": layers,
                "errors": list(self._errors),
                "water_geometry": {
                    "axial_cm": self.WATER_AXIAL_CM,
                    "lateral_cm": self.WATER_LATERAL_CM,
                    "total_z_cm": (
                        wafer_geom["total_thickness_cm"] + 2.0 * self.WATER_AXIAL_CM
                    ),
                    "total_x_cm": wafer_geom["x_cm"] + 2.0 * self.WATER_LATERAL_CM,
                    "total_y_cm": wafer_geom["y_cm"] + 2.0 * self.WATER_LATERAL_CM,
                },
                "metadata": {
                    "version": self.VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "n_layers": len(layers),
                    "n_materials_total": len(materials_dict),
                    "n_materials_wafer": len(wafer_cells),
                    "n_cells_wafer": len(wafer_cells),
                    "n_cells_total": len(cells_dict_all),
                    "n_water_cells": len(water_cells),
                    "n_surfaces": len(self._surface_mgr),
                    "area_cm2": wafer_geom["area_cm2"],
                    "total_thickness_cm": wafer_geom["total_thickness_cm"],
                    "water_axial_cm": self.WATER_AXIAL_CM,
                    "water_lateral_cm": self.WATER_LATERAL_CM,
                    "shared_materials_allowed": True,
                },
                "updater": updater,
            }

            ok, msg = ContractValidator.validate(result)
            if not ok:
                return self._fail(msg)

            return result

        except Exception as exc:
            _log.exception("GeometryBuilder.build() — exceção: %s", exc)
            return self._fail(str(exc))

    # ── Normalização ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_layers(raw) -> list:
        if isinstance(raw, dict):
            items = [lay for lay in raw.values() if isinstance(lay, dict)]
        elif isinstance(raw, (list, tuple)):
            items = [lay for lay in raw if isinstance(lay, dict)]
        else:
            return []

        if items and all("number" in lay for lay in items):
            items = sorted(items, key=lambda x: int(x["number"]))
        return items

    @staticmethod
    def _thick_cm(lay: dict) -> float:
        if lay.get("thickness_cm") is not None:
            return float(lay["thickness_cm"])
        return float(lay.get("thickness_mm", 1.0)) / 10.0

    def _extract_wafer_geometry(self, parser_result: dict) -> dict:
        wg = parser_result.get("wafer_geometry", {})
        x_cm = float(wg.get("x_cm") or wg.get("x") or 1.69)
        y_cm = float(wg.get("y_cm") or wg.get("y") or 1.69)
        water_temp_k = float(wg.get("water_temp_k", 300.0))

        layers = self._normalize_layers(parser_result["layers"])
        total_cm = sum(self._thick_cm(lay) for lay in layers)

        return {
            "x_cm": x_cm,
            "y_cm": y_cm,
            "area_cm2": x_cm * y_cm,
            "total_thickness_cm": total_cm,
            "water_temp_k": water_temp_k,
        }

    # ── Validações ────────────────────────────────────────────────────────

    def _validate_layers(self, layers: list) -> None:
        if not layers:
            raise ValueError("Lista de camadas vazia.")

        for i, lay in enumerate(layers):
            if self._thick_cm(lay) <= 0.0:
                raise ValueError(
                    f"Camada '{self._layer_name(lay, i)}': thickness <= 0"
                )

    def _validate_nanoscale(self, layers: list) -> None:
        for i, lay in enumerate(layers):
            thick_cm = self._thick_cm(lay)
            if thick_cm < _GL.NANOSCALE_MIN_CM:
                _log.warning(
                    "NANOSCALE: '%s' esp=%.2e cm < %.2e cm",
                    self._layer_name(lay, i),
                    thick_cm,
                    _GL.NANOSCALE_MIN_CM,
                )

    # ── Materiais ─────────────────────────────────────────────────────────

    @staticmethod
    def _snap_temperature(t_k: float) -> float:
        """Retorna o ponto de temperatura de biblioteca mais próximo."""
        return min(GeometryBuilder._LIB_TEMPS_K, key=lambda t: abs(t - t_k))

    def _build_materials(
        self,
        layers: list,
        wafer_geom: dict,
    ) -> Tuple[Dict[str, openmc.Material], openmc.Materials]:
        area_cm2 = wafer_geom["area_cm2"]
        materials_dict: Dict[str, openmc.Material] = {}
        mat_list: List[openmc.Material] = []

        for i, lay in enumerate(layers):
            cell_name = self._layer_name(lay, i)
            thick_cm = self._thick_cm(lay)
            volume_cm3 = area_cm2 * thick_cm

            if volume_cm3 <= 0.0:
                raise ValueError(
                    f"'{cell_name}': volume_cm3={volume_cm3:.4e} <= 0"
                )

            density = self._density_from_layer(lay, volume_cm3)

            if density <= 0.0:
                cell_name_up = cell_name.upper()
                fissile_tokens = ("UAL", "UO2", "U_ME", "FUEL", "COMBUST", "ALVO", "TARGET")
                is_fissile = any(tok in cell_name_up for tok in fissile_tokens)
                if is_fissile:
                    raise ValueError(
                        f"'{cell_name}': densidade calculada = {density:.4e} g/cm³. "
                        "Camada combustível com densidade zero indica erro no input "
                        "(massas zeradas ou ausentes). Verifique as massas isotópicas."
                    )

                _log.warning(
                    "'%s': density=%.4e ≤ 0 → usando 1.0 g/cm³ "
                    "(camada estrutural; se for combustível, corrija o input)",
                    cell_name, density,
                )
                density = 1.0

            if density > _VL.RHO_MAX_GCM3:
                msg = (
                    f"DENSIDADE IMPOSSÍVEL '{cell_name}': ρ={density:.2f} g/cm³ "
                    f"> {_VL.RHO_MAX_GCM3:.0f} g/cm³"
                )
                _log.error(msg)
                self._errors.append(msg)

            t_k = float(np.clip(
                float(
                    lay.get("temperature_k")
                    or lay.get("initial_temp_k")
                    or lay.get("temperature")
                    or 300.0
                ),
                _VL.TEMP_MIN_K,
                _VL.TEMP_MAX_K,
            ))
            t_snapped = self._snap_temperature(t_k)

            if abs(t_snapped - t_k) > 1.0:
                _log.debug(
                    "'%s': T=%.1f K → snapped para %.1f K",
                    cell_name, t_k, t_snapped
                )

            mat = openmc.Material(name=cell_name)
            mat.set_density("g/cm3", density)
            mat.volume = volume_cm3
            mat.depletable = True
            mat.temperature = t_snapped
            mat._temperature_user = t_k

            self._add_nuclides(mat, lay, cell_name)

            materials_dict[cell_name] = mat
            mat_list.append(mat)

        return materials_dict, openmc.Materials(mat_list)

    @staticmethod
    def _density_from_layer(lay: dict, volume_cm3: float) -> float:
        for key in ("density_gcm3", "density_g_cm3", "density", "rho"):
            val = lay.get(key)
            if val is not None and float(val) > 0.0:
                return float(val)

        for key in ("total_mass_g", "mass_total_g", "massa_total_g"):
            val = lay.get(key)
            if val is not None and float(val) > 0.0 and volume_cm3 > 0.0:
                return float(val) / volume_cm3

        nuc_map = lay.get("isotopes") or lay.get("fractions_by_mass") or {}
        if isinstance(nuc_map, dict) and nuc_map:
            total_g = sum(float(v) for v in nuc_map.values())
            if total_g > 0.0 and volume_cm3 > 0.0:
                return total_g / volume_cm3

        return 0.0

    @staticmethod
    def _add_nuclides(mat: openmc.Material, lay: dict, cell_name: str) -> int:
        fbm = lay.get("fractions_by_mass")
        iso = lay.get("isotopes")
        frc = lay.get("fractions")

        if fbm and isinstance(fbm, dict):
            nuc_map = {str(k): float(v) for k, v in fbm.items()}
        elif iso and isinstance(iso, dict):
            nuc_map = {str(k): float(v) for k, v in iso.items()}
        elif iso and isinstance(iso, list) and frc and isinstance(frc, list):
            nuc_map = {str(k): float(v) for k, v in zip(iso, frc)}
        else:
            _log.warning("'%s': nenhum formato de nuclídeos reconhecido", cell_name)
            return 0

        total = sum(nuc_map.values())
        if total <= 0.0:
            return 0

        n_added = 0
        for nuc_raw, val in nuc_map.items():
            frac = val / total
            if frac < 1.0e-14:
                continue
            try:
                mat.add_nuclide(nuc_raw.replace("-", "").strip(), frac, "wo")
                n_added += 1
            except Exception as exc:
                _log.warning(
                    "'%s': add_nuclide('%s') falhou: %s",
                    cell_name, nuc_raw, exc
                )
        return n_added

    # ── Água e geometria ───────────────────────────────────────────────────

    def _build_water_material(
        self,
        wafer_geom: dict,
    ) -> openmc.Material:
        t_water_k_real = float(wafer_geom.get("water_temp_k", 300.0))
        t_water_k = self._snap_temperature(t_water_k_real)

        mat_water = openmc.Material(name="water_reflector")
        mat_water.add_nuclide("H1", 2.0 / 3.0, "ao")
        mat_water.add_nuclide("O16", 1.0 / 3.0, "ao")
        mat_water.set_density("g/cm3", 0.9982)
        mat_water.depletable = False
        mat_water.temperature = t_water_k
        mat_water._temperature_user = t_water_k_real

        # Água térmica leve — melhora física na faixa térmica quando os dados SAB
        # estiverem disponíveis na biblioteca carregada pelo OpenMC.
        try:
            mat_water.add_s_alpha_beta("c_H_in_H2O")
            _log.info("water_reflector: S(a,b) 'c_H_in_H2O' habilitado")
        except Exception as exc:
            _log.warning(
                "water_reflector: não foi possível habilitar S(a,b) c_H_in_H2O: %s",
                exc,
            )

        return mat_water

    def _build_geometry(
        self,
        layers: list,
        materials_dict: Dict[str, openmc.Material],
        openmc_mats: openmc.Materials,
        wafer_geom: dict,
    ) -> Tuple[Dict[str, openmc.Cell], openmc.Geometry]:
        """
        Layout:
        [vacuum] | water_front | wafer multicamada | water_back | [vacuum]
        + water_lateral ao redor do wafer na faixa axial do wafer.
        """

        x_cm = wafer_geom["x_cm"]
        y_cm = wafer_geom["y_cm"]
        n_layers = len(layers)
        total_wafer_cm = wafer_geom["total_thickness_cm"]

        dax = self.WATER_AXIAL_CM
        dlat = self.WATER_LATERAL_CM

        mat_water = self._build_water_material(wafer_geom)

        if "water_reflector" not in materials_dict:
            materials_dict["water_reflector"] = mat_water
            openmc_mats.append(mat_water)

        xmin_ext = openmc.XPlane(x0=-(x_cm / 2.0 + dlat), boundary_type="vacuum")
        xmax_ext = openmc.XPlane(x0=+(x_cm / 2.0 + dlat), boundary_type="vacuum")
        ymin_ext = openmc.YPlane(y0=-(y_cm / 2.0 + dlat), boundary_type="vacuum")
        ymax_ext = openmc.YPlane(y0=+(y_cm / 2.0 + dlat), boundary_type="vacuum")

        z_ext_bot = openmc.ZPlane(z0=-dax, boundary_type="vacuum")
        z_wafer_bot = openmc.ZPlane(z0=0.0)
        z_wafer_top = openmc.ZPlane(z0=total_wafer_cm)
        z_ext_top = openmc.ZPlane(z0=total_wafer_cm + dax, boundary_type="vacuum")

        xmin_waf = openmc.XPlane(x0=-x_cm / 2.0)
        xmax_waf = openmc.XPlane(x0=+x_cm / 2.0)
        ymin_waf = openmc.YPlane(y0=-y_cm / 2.0)
        ymax_waf = openmc.YPlane(y0=+y_cm / 2.0)

        self._surface_mgr.clear()

        cells_dict: Dict[str, openmc.Cell] = {}
        z_current = 0.0

        for i, lay in enumerate(layers):
            cell_name = self._layer_name(lay, i)
            z_next = z_current + self._thick_cm(lay)

            z_bot = self._surface_mgr.get_or_create(z_current)
            z_top = self._surface_mgr.get_or_create(z_next)

            cell = openmc.Cell(
                name=cell_name,
                fill=materials_dict[cell_name],
                region=(
                    +xmin_waf & -xmax_waf &
                    +ymin_waf & -ymax_waf &
                    +z_bot & -z_top
                ),
            )
            cells_dict[cell_name] = cell
            z_current = z_next

        delta = abs(z_current - total_wafer_cm)
        if delta > _GL.GAP_TOLERANCE_CM:
            _log.warning(
                "GAPS: empilhado=%.10f cm != esperado=%.10f cm (delta=%.2e)",
                z_current, total_wafer_cm, delta
            )

        wafer_xy = +xmin_waf & -xmax_waf & +ymin_waf & -ymax_waf

        water_front = openmc.Cell(
            name="water_front",
            fill=mat_water,
            region=(
                +xmin_ext & -xmax_ext &
                +ymin_ext & -ymax_ext &
                +z_ext_bot & -z_wafer_bot
            ),
        )
        water_front.volume = (x_cm + 2*dlat) * (y_cm + 2*dlat) * dax
        cells_dict["water_front"] = water_front

        water_back = openmc.Cell(
            name="water_back",
            fill=mat_water,
            region=(
                +xmin_ext & -xmax_ext &
                +ymin_ext & -ymax_ext &
                +z_wafer_top & -z_ext_top
            ),
        )
        water_back.volume = (x_cm + 2*dlat) * (y_cm + 2*dlat) * dax
        cells_dict["water_back"] = water_back

        water_lateral = openmc.Cell(
            name="water_lateral",
            fill=mat_water,
            region=(
                +xmin_ext & -xmax_ext &
                +ymin_ext & -ymax_ext &
                +z_wafer_bot & -z_wafer_top &
                ~wafer_xy
            ),
        )
        water_lateral.volume = ((x_cm + 2*dlat) * (y_cm + 2*dlat) - x_cm * y_cm) * total_wafer_cm
        cells_dict["water_lateral"] = water_lateral

        universe = openmc.Universe(cells=list(cells_dict.values()))
        geometry = openmc.Geometry(universe)

        _log.info(
            "_build_geometry: %d camadas wafer + 3 células de água "
            "(front=%.1fcm, back=%.1fcm, lateral=%.1fcm)",
            n_layers, dax, dax, dlat
        )

        return cells_dict, geometry

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _layer_name(lay: dict, idx: int) -> str:
        return str(
            lay.get("name")
            or lay.get("material_name")
            or lay.get("material")
            or f"layer_{idx + 1}"
        )

    @staticmethod
    def _fail(msg: str) -> dict:
        return {
            "success": False,
            "error": msg,
            "openmc_geometry": None,
            "openmc_materials": None,
            "materials_dict": {},
            "cells_dict": {},
            "cells_dict_all": {},
            "water_cells": {},
            "wafer_geometry": {},
            "layers": [],
            "errors": [msg],
            "metadata": {"version": GeometryBuilder.VERSION},
            "updater": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry(parser_result: dict, debug: bool = False) -> dict:
    return GeometryBuilder(debug=debug).build(parser_result)