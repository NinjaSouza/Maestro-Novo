#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source_calibration.py — Calibração da intensidade da fonte para reproduzir fluxo experimental.

Módulo V246 — CORREÇÃO _get_calibration_volume: usa filt.bins em vez de filt.cells

CONTRATO FÍSICO:
  Quando FLUXO + espectro são fornecidos no input, o simulador executa uma
  etapa de calibração para encontrar source_rate que reproduza o fluxo-alvo
  experimental na face do alvo.

ALGORITMO:
  1. Constrói modelo OpenMC curto com geometria real COMPLETA
  2. Adiciona tally de fluxo na região receptora (primeira camada do alvo)
  3. Executa simulação curta com source_rate inicial de referência
  4. Mede fluxo obtido ϕ_medido (em n·cm/src por partícula-fonte)
  5. Converte: ϕ_físico = ϕ_medido × source_rate / volume_regiao
  6. Atualiza: sr_novo = sr_velho × (ϕ_alvo / ϕ_físico)
  7. Repete até |ϕ_alvo - ϕ_físico| / ϕ_alvo < tolerância

RESULTADO:
  source_rate calibrado que reproduz o FLUXO experimental dentro da tolerância.
  Este valor é congelado e usado em toda a depleção subsequente.
  
FIXES V246:
  - _get_calibration_volume() agora usa getattr(filt, 'bins', getattr(filt, 'cells', []))
    para compatibilidade com OpenMC >= 0.13 onde o atributo mudou de 'cells' para 'bins'
  - Log detalhado de células e volumes para debug
  
FIXES V245 (OPENMC 0.15.3):
  - CellFilter agora usa IDs inteiros, não objetos cell (compatibilidade OpenMC 0.15.3)
  - get_values() convertido explicitamente com np.asarray().ravel() para garantir array 1D
  - Log detalhado do tipo/valor RAW retornado por get_values() para debug
  - Log de exceções completas com stack trace (exc_info=True)
  - Extração explícita de cell.id antes de criar CellFilter
  
FIXES V242:
  - EXTRAÇÃO DIRETA via tally.get_values() como fallback quando pandas falha
  - get_values() retorna array numpy limpo, evitando erro "setting an array element"
  - Log detalhado de shape/dtype do array retornado por get_values()
  - Mensagens de erro melhoradas com scores e filters do tally
  
FIXES V241:
  - Extração robusta de fluxo do tally pandas DataFrame para arrays aninhados
  - Implementação _safe_extract_flux usando np.asarray().ravel()
  - Log detalhado de dtypes do DataFrame para debug
  - Validação explícita de fluxo > 0 antes de prosseguir
  
FIXES V240:
  - Exporta TODOS os XMLs (geometry, materials, settings, tallies) no mesmo diretório
  - Converte corretamente tally por partícula-fonte para fluxo físico [n/cm²/s]
  - Usa chaves corretas do contrato geometry.py (cellsdict, não cells_dict)
  - Identifica primeira camada corretamente sem filtro desnecessário
  - Posiciona fonte baseada nos metadados reais da geometria
  - Usa área da face do alvo para chute inicial, não área expandida
  - Log de atualização corrigido para mostrar valores reais
  - Integração completa com openmc.Model para exportação consistente
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import openmc
    _OPENMC_OK = True
except ImportError:
    _OPENMC_OK = False

from config import (
    SourceCalibrationConfig,
    GeometryContract,
    PhysicsConstants,
    ValidationLimits,
)

logger = logging.getLogger(__name__)

_EV_TO_J = PhysicsConstants.EV_TO_J


# ─────────────────────────────────────────────────────────────────────────────
# Resultado da calibração
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Resultado da calibração da fonte."""
    success: bool
    flux_target: float          # Fluxo-alvo experimental [n/cm²/s]
    flux_achieved: float        # Fluxo medido na convergência [n/cm²/s]
    source_rate_calibrated: float  # source_rate final calibrado [n/s]
    source_rate_initial: float     # source_rate inicial [n/s]
    n_iterations: int           # Número de iterações executadas
    converged: bool             # True se atingiu tolerância
    error_relative_final: float # Erro relativo final |ϕ_alvo - ϕ_medido| / ϕ_alvo
    tally_used: str             # Nome do tally usado para calibração
    calibration_region_volume_cm3: float = 0.0  # Volume da região de medição
    iterations_history: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "flux_target_n_cm2_s": self.flux_target,
            "flux_achieved_n_cm2_s": self.flux_achieved,
            "source_rate_calibrated_n_s": self.source_rate_calibrated,
            "source_rate_initial_n_s": self.source_rate_initial,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "error_relative_final": self.error_relative_final,
            "tally_used": self.tally_used,
            "calibration_region_volume_cm3": self.calibration_region_volume_cm3,
            "iterations_history": self.iterations_history,
            "error_message": self.error_message,
            "timestamp": datetime.now().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SourceCalibrator
# ─────────────────────────────────────────────────────────────────────────────

class SourceCalibrator:
    """
    Calibra a intensidade da fonte para reproduzir fluxo experimental.
    
    Uso:
      calibrator = SourceCalibrator(flux_target, geometry_result, energy_source)
      result = calibrator.run()
      source_rate = result.source_rate_calibrated
    
    FIX V241:
      - Extração robusta de fluxo com np.asarray().ravel() para evitar erro numpy
      - Log de dtypes do DataFrame para debug de arrays aninhados
    
    FIX V239:
      - Recebe openmc.Geometry e openmc.Materials reais
      - Exporta todos XMLs no mesmo diretório temporário
      - Converte tally por partícula-fonte para fluxo físico corretamente
      - Usa chaves do contrato atual (cellsdict, water_front, etc.)
    """
    
    VERSION = "V243"
    
    def __init__(
        self,
        flux_target: float,                    # Fluxo-alvo experimental [n/cm²/s]
        geometry_result: Dict,                 # Resultado de geometry.py
        energy_source: Dict,                   # Espectro de energia da fonte
        wafer_x_cm: float,                     # Dimensão x do alvo [cm]
        wafer_y_cm: float,                     # Dimensão y do alvo [cm]
        water_lateral_cm: float,               # Água lateral [cm]
        water_axial_cm: float = 5.0,           # Água axial [cm]
        config: Optional[SourceCalibrationConfig] = None,
        debug: bool = False,
    ) -> None:
        self.flux_target = float(flux_target)
        self.geometry_result = geometry_result
        self.energy_source = energy_source or {}
        self.wafer_x_cm = float(wafer_x_cm)
        self.wafer_y_cm = float(wafer_y_cm)
        self.water_lateral_cm = float(water_lateral_cm)
        self.water_axial_cm = float(water_axial_cm)
        self.config = config or SourceCalibrationConfig()
        self.debug = debug
        
        # FIX BUG 2: source_box deve ter exatamente a dimensão da face do alvo.
        self.source_x_cm = self.wafer_x_cm
        self.source_y_cm = self.wafer_y_cm
        self.source_area_cm2 = self.source_x_cm * self.source_y_cm
        self.target_face_area_cm2 = self.wafer_x_cm * self.wafer_y_cm
        
        # Source rate inicial estimado: flux × area_do_alvo (não área expandida)
        self.source_rate_initial = self.flux_target * self.target_face_area_cm2
        
        # História de iterações
        self._history: List[Dict[str, Any]] = []
        
        # Volume da região de calibração (preenchido após identificação da célula)
        self._calibration_volume_cm3 = 0.0
        
        logger.info(
            "SourceCalibrator V240 initialized: flux_target=%.4e n/cm²/s, "
            "target_area=%.4f cm², source_area=%.4f cm², source_rate_initial=%.4e n/s",
            self.flux_target, self.target_face_area_cm2, self.source_area_cm2, 
            self.source_rate_initial,
        )
    
    def run(self) -> CalibrationResult:
        """Executa a calibração da fonte."""
        if not _OPENMC_OK:
            return CalibrationResult(
                success=False,
                flux_target=self.flux_target,
                flux_achieved=0.0,
                source_rate_calibrated=self.source_rate_initial,
                source_rate_initial=self.source_rate_initial,
                n_iterations=0,
                converged=False,
                error_relative_final=1.0,
                tally_used=self.config.CALIBRATION_TALLY_NAME,
                error_message="OpenMC não disponível",
            )
        
        if not self.config.ENABLE_CALIBRATION:
            logger.warning("Calibração desabilitada — usando source_rate inicial")
            return CalibrationResult(
                success=True,
                flux_target=self.flux_target,
                flux_achieved=self.flux_target,  # assume perfeito sem calibração
                source_rate_calibrated=self.source_rate_initial,
                source_rate_initial=self.source_rate_initial,
                n_iterations=0,
                converged=True,
                error_relative_final=0.0,
                tally_used=self.config.CALIBRATION_TALLY_NAME,
                calibration_region_volume_cm3=0.0,
            )
        
        source_rate_current = self.source_rate_initial
        source_rate_previous = source_rate_current  # Para log correto
        
        for iteration in range(self.config.MAX_ITERATIONS):
            logger.info(
                "Calibração iteração %d/%d: source_rate=%.4e n/s",
                iteration + 1, self.config.MAX_ITERATIONS, source_rate_current,
            )
            
            # Executa simulação de calibração
            flux_measured_per_particle, flux_unc, vol_region = self._run_calibration_iteration(
                source_rate_current, iteration
            )
            
            if flux_measured_per_particle <= 0.0 or np.isnan(flux_measured_per_particle):
                logger.error("Fluxo medido zero ou NaN — abortando calibração")
                return CalibrationResult(
                    success=False,
                    flux_target=self.flux_target,
                    flux_achieved=0.0,
                    source_rate_calibrated=source_rate_current,
                    source_rate_initial=self.source_rate_initial,
                    n_iterations=iteration + 1,
                    converged=False,
                    error_relative_final=1.0,
                    tally_used=self.config.CALIBRATION_TALLY_NAME,
                    calibration_region_volume_cm3=vol_region,
                    iterations_history=list(self._history),
                    error_message=f"Fluxo medido zero/NaN na iteração {iteration + 1}",
                )
            
            # FIX V239: Conversão correta de tally por partícula-fonte para fluxo físico
            # O tally de fluxo em OpenMC fixed source retorna [n·cm/src] (por partícula-fonte)
            # Para obter fluxo físico [n/cm²/s], precisamos:
            #   ϕ_físico = (tally_flux × source_rate) / volume_regiao
            # Justificativa:
            #   - tally_flux tem unidades de track-length estimate: [n·cm/src]
            #   - Multiplicando por source_rate [n/s]: [n·cm/s]
            #   - Dividindo por volume [cm³]: [n/cm²/s] ✓
            flux_physical = (flux_measured_per_particle * source_rate_current) / vol_region
            
            # Calcula erro relativo contra fluxo-alvo experimental
            error_rel = abs(self.flux_target - flux_physical) / self.flux_target
            
            # Registra histórico
            iter_record = {
                "iteration": iteration + 1,
                "source_rate_n_s": source_rate_current,
                "source_rate_previous_n_s": source_rate_previous,
                "flux_tally_per_particle": flux_measured_per_particle,
                "flux_physical_n_cm2_s": flux_physical,
                "flux_uncertainty": flux_unc,
                "flux_target_n_cm2_s": self.flux_target,
                "calibration_volume_cm3": vol_region,
                "error_relative": error_rel,
                "converged": error_rel <= self.config.FLUX_TOLERANCE_REL,
            }
            self._history.append(iter_record)
            
            logger.info(
                "  Tally flux: %.4e ± %.4e n·cm/src  |  Volume: %.4f cm³",
                flux_measured_per_particle, flux_unc, vol_region,
            )
            logger.info(
                "  Fluxo físico: %.4e n/cm²/s (alvo: %.4e), erro: %.4f%%",
                flux_physical, self.flux_target, error_rel * 100,
            )
            
            # Verifica convergência
            if error_rel <= self.config.FLUX_TOLERANCE_REL:
                logger.info(
                    "Calibração convergiu em %d iterações: erro=%.4f%% <= %.4f%%",
                    iteration + 1, error_rel * 100, self.config.FLUX_TOLERANCE_REL * 100,
                )
                return CalibrationResult(
                    success=True,
                    flux_target=self.flux_target,
                    flux_achieved=flux_physical,
                    source_rate_calibrated=source_rate_current,
                    source_rate_initial=self.source_rate_initial,
                    n_iterations=iteration + 1,
                    converged=True,
                    error_relative_final=error_rel,
                    tally_used=self.config.CALIBRATION_TALLY_NAME,
                    calibration_region_volume_cm3=vol_region,
                    iterations_history=list(self._history),
                )
            
            # Atualiza source_rate com under-relaxation
            source_rate_previous = source_rate_current
            sr_new = source_rate_current * (self.flux_target / flux_physical)
            source_rate_current = (
                self.config.UNDER_RELAXATION * sr_new +
                (1.0 - self.config.UNDER_RELAXATION) * source_rate_current
            )
            
            # FIX V239: Log corrigido mostrando valor anterior REAL
            logger.info(
                "  source_rate atualizado: %.4e → %.4e (under-relax=%.2f, fator=%.4f)",
                source_rate_previous, source_rate_current, 
                self.config.UNDER_RELAXATION, sr_new / source_rate_previous if source_rate_previous > 0 else 0,
            )
        
        # Não convergiu dentro do máximo de iterações
        logger.warning(
            "Calibração não convergiu em %d iterações — usando último valor",
            self.config.MAX_ITERATIONS,
        )
        return CalibrationResult(
            success=False,
            flux_target=self.flux_target,
            flux_achieved=flux_physical,
            source_rate_calibrated=source_rate_current,
            source_rate_initial=self.source_rate_initial,
            n_iterations=self.config.MAX_ITERATIONS,
            converged=False,
            error_relative_final=error_rel,
            tally_used=self.config.CALIBRATION_TALLY_NAME,
            calibration_region_volume_cm3=vol_region,
            iterations_history=list(self._history),
            error_message=f"Não convergiu em {self.config.MAX_ITERATIONS} iterações",
        )
    
    def _run_calibration_iteration(
        self,
        source_rate: float,
        iteration: int,
    ) -> Tuple[float, float, float]:
        """
        Executa uma iteração de calibração e retorna (fluxo_por_particula, incerteza, volume_regiao).
        
        FIX V239:
          - Exporta TODOS os XMLs (geometry, materials, settings, tallies)
          - Usa openmc.Model completo para consistência
          - Retorna volume da região de calibração para conversão física
        """
        # Obtém geometria e materiais reais do resultado
        openmc_geometry = self.geometry_result.get("openmc_geometry")
        openmc_materials = self.geometry_result.get("openmc_materials")
        
        if openmc_geometry is None or openmc_materials is None:
            logger.error("geometry_result não contém openmc_geometry ou openmc_materials")
            return 0.0, 0.0, 0.0
        
        # Reseta IDs automáticos do OpenMC para evitar IDWarning
        # "Another Material instance already exists with id=N" que ocorre quando
        # a calibração é executada após o depletor no mesmo processo Python.
        try:
            openmc.reset_auto_ids()
        except AttributeError:
            pass  # OpenMC < 0.13 não tem reset_auto_ids

        # Configura modelo OpenMC mínimo para calibração
        settings = self._build_calibration_settings(source_rate, iteration)
        tallies = self._build_calibration_tallies()
        
        # Cria modelo completo
        model = openmc.Model(
            geometry=openmc_geometry,
            materials=openmc_materials,
            settings=settings,
            tallies=tallies,
        )
        
        # FIX BUG RAIZ (statepoint errado):
        # Path("temp_calibration") é relativo ao cwd do processo, que pode ser o
        # mesmo diretório de saída do depletor. O depletor grava "openmc_simulation_nN.h5"
        # no cwd; o glob pegava esses statepoints (mais recentes por mtime) em vez do
        # da calibração. Statepoints de depleção têm tallies de reaction rates com
        # df["mean"] como arrays por nuclídeo → "setting an array element with a sequence".
        # Correção: tempfile.mkdtemp() cria diretório absoluto e exclusivo.
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="calib_")).resolve()

        model.export_to_xml(temp_dir)
        logger.debug("XMLs de calibração exportados para: %s", temp_dir)

        # FIX BUG 1: openmc.run() não aceita 'particles' nem 'batches' como kwargs.
        try:
            openmc.run(cwd=str(temp_dir))
        except Exception as exc:
            logger.error("OpenMC falhou na calibração: %s", exc)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0.0, 0.0, 0.0

        # Lê StatePoint — apenas dentro do temp_dir isolado
        sp_files = list(temp_dir.glob("statepoint.*.h5"))
        if not sp_files:
            logger.error("Nenhum statepoint encontrado em %s após calibração", temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0.0, 0.0, 0.0

        sp_path = max(sp_files, key=lambda p: p.stat().st_mtime)

        try:
            with openmc.StatePoint(str(sp_path)) as sp:
                tally = sp.get_tally(name=self.config.CALIBRATION_TALLY_NAME)

                # Validar que é o tally correto (score 'flux', não reaction rates)
                if "flux" not in tally.scores:
                    logger.error(
                        "Tally '%s' tem scores=%s — esperado ['flux']. "
                        "Statepoint incorreto ou tally sobreescrito pelo depletor.",
                        self.config.CALIBRATION_TALLY_NAME, tally.scores,
                    )
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return 0.0, 0.0, 0.0

                # FIX BUG RAIZ V244: NUNCA usar get_pandas_dataframe() para tallies de fluxo
                # O erro "setting an array element with a sequence" ocorre DENTRO do pandas
                # ao tentar converter Series com arrays numpy aninhados (dtype=object).
                # SOLUÇÃO DEFINITIVA: Usar APENAS tally.get_values() que retorna array limpo.
                
                logger.debug("Extraindo fluxo via tally.get_values() (sem pandas)...")
                logger.debug("Tally info: name=%s, scores=%s, filters=%s", 
                    tally.name, tally.scores, 
                    [type(f).__name__ for f in tally.filters] if hasattr(tally, 'filters') else 'N/A'
                )
                
                mean_flux_per_particle = 0.0
                std_flux = 0.0
                
                try:
                    # Extrai valor médio direto como array numpy
                    flux_values = tally.get_values(value='mean')
                    logger.debug(
                        "RAW flux_values: type=%s, value=%s",
                        type(flux_values), repr(flux_values)
                    )
                    if flux_values is not None:
                        # FIX V245: Converter para numpy array explicitamente e achatar
                        flux_array = np.asarray(flux_values, dtype=float).ravel()
                        logger.debug(
                            "flux_array: shape=%s, dtype=%s, sum=%.4e",
                            flux_array.shape, flux_array.dtype, flux_array.sum()
                        )
                        mean_flux_per_particle = float(flux_array.sum())
                        logger.info(
                            "get_values(mean) sucesso: flux=%.4e (achatar: shape=%s dtype=%s)",
                            mean_flux_per_particle, flux_array.shape, flux_array.dtype
                        )
                    else:
                        logger.error("get_values(mean) retornou None")
                        mean_flux_per_particle = 0.0
                except Exception as exc_mean:
                    logger.error("get_values(mean) falhou com exceção: %s", exc_mean, exc_info=True)
                    mean_flux_per_particle = 0.0
                
                # Extrai desvio padrão
                try:
                    std_values = tally.get_values(value='std_dev')
                    if std_values is not None and std_values.size > 0:
                        std_flux = float(np.sum(std_values))
                        logger.debug("get_values(std_dev): %.4e", std_flux)
                    else:
                        std_flux = 0.0
                except Exception as exc_std:
                    logger.warning("get_values(std_dev) falhou: %s", exc_std)
                    std_flux = 0.0
                
                # Validação explícita: fluxo deve ser escalar positivo
                if mean_flux_per_particle <= 0.0 or np.isnan(mean_flux_per_particle):
                    logger.error(
                        "Fluxo extraído inválido: %.4e (tally=%s, scores=%s)",
                        mean_flux_per_particle, tally.name, tally.scores,
                    )
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return 0.0, 0.0, 0.0

                vol_region = self._get_calibration_volume(tally, sp)
                
                if vol_region <= 0.0:
                    logger.warning("Volume da região de calibração não determinado — usando estimativa")
                    # Estimativa baseada na primeira camada
                    first_layer_thick = self._estimate_first_layer_thickness()
                    vol_region = self.target_face_area_cm2 * first_layer_thick
                
                logger.debug(
                    "Calibração: tally_flux=%.4e ± %.4e n·cm/src, volume=%.4f cm³",
                    mean_flux_per_particle, std_flux, vol_region
                )
                
                # FIX V239: Retorna fluxo POR PARTÍCULA-FONTE e volume separadamente
                # A conversão para fluxo físico será feita no loop principal
                shutil.rmtree(temp_dir, ignore_errors=True)
                return mean_flux_per_particle, std_flux, vol_region

        except Exception as exc:
            logger.error("Erro ao ler statepoint da calibração: %s", exc)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0.0, 0.0, 0.0
        finally:
            # Limpa arquivos temporários
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_calibration_volume(self, tally: openmc.Tally, sp: openmc.StatePoint) -> float:
        """
        Obtém volume da região de calibração a partir do tally ou geometria.
        
        FIX V246: Usa filt.bins em vez de filt.cells para compatibilidade com OpenMC >= 0.13
        """
        # Tenta obter volume dos filtros do tally
        for filt in tally.filters:
            if isinstance(filt, openmc.CellFilter):
                # FIX V246: Em OpenMC >= 0.13, o atributo é 'bins', não 'cells'
                cell_ids = getattr(filt, 'bins', getattr(filt, 'cells', []))
                
                if len(cell_ids) > 0:
                    # Soma volumes das células
                    total_vol = 0.0
                    for cell_id in cell_ids:
                        # Tenta obter volume do resumo da geometria
                        try:
                            summary = sp.summary
                            if summary and hasattr(summary, 'geometry'):
                                cell = summary.geometry.get_cell_by_id(cell_id)
                                if cell and hasattr(cell, 'volume') and cell.volume is not None:
                                    total_vol += float(cell.volume)
                        except Exception as e:
                            logger.debug("Erro ao obter volume da célula %d: %s", cell_id, e)
                            pass
                    
                    if total_vol > 0.0:
                        logger.debug("Volume da região de calibração: %.4f cm³ (células=%s)", total_vol, cell_ids)
                        return total_vol
        
        return 0.0  # Volume não determinado
    
    def _estimate_first_layer_thickness(self) -> float:
        """Estima espessura da primeira camada do alvo."""
        layers = self.geometry_result.get("layers", [])
        if layers and len(layers) > 0:
            # Pega espessura da primeira camada
            first_layer = layers[0]
            thick_cm = first_layer.get("thickness_cm")
            if thick_cm is None:
                thick_mm = first_layer.get("thickness_mm", 0.0)
                thick_cm = thick_mm / 10.0 if thick_mm > 0 else 0.1  # fallback 1mm
            return float(thick_cm) if thick_cm > 0 else 0.1
        return 0.1  # fallback 1mm
    
    def _build_calibration_settings(
        self,
        source_rate: float,
        iteration: int,
    ) -> openmc.Settings:
        """Constrói settings OpenMC para iteração de calibração."""
        settings = openmc.Settings()
        settings.run_mode = "fixed source"
        settings.particles = self.config.PARTICLES_PER_ITERATION
        settings.batches = self.config.BATCHES_PER_ITERATION
        settings.inactive = 0
        
        # Semente aleatória para reprodutibilidade
        if self.config.RANDOM_SEED is not None:
            settings.seed = self.config.RANDOM_SEED + iteration
        
        # FIX V239: Posiciona fonte baseado nos metadados da geometria real
        # Usa WATER_AXIAL_CM da geometria construída, não valor hardcoded
        metadata = self.geometry_result.get("metadata", {})
        water_geom = metadata.get("water_geometry", {})
        water_axial = water_geom.get("axial_cm", self.water_axial_cm)
        
        # Fonte deve estar na face de entrada da água frontal
        # z = -water_axial (face externa) + pequena folga
        z_source = -water_axial + 1e-6
        
        # Espessura infinitesimal da fonte plana
        dz_source = GeometryContract.SOURCE_PLANE_THICKNESS_CM
        
        # Dimensões da fonte derivadas da geometria real
        # Deve cobrir: alvo + água lateral
        source_box = openmc.stats.Box(
            [-self.source_x_cm / 2, -self.source_y_cm / 2, z_source],
            [ self.source_x_cm / 2,  self.source_y_cm / 2, z_source + dz_source],
        )
        
        settings.source = [openmc.IndependentSource(
            space=source_box,
            energy=self._build_energy_distribution(),
            angle=openmc.stats.Monodirectional(GeometryContract.SOURCE_DIRECTION),
        )]
        
        settings.output = {"summary": True}
        
        logger.debug(
            "Fonte configurada: z=%.4e cm, dimensões=%.3f×%.3f cm², source_rate=%.4e n/s",
            z_source, self.source_x_cm, self.source_y_cm, source_rate,
        )
        
        return settings
    
    def _build_energy_distribution(self) -> "openmc.stats.Univariate":
        """Constrói distribuição energética da fonte a partir do espectro.
        
        FIX V243 — BUG FÍSICO CRÍTICO:
          O parser retorna espectro TABULAR com estrutura:
            {"type": "tabular", "data": {"energies_ev": [...], "probabilities": [...]}}
          Este método agora acessa corretamente os dados aninhados em 'data'.
        """
        # CASO 1: Parser retornou espectro TABULAR direto (ORIGEN252, etc.)
        # Estrutura: {"type": "tabular", "data": {"energies_ev": [...], "probabilities": [...]}}
        if self.energy_source.get("type") == "tabular":
            data = self.energy_source.get("data", {})
            energies = data.get("energies_ev") or data.get("parsed_data", {}).get("energies_ev")
            probs = data.get("probabilities") or data.get("parsed_data", {}).get("probabilities")
            if energies and probs and len(energies) == len(probs):
                logger.info(
                    "Usando espectro tabular ORIGEN252: %d pontos, source=%s, weights=%s",
                    len(energies),
                    data.get("source", self.energy_source.get("source", "desconhecida")),
                    data.get("weights", "N/A")
                )
                return openmc.stats.Tabular(
                    [float(e) for e in energies],
                    [float(p) for p in probs],
                    interpolation="histogram"
                )
        
        # CASO 2: Estrutura antiga com 'energies_ev' direto no root (compatibilidade)
        if "energies_ev" in self.energy_source and "probabilities" in self.energy_source:
            energies = self.energy_source["energies_ev"]
            probs = self.energy_source["probabilities"]
            if energies and probs and len(energies) == len(probs):
                logger.info(
                    "Usando espectro tabular direto (legacy): %d pontos, origem=%s",
                    len(energies),
                    self.energy_source.get("source", "desconhecida"),
                )
                return openmc.stats.Tabular(
                    [float(e) for e in energies],
                    [float(p) for p in probs],
                    interpolation="histogram"
                )
        
        # CASO 3: Estrutura aninhada (tipo antigo com 'type' e 'data')
        stype = self.energy_source.get("type", "maxwell")
        data = self.energy_source.get("data", {})
        kb_eV = PhysicsConstants.KB_EV
        
        if stype == "single":
            ev = data.get("energy_ev")
            if ev is not None:
                return openmc.stats.Discrete([float(ev)], [1.0])
        
        if stype == "discrete":
            energies = data.get("energies_ev", [])
            probs = data.get("probabilities", [])
            if energies and probs:
                return openmc.stats.Discrete(
                    [float(e) for e in energies],
                    [float(p) for p in probs],
                )
        
        if stype == "maxwell":
            params = data.get("parameters", {})
            theta = float(params.get("theta", 300.0 * kb_eV))
            return openmc.stats.Maxwell(theta)
        
        if stype == "watt":
            params = data.get("parameters", {})
            return openmc.stats.Watt(
                float(params.get("a", 0.988e6)),
                float(params.get("b", 2.249e-6)),
            )
        
        if stype == "tabular":
            energies = data.get("energies_ev", [])
            probs = data.get("probabilities", [])
            if energies and probs and len(energies) == len(probs):
                return openmc.stats.Tabular(energies, probs)
        
        # Fallback: Maxwell com temperatura padrão
        logger.warning("Espectro não reconhecido, usando Maxwell fallback")
        return openmc.stats.Maxwell(300.0 * kb_eV)
    
    def _build_calibration_tallies(self) -> openmc.Tallies:
        """
        Constrói tallies para calibração.
        
        FIX V239:
          - Usa chave CORRETA 'cellsdict' do contrato geometry.py
          - Remove filtro desnecessário de nomes de água
          - cellsdict já contém apenas células do wafer
        """
        # FIX V240: Chave correta é 'cells_dict' conforme contrato geometry.py (linha 280)
        # geometry.py retorna: "cells_dict": wafer_cells (apenas células do wafer)
        cells_dict = self.geometry_result.get("cells_dict", {})
        
        if not cells_dict:
            raise ValueError(
                "cells_dict ausente na geometria. "
                "Verifique se geometry.py foi executado corretamente e retornou 'cells_dict'. "
                f"Chaves disponíveis em geometry_result: {list(self.geometry_result.keys())}"
            )
        
        # FIX V239: cellsdict já contém apenas células do wafer (sem água)
        # Portanto, pegamos simplesmente a primeira célula do dicionário
        # A ordem das camadas é preservada na construção da geometria
        first_layer_name = None
        first_layer_cell = None
        
        # Itera nas células na ordem em que foram inseridas (preserva ordem das camadas)
        for name, cell in cells_dict.items():
            first_layer_name = name
            first_layer_cell = cell
            break  # Pega a primeira
        
        if first_layer_cell is None:
            raise ValueError("Nenhuma camada do alvo encontrada em cellsdict")
        
        # FIX V245: Extrair ID da célula explicitamente
        # OpenMC 0.15.3 CellFilter pode exigir IDs (int) em vez de objetos cell
        cell_id = first_layer_cell.id if hasattr(first_layer_cell, 'id') else int(first_layer_cell)
        
        logger.info(
            "Região de calibração: '%s' (id=%d, type=%s)",
            first_layer_name,
            cell_id,
            type(first_layer_cell).__name__
        )
        
        # Tally de fluxo na primeira camada
        # FIX V245: Passar ID como inteiro, não objeto cell
        cell_filter = openmc.CellFilter([cell_id])
        
        tally = openmc.Tally(name=self.config.CALIBRATION_TALLY_NAME)
        tally.filters = [cell_filter]
        tally.scores = ["flux"]
        
        # Estima volume da região para conversão posterior
        self._calibration_volume_cm3 = self._estimate_first_layer_thickness() * self.target_face_area_cm2
        
        tallies = openmc.Tallies([tally])
        
        logger.info(
            "Tally de calibração criado: '%s' na célula '%s', volume estimado=%.4f cm³",
            self.config.CALIBRATION_TALLY_NAME, first_layer_name, self._calibration_volume_cm3,
        )
        
        return tallies


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_source(
    flux_target: float,
    geometry_result: Dict,
    energy_source: Dict,
    wafer_x_cm: float,
    wafer_y_cm: float,
    water_lateral_cm: float,
    config: Optional[SourceCalibrationConfig] = None,
    debug: bool = False,
) -> CalibrationResult:
    """
    Função de alto nível para calibrar a fonte.
    
    Args:
        flux_target: Fluxo-alvo experimental [n/cm²/s]
        geometry_result: Resultado de geometry.build()
        energy_source: Espectro de energia (parser_data['energy_source'])
        wafer_x_cm: Dimensão x do alvo [cm]
        wafer_y_cm: Dimensão y do alvo [cm]
        water_lateral_cm: Espessura de água lateral [cm]
        config: Configuração opcional de calibração
        debug: Modo debug
    
    Returns:
        CalibrationResult com source_rate calibrado
    """
    calibrator = SourceCalibrator(
        flux_target=flux_target,
        geometry_result=geometry_result,
        energy_source=energy_source,
        wafer_x_cm=wafer_x_cm,
        wafer_y_cm=wafer_y_cm,
        water_lateral_cm=water_lateral_cm,
        config=config,
        debug=debug,
    )
    return calibrator.run()


__all__ = [
    "SourceCalibrator",
    "CalibrationResult",
    "calibrate_source",
]
