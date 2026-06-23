#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py V2 — Fonte única de todas as constantes e configurações do pipeline.

Importado por: geometry, settings, simulation, thermal, pyne_bridge, tallies, maestro.

CHANGELOG V2:
  - SimulationDefaults: DT_H separado em DT_H_OUTPUT e DT_H_DEPLETION;
    adicionado DEPLETION_SUBSTEPS e DEPLETION_INTEGRATOR.
  - TNLoopConfig: PICARD_MAX_INTERNAL ↑ 5→10; CONVERGENCE_EPSILON_TEMP ↓ 1→0.5K.
  - CoolingConfig.n_snapshots: evita divisão por zero quando COOLING_TIME_H=0.
  - NuclearDataPaths: candidatos de chain agora incluem chain_endfb80_act.xml
    (yields cumulativos, melhor para ativação de alvos).

CHANGELOG V238 (Refatoração de Calibração de Fonte):
  - SourceCalibrationConfig: nova classe com parâmetros de calibração da fonte.
  - GeometryContract: regras geométricas explícitas (distância fonte-face = 1 cm).
  - Defaults centralizados para calibração: tolerância, iterações máximas, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "ValidationLimits",
    "PhysicsConstants",
    "TNLoopConfig",
    "CoolingConfig",
    "NuclearDataPaths",
    "GeometryLimits",
    "GeometryContract",
    "SourceCalibrationConfig",
    "ThermalSolverConfig",
    "ChainDataProxy",
    "SimulationDefaults",
    "SimulationModes",
    "DepletionAutoTuner",
    "DepletionParams",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES FÍSICAS
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsConstants:
    """Constantes físicas CODATA 2018 — imutáveis."""
    EV_TO_J: float = 1.602_176_634e-19   # J/eV  (exato)
    N_A:     float = 6.022_140_76e23     # mol⁻¹ (exato)
    KB_EV:   float = 8.617_333_262e-5    # eV/K  (kB em eV)
    LN2:     float = 0.693_147_180_559_945_3  # ln(2)
    SECONDS_PER_HOUR: float = 3600.0     # s/h — evita magic number nas conversões


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIMITES DE VALIDAÇÃO DE TEMPERATURA E DENSIDADE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationLimits:
    """Limites de segurança física. Todos os valores em Kelvin."""
    TEMP_MIN_K:             float = 250.0
    TEMP_MAX_K:             float = 3500.0
    TEMP_COMBUSTIVEL_MAX_K: float = 3000.0
    TEMP_AGUA_MAX_K:        float = 623.0
    TEMP_ZIRCALOY_MAX_K:    float = 2263.0
    TEMP_ALUMINIO_MAX_K:    float = 855.0
    TEMP_ACA_MAX_K:         float = 1673.0
    RHO_MAX_GCM3:           float = 25.0    # acima de Os = 22.6 g/cm³
    SOURCE_RATE_MIN:        float = 1e3     # n/s mínimo aceitável


# ══════════════════════════════════════════════════════════════════════════════
# 2.5. CONTRATO GEOMÉTRICO DA FONTE INCIDENTE
# ══════════════════════════════════════════════════════════════════════════════

class GeometryContract:
    """
    Regras geométricas obrigatórias para o contrato físico da fonte incidente.
    
    Este contrato define a geometria padrão da simulação:
    - Alvo: paralelepípedo retangular com seção x×y e profundidade = soma das camadas
    - Fonte: plana, unidirecional, incidente normalmente na face frontal (+z)
    - Posição da fonte: em água frontal, a DISTANCE_SOURCE_TO_FACE_CM da face do alvo
    - Dimensões da fonte: coincidem com a célula frontal (alvo + água lateral)
    
    Estas regras NÃO são números mágicos — são derivadas do contrato físico e
    devem ser respeitadas por geometry.py e simulation.py.
    """
    # Distância fixa da fonte à face frontal do alvo [cm]
    # Regra explícita do contrato: fonte está a 1 cm da face de entrada
    DISTANCE_SOURCE_TO_FACE_CM: float = 1.0
    
    # Espessura infinitesimal da fonte plana [cm]
    SOURCE_PLANE_THICKNESS_CM: float = 1e-6
    
    # A fonte deve cobrir automaticamente: dimensão do alvo + água lateral
    # Isso é implementado em geometry.py ao construir a região de água frontal
    SOURCE_COVERS_FRONT_FACE: bool = True
    
    # Orientação: monodirecional em +Z (incidente normal à face x-y)
    SOURCE_DIRECTION: Tuple[float, float, float] = (0.0, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.6. CONFIGURAÇÃO DE CALIBRAÇÃO DA FONTE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceCalibrationConfig:
    """
    Parâmetros de calibração da intensidade da fonte.
    
    Quando FLUXO + espectro são fornecidos, o simulador deve executar uma
    etapa de calibração para encontrar source_rate que reproduza o fluxo-alvo
    experimental na face do alvo.
    
    A calibração usa um modelo OpenMC curto com tally de fluxo na região
    receptora imediatamente após a face de entrada do alvo.
    
    Algoritmo: sr_novo = sr_velho × (fluxo_alvo / fluxomedido)
    """
    # Habilitar calibração automática da fonte
    ENABLE_CALIBRATION: bool = True
    
    # Tolerância relativa para convergência do fluxo calibrado
    # Ex: 0.02 = 2% de tolerância
    FLUX_TOLERANCE_REL: float = 0.02
    
    # Número máximo de iterações de calibração
    MAX_ITERATIONS: int = 10
    
    # Contagem de partículas por simulação de calibração
    # Valor menor que produção para velocidade, mas suficiente para estatística
    PARTICLES_PER_ITERATION: int = 50_000
    
    # Batches por iteração de calibração
    BATCHES_PER_ITERATION: int = 5
    
    # Fator de under-relaxation para estabilidade numérica
    # Evita oscilações quando ruído estatístico é alto
    UNDER_RELAXATION: float = 0.8
    
    # Fluxo mínimo aceitável para evitar divisão por zero
    MIN_FLUX_MEASURED: float = 1e-6  # n/cm²/s
    
    # Se True, usa tally de corrente superficial na face; se False, usa
    # tally de fluxo volumétrico na primeira camada como proxy
    USE_SURFACE_CURRENT: bool = False
    
    # Nome do tally de calibração (para identificação no StatePoint)
    CALIBRATION_TALLY_NAME: str = "flux_calibration"
    
    # Semente aleatória para reprodutibilidade da calibração
    RANDOM_SEED: Optional[int] = None
    
    def __post_init__(self) -> None:
        if not 0 < self.FLUX_TOLERANCE_REL < 1:
            raise ValueError(f"FLUX_TOLERANCE_REL deve estar em (0, 1), obteve {self.FLUX_TOLERANCE_REL}")
        if self.MAX_ITERATIONS < 1:
            raise ValueError(f"MAX_ITERATIONS deve ser >= 1, obteve {self.MAX_ITERATIONS}")
        if not 0 < self.UNDER_RELAXATION <= 1:
            raise ValueError(f"UNDER_RELAXATION deve estar em (0, 1], obteve {self.UNDER_RELAXATION}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIMITES GEOMÉTRICOS
# ══════════════════════════════════════════════════════════════════════════════

class GeometryLimits:
    """Limites para construção de geometria OpenMC."""
    NANOSCALE_MIN_CM: float = 1.0e-6   # espessura mínima de camada [cm]
    GAP_TOLERANCE_CM: float = 1.0e-8   # tolerância de gap entre camadas [cm]


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIGURAÇÃO DO LOOP TÉRMICO-NEUTRÔNICO
# ══════════════════════════════════════════════════════════════════════════════

class TNLoopConfig:
    """Parâmetros de controle do acoplamento Térmico-Neutrônico (Phase F).

    ENABLE_TN_COUPLING é definido em runtime por TNLoopConfig.from_parser()
    a partir do campo 'thermal_coupling' do Input-simulador.txt.
    O default de classe é False para garantir que uma simulação pura OpenMC
    (sem PyNE/thermal) rode quando o arquivo de input não for lido ainda.
    """

    ENABLE_TN_COUPLING:       bool  = False   # ALTERADO: default False — lido do input
    RELAXATION_FACTOR:        float = 0.5
    MAX_TN_ITERATIONS:        int   = 20
    CONVERGENCE_EPSILON_TEMP: float = 0.5
    CONVERGENCE_EPSILON_POWER: float = 0.01
    CONVERGENCE_EPSILON_RHO:  float = 0.001
    MAX_TEMPERATURE_K:        float = 3500.0
    MIN_TEMPERATURE_K:        float = 273.0
    DYNAMIC_RELAXATION:       bool  = False
    PICARD_MAX_INTERNAL:      int   = 10

    @classmethod
    def from_parser(cls, parser_data: dict) -> None:
        """Lê THERMAL_COUPLING do input e configura ENABLE_TN_COUPLING.

        Chamado por maestro.run_pipeline() logo após parse, antes de Phase D.
        Quando THERMAL_COUPLING = false no input:
          - Phase F (T-N loop) é completamente ignorada
          - PyNE não é chamado em nenhum ponto do ciclo de irradiação
          - _run_cooling_pyne() ainda roda se COOLING_TIME_H > 0 (PyNE para decay)
          - Permite diagnóstico de simulação OpenMC pura sem interferência do acoplamento
        """
        sim_params = parser_data.get("simulation_parameters", {})
        raw = sim_params.get("thermal_coupling", sim_params.get("THERMAL_COUPLING", None))
        if raw is None:
            # não encontrado no input → manter default False (simulação pura)
            cls.ENABLE_TN_COUPLING = False
            return
        if isinstance(raw, bool):
            cls.ENABLE_TN_COUPLING = raw
        else:
            cls.ENABLE_TN_COUPLING = str(raw).strip().lower() in ("true", "1", "yes", "sim")

    @classmethod
    def validate(cls) -> None:
        assert 0 < cls.RELAXATION_FACTOR < 1
        assert 0 < cls.CONVERGENCE_EPSILON_POWER < 1
        assert cls.MAX_TN_ITERATIONS >= 1
        assert cls.MIN_TEMPERATURE_K < cls.MAX_TEMPERATURE_K


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONFIGURAÇÃO DO SOLVER TÉRMICO FDM
# ══════════════════════════════════════════════════════════════════════════════

class ThermalSolverConfig:
    """Parâmetros do solver FDM 1D (thermal.py)."""
    H_CONV:     float = 5000.0  # coef. convecção superficial [W/m²K]
    PICARD_TOL: float = 0.1     # tolerância Picard interna [K]


# ══════════════════════════════════════════════════════════════════════════════
# 6. CONFIGURAÇÃO DE RESFRIAMENTO PÓS-IRRADIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoolingConfig:
    """Parâmetros para o cálculo de decaimento pós-shutdown (Phases G/H/I)."""
    COOLING_TIME_H:     float = 0.0
    COOLING_INTERVAL_H: float = 1.0
    ACTIVATE_STRUCTURAL: bool = False
    COMPUTE_DOSE:        bool = False

    def __post_init__(self) -> None:
        if self.COOLING_TIME_H < 0:
            raise ValueError(f"COOLING_TIME_H deve ser >= 0, obteve {self.COOLING_TIME_H}")
        if self.COOLING_INTERVAL_H <= 0:
            raise ValueError(f"COOLING_INTERVAL_H deve ser > 0, obteve {self.COOLING_INTERVAL_H}")
        if self.COOLING_TIME_H > 0 and self.COOLING_INTERVAL_H > self.COOLING_TIME_H:
            raise ValueError(
                f"COOLING_INTERVAL_H ({self.COOLING_INTERVAL_H}h) > "
                f"COOLING_TIME_H ({self.COOLING_TIME_H}h)"
            )

    @property
    def n_snapshots(self) -> int:
        # FIX: proteção contra divisão por zero quando COOLING_TIME_H == 0
        if self.COOLING_TIME_H <= 0:
            return 0
        n = round(self.COOLING_TIME_H / self.COOLING_INTERVAL_H)
        return max(1, n) + 1

    @property
    def phase_g_active(self) -> bool:
        return self.COOLING_TIME_H > 0

    @property
    def phase_h_active(self) -> bool:
        return self.ACTIVATE_STRUCTURAL

    @property
    def phase_i_active(self) -> bool:
        return self.COMPUTE_DOSE

    @classmethod
    def from_simulation_params(cls, sim_params: dict) -> "CoolingConfig":
        return cls(
            COOLING_TIME_H=float(sim_params.get("cooling_time_h", 0.0)),
            COOLING_INTERVAL_H=float(sim_params.get("cooling_interval_h", 1.0)),
            ACTIVATE_STRUCTURAL=bool(sim_params.get("activate_structural", False)),
            COMPUTE_DOSE=bool(sim_params.get("compute_dose", False)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. CAMINHOS DE DADOS NUCLEARES
# ══════════════════════════════════════════════════════════════════════════════

class NuclearDataPaths:
    """
    Hierarquia canônica de bibliotecas de cross-sections e chain files.

    CHAIN: preferência por chain_endfb80_act.xml que usa yields CUMULATIVOS
    e inclui isômeros de curta meia-vida relevantes para produção de Mo99/Tc99m.
    Yields cumulativos > independentes para Mo99: 6.11% vs 5.66%.
    """
    _BASE = Path.home() / "nuclear_data"

    XS_CANDIDATES: List[Tuple[str, Path]] = [
        ("ENDF-B-VIII.0", _BASE / "endf_b_viii_0_hdf5"  / "cross_sections.xml"),
        ("ENDF-B-VIII.0", _BASE / "endfb_viii_0_hdf5"   / "cross_sections.xml"),
        ("TENDL-2021",    _BASE / "hdf5_lib_tendl2021"  / "cross_sections.xml"),
        ("TENDL-2015",    _BASE / "hdf5_lib_tendl2015"  / "cross_sections.xml"),
        ("ENDF-B-VII.1",  _BASE / "endfb71_hdf5"        / "cross_sections.xml"),
        ("JEFF-3.3",      _BASE / "jeff33_hdf5"         / "cross_sections.xml"),
    ]

    # FIX: chain_endfb80_act.xml tem yields cumulativos → Mo99 mais preciso.
    # Adicionado antes dos candidatos _pwr para priorizar em simulações de ativação.
    CHAIN_CANDIDATES: List[Path] = [
        _BASE / "chain_endfb80_act.xml",          # yields cumulativos — preferido
        _BASE / "chain_endfb80_pwr.xml",
        _BASE / "chain_endfb80.xml",
        _BASE / "chain_endfb71_act.xml",
        _BASE / "chain_endfb71.xml",
        Path("chain_endfb80_act.xml"),
        Path("chain_endfb80_pwr.xml"),
        Path("chain_endfb80.xml"),
        Path("chain_endfb71.xml"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 8. CHAIN DATA PROXY
# ══════════════════════════════════════════════════════════════════════════════

class ChainDataProxy:
    """
    Proxy mínimo para data_manager exigido por simulation.py.
    Expõe get_chain_file() e get_xs_path() sem depender do data_manager legado.
    """

    def __init__(self, chain_path: Path, xs_path: Path) -> None:
        self._chain = chain_path
        self._xs    = xs_path

    def get_chain_file(self) -> Optional[Path]:
        return self._chain if self._chain and self._chain.exists() else None

    def get_xs_path(self) -> Optional[Path]:
        return self._xs if self._xs and self._xs.exists() else None

    def __repr__(self) -> str:
        return (f"ChainDataProxy(chain={self._chain.name if self._chain else 'None'}, "
                f"xs={self._xs.name if self._xs else 'None'})")


# ══════════════════════════════════════════════════════════════════════════════
# 8.5. MODOS DE SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class SimulationModes:
    """Constantes para SIMULATION_MODE no Input-simulador.txt."""
    ACTIVATION   = "ACTIVATION"   # Fixed-source depletion, source-rate norm
    BURNUP       = "BURNUP"       # Fixed-source depletion, fission-q norm, potência W
    CRITICALITY  = "CRITICALITY"  # Eigenvalue + depletion, fission-q norm
    AUTO         = "AUTO"         # Detecta automaticamente por conteúdo físsil

    ALL = {ACTIVATION, BURNUP, CRITICALITY, AUTO}

    @classmethod
    def is_valid(cls, mode: str) -> bool:
        return mode.upper() in cls.ALL

    @classmethod
    def needs_power(cls, mode: str) -> bool:
        """True se o modo requer POTENCIA_W em vez de FLUXO."""
        return mode.upper() in {cls.BURNUP, cls.CRITICALITY}

    @classmethod
    def is_eigenvalue(cls, mode: str) -> bool:
        """True se o modo requer run_mode='eigenvalue'."""
        return mode.upper() == cls.CRITICALITY


# ══════════════════════════════════════════════════════════════════════════════
# 9. DEFAULTS DE SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class SimulationDefaults:
    """
    Valores padrão usados quando parâmetros estão ausentes no input.

    DT_H_DEPLETION e DEPLETION_INTEGRATOR NÃO precisam ser definidos pelo
    usuário — são escolhidos automaticamente por DepletionAutoTuner com base
    em DT_H. Ver DepletionAutoTuner para a tabela de decisão completa.
    """
    # ── Temporal ────────────────────────────────────────────────────────────
    DT_H:              float = 12.0   # LEGADO — mantido para backward compat
    DT_H_OUTPUT:       float = 12.0   # passo de saída de inventário [h]
    DT_H_DEPLETION:    float = 6.0    # passo interno padrão [h] (override automático)
    TOTAL_TIME_H:      float = 48.0   # tempo total de irradiação [h]
    COOLING_TIME_H:    float = 6.0    # tempo de resfriamento [h]

    # ── Depleção (valores fallback; DepletionAutoTuner sobrescreve) ──────────
    DEPLETION_INTEGRATOR:     str  = "celi"         # CE/LI — padrão para Mo99
    DEPLETION_NORMALIZATION:  str  = "source-rate"  # normalização por fluxo
    DEPLETION_SUBSTEPS:       int  = 2              # sub-passos por DT_H_OUTPUT

    # ── Monte Carlo ─────────────────────────────────────────────────────────
    FLUX:           float = 1e13      # fluxo de nêutrons [n/cm²/s]
    SOURCE_TEMP_K:  float = 300.0     # temperatura da fonte [K]
    NPARTICLES:     int   = 100_000   # nêutrons por passo Monte Carlo
    NBATCHES:       int   = 10        # batches ativos
    NINACTIVE:      int   = 0         # batches inativos (0 para fixed source)
    NINACTIVE_EIGENVALUE: int = 50    # batches inativos para criticidade

    # ── Geometria / Material ─────────────────────────────────────────────────
    WAFER_SIDE_CM:  float = 1.69      # dimensão padrão do wafer [cm]
    WATER_TEMP_C:   float = 25.0      # temperatura da água [°C]
    WATER_FLOW_M3S: float = 0.001     # vazão mássica [m³/s]
    POWER_W:        float = None      # potência total [W]; None = usar FLUXO


# ══════════════════════════════════════════════════════════════════════════════
# 10. AUTO-TUNER DE DEPLEÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class DepletionAutoTuner:
    """
    Escolhe automaticamente DT_H_DEPLETION e DEPLETION_INTEGRATOR com base
    no DT_H que o usuário define no Input-simulador.txt.

    O usuário nunca precisa conhecer esses parâmetros internos. A lógica
    garante precisão máxima compatível com o custo computacional do passo
    de output escolhido.

    ── Tabela de decisão ────────────────────────────────────────────────────

      DT_H_OUTPUT    DT_H_DEPLETION   Integrador   n_sub   Raciocínio
      ─────────────  ───────────────  ───────────  ──────  ────────────────────
      ≤ 6h           DT_H/2 (≥1h)    celi         2       passos finos, CELI basta
      6h < DT ≤ 12h  DT_H/2          celi         2       caso de produção padrão
                                                           DT=12h → Δt_dep=6h
      12h < DT ≤ 24h DT_H/4          celi         4       passos intermediários
      > 24h           DT_H/6          si_celi      6       passos grosseiros →
                                                           integrador implícito

    ── Regra para Mo99 ──────────────────────────────────────────────────────

      t½(Mo99) = 65.94h. DT_H_DEPLETION ≤ t½/3 ≈ 22h é o limite de estabilidade.
      Com DT=12h → DT_dep=6h: λ·Δt = 0.063 → erro de truncamento < 0.4%/passo.

    ── Override pelo usuário ─────────────────────────────────────────────────

      Se o input contiver DT_H_DEPLETION ou DEPLETION_INTEGRATOR explícitos,
      esses valores prevalecem sobre o auto-tune. O auto-tune preenche apenas
      o que estiver ausente.

    ── Limiares configuráveis ────────────────────────────────────────────────

      Altere as constantes abaixo para ajustar o comportamento sem tocar na
      lógica de decisão.
    """

    # Limiares de DT_H_OUTPUT que separam as faixas de decisão [h]
    THRESHOLD_FINE:     float = 6.0    # ≤ este valor → faixa "fina"
    THRESHOLD_MEDIUM:   float = 12.0   # ≤ este valor → faixa "média" (produção)
    THRESHOLD_COARSE:   float = 24.0   # ≤ este valor → faixa "grossa"
                                       # > THRESHOLD_COARSE → faixa "muito grossa"

    # Divisores de DT_H_OUTPUT para obter DT_H_DEPLETION por faixa
    DIVISOR_FINE:       int   = 2      # Δt_dep = DT/2  para DT ≤ 6h
    DIVISOR_MEDIUM:     int   = 2      # Δt_dep = DT/2  para 6h < DT ≤ 12h
    DIVISOR_COARSE:     int   = 4      # Δt_dep = DT/4  para 12h < DT ≤ 24h
    DIVISOR_VERYCOARSE: int   = 6      # Δt_dep = DT/6  para DT > 24h

    # Integradores por faixa
    INTEGRATOR_FINE:       str = "celi"     # 2ª ordem linear, 2 MC/passo
    INTEGRATOR_MEDIUM:     str = "celi"     # idem
    INTEGRATOR_COARSE:     str = "celi"     # idem
    INTEGRATOR_VERYCOARSE: str = "si_celi"  # implícito, mais estável

    # DT_H_DEPLETION mínimo absoluto [h] — evita sub-passos < 1h
    DT_DEPLETION_MIN_H: float = 1.0

    @classmethod
    def tune(
        cls,
        dt_output_h: float,
        user_dt_depletion_h: Optional[float] = None,
        user_integrator: Optional[str]       = None,
    ) -> "DepletionParams":
        """
        Retorna os parâmetros de depleção otimizados para dt_output_h.

        Args:
            dt_output_h:         DT_H definido pelo usuário no input [h]
            user_dt_depletion_h: DT_H_DEPLETION explícito (None = auto)
            user_integrator:     DEPLETION_INTEGRATOR explícito (None = auto)

        Returns:
            DepletionParams com dt_depletion_h, integrador, n_substeps e
            flag auto_tuned indicando se o auto-tune foi aplicado.
        """
        # ── Determinar faixa e parâmetros automáticos ─────────────────────
        if dt_output_h <= cls.THRESHOLD_FINE:
            auto_dep   = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_FINE)
            auto_integ = cls.INTEGRATOR_FINE
            band       = "fina"
        elif dt_output_h <= cls.THRESHOLD_MEDIUM:
            auto_dep   = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_MEDIUM)
            auto_integ = cls.INTEGRATOR_MEDIUM
            band       = "média (produção)"
        elif dt_output_h <= cls.THRESHOLD_COARSE:
            auto_dep   = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_COARSE)
            auto_integ = cls.INTEGRATOR_COARSE
            band       = "grossa"
        else:
            auto_dep   = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_VERYCOARSE)
            auto_integ = cls.INTEGRATOR_VERYCOARSE
            band       = "muito grossa"

        # ── Aplicar override do usuário se presente ───────────────────────
        final_dep   = user_dt_depletion_h if user_dt_depletion_h is not None else auto_dep
        final_integ = user_integrator      if user_integrator      is not None else auto_integ
        auto_tuned  = (user_dt_depletion_h is None or user_integrator is None)

        n_substeps = max(1, round(dt_output_h / final_dep))

        return DepletionParams(
            dt_depletion_h  = round(final_dep, 6),
            integrator       = final_integ,
            n_substeps       = n_substeps,
            band             = band,
            auto_tuned       = auto_tuned,
            dt_output_h      = dt_output_h,
        )


@dataclass
class DepletionParams:
    """
    Parâmetros de depleção resolvidos por DepletionAutoTuner.

    Exposto como dataclass para facilitar logging e serialização.
    """
    dt_depletion_h: float   # passo interno do integrador [h]
    integrador:     str = ""
    integrator:     str = ""   # alias en inglês para compatibilidade com simulation.py
    n_substeps:     int   = 1  # sub-passos por intervalo de output
    band:           str   = "" # faixa de decisão para log
    auto_tuned:     bool  = True
    dt_output_h:    float = 0.0

    def __post_init__(self) -> None:
        # Garantir consistência dos aliases
        if self.integrador and not self.integrator:
            self.integrator = self.integrador
        elif self.integrator and not self.integrador:
            self.integrador = self.integrator

    def log_summary(self, logger_fn) -> None:
        source = "auto-tune" if self.auto_tuned else "input usuário"
        logger_fn(
            "DepletionAutoTuner [%s, %s]: DT_output=%.1fh → "
            "DT_dep=%.1fh × %d sub-passos, integrador=%s",
            self.band, source,
            self.dt_output_h, self.dt_depletion_h,
            self.n_substeps, self.integrator,
        )
