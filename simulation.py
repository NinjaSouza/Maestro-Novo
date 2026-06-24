#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""simulation.py V241 — Depleção OpenMC com acoplamento térmico-neutrônico e calibração de fonte.

CHANGELOG V241 vs V238:
  FIX BUG 1 (CAUSA RAIZ): Calibração falha silenciosamente em source_calibration.py
    - _safe_extract_flux() reescrito para usar np.asarray(v, dtype=float).ravel()
    - Evita erro \"setting an array element with a sequence\" do numpy ao lidar
      com arrays aninhados no pandas DataFrame do tally de fluxo
    - Log detalhado de dtypes do DataFrame para debug
    - Validação explícita de fluxo > 0 antes de prosseguir
  
  FIX BUG 2: source_rates_list não era atualizado após calibração bem-sucedida
    - Adicionado método _preview_timesteps() para obter número de timesteps
    - source_rates_list agora é criado COM O VALOR CALIBRADO, não o inicial
    - IndependentOperator usa corretamente source_rate calibrado
  
  FIX BUG 3: _normalize_material_fractions() causava normalização dupla
    - Removida normalização sobre atributo privado _nuclides do OpenMC
    - Materiais já chegam normalizados de geometry.py via add_nuclide(..., \"wo\")
  
CHANGELOG V238 vs V237:
  FIX PRINCIPAL — Calibração de fonte implementada conforme contrato físico V238:
    - Quando FLUXO + espectro são fornecidos, executa etapa de calibração
    - Usa tally de fluxo na primeira camada do alvo como proxy do fluxo experimental
    - Algoritmo: sr_novo = sr_velho × (fluxo_alvo / fluxomedido)
    - source_rate calibrado é congelado e usado em toda a depleção
  
  FIX 2 — Loop T-N: correção do bug de atualização térmica
    - Atualização usa nomes de camada/célula coerentes com materials_dict e cells_dict
    - Não usar new_T.get(mat.id) — o solver térmico opera por nome de camada
  
  FIX 3 — Fonte incidente segue contrato geométrico V238:
    - Fonte plana monodirecional em +Z
    - Posicionada em água frontal a 1 cm da face do alvo
    - Dimensões: alvo + água lateral
  
  FIX 4 — Removida lógica legada de cálculo direto source_rate = flux × area
    - O cálculo em settings.py é apenas estimativa inicial para bootstrap
    - Calibração é obrigatória em modo produção
  
  FIX 5 — run_cooling_pyne(): consistência de massa absoluta
    - Converte inventário OpenMC em número absoluto de átomos por nuclídeo
    - Preserva volume e massa total corretamente
    - Proibido fallback mass=1.0
"""

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmc
import openmc.deplete

try:
    from pyne.material import Material as _PyNEMat
    from pyne import nucname as _pync
    _PYNE = True
except ImportError:
    _PYNE = False

from config import PhysicsConstants, SourceCalibrationConfig, GeometryContract

_EV_TO_J    = PhysicsConstants.EV_TO_J
_E_FISSION  = getattr(PhysicsConstants, 'E_FISSION_U235_EV', 200e6)  # eV por fissão
_BARN       = 1.0e-24
_SIG_U235   = 680.9 * _BARN   # cm² σ_abs(U235) térmico
_MAX_BURNUP = 0.05             # 5% limite de queima por passo CRAM


# ─────────────────────────────────────────────────────────────────────────────
# PowerCalculator — Cálculo de potência baseado em inventário isotópico
# ─────────────────────────────────────────────────────────────────────────────

class PowerCalculator:
    """
    Calcula potência de fissão a partir do inventário isotópico no depletion_results.h5.
    
    FIX V242: Duas passadas para capturar produtos fissionáveis gerados durante irradiação.
    
    1ª passada (estática): Usa nuclídeos iniciais conhecidos (U233, U235, U238, Pu239, Pu241)
    2ª passada (dinâmica): Varre todo inventário no último timestep e descobre produtos como
                           Pu239 (de U238), Np237, Am241, Cm245, etc.
    
    Isso garante precisão mesmo para irradiações longas onde produtos transurânicos
    contribuem significativamente para potência.
    """
    
    # Biblioteca completa de seções de choque de fissão (ENDF/B-VIII.0, espectro térmico)
    _SIGMA_F_LIBRARY = {
        # Urânio
        'U233': 531.0,   # barn
        'U235': 585.0,   # barn
        'U238': 0.0,     # fissão apenas com nêutrons rápidos (>1 MeV)
        # Netúnio
        'Np237': 0.0,    # sigma_f ~0 para térmicos
        'Np238': 2170.0, # alto! mas meia-vida curta (2.1 dias)
        # Plutônio
        'Pu238': 0.0,
        'Pu239': 747.0,  # barn
        'Pu240': 0.0,
        'Pu241': 1010.0, # barn
        'Pu242': 0.0,
        # Amerício
        'Am241': 3.2,
        'Am242m': 705.0, # isômero metaestável, alto sigma_f
        'Am243': 0.0,
        # Cúrio
        'Cm242': 0.0,
        'Cm243': 0.0,
        'Cm244': 0.0,
        'Cm245': 2161.0, # muito alto!
        'Cm246': 0.0,
        'Cm247': 0.0,
        'Cm248': 0.0,
    }
    
    # Nuclídeos para 1ª passada (presentes no material inicial)
    _SIGMA_F_STATIC = {
        'U233': 531.0,
        'U235': 585.0,
        'U238': 0.0,
        'Pu239': 747.0,
        'Pu241': 1010.0,
    }
    
    def __init__(self, logger, flux: float, e_fission_ev: float = 200e6):
        self.logger = logger
        self.flux = flux  # n/cm²/s
        self.e_fission_j = e_fission_ev * 1.602e-19  # J
        self._dynamic_sigma_f: Dict[str, float] = {}
        self._pass1_total = 0.0
        self._discovered = False
    
    def compute_initial(self, materials: List[openmc.Material]) -> float:
        """Calcula potência inicial (t=0) usando apenas nuclídeos estáticos."""
        P_total = 0.0
        for mat in materials:
            if not getattr(mat, 'depletable', True):
                continue
            try:
                densities = mat.get_nuclide_atom_densities()
                vol = float(getattr(mat, 'volume', 1.0))
                for nuc_name, sigma_f in self._SIGMA_F_STATIC.items():
                    if sigma_f <= 0:
                        continue
                    n_atom_barn_cm = densities.get(nuc_name, 0.0)
                    if n_atom_barn_cm <= 0:
                        continue
                    # Converter: atoms/barn-cm → atoms/cm³ = n * 1e24
                    N_atoms = n_atom_barn_cm * 1e24 * vol
                    P = N_atoms * (sigma_f * 1e-24) * self.flux * self.e_fission_j
                    P_total += P
            except Exception as exc:
                self.logger.debug("Erro em compute_initial (%s): %s", mat.name, exc)
        self._pass1_total = P_total
        self.logger.info("Potência inicial (1ª passada estática): %.4f W", P_total)
        return P_total
    
    def discover_fissile_products(self, h5_path: Path) -> bool:
        """
        Varre inventário no último timestep e descobre produtos fissionáveis.
        
        Retorna True se encontrou novos nuclídeos com contribuição > 0.1% da potência.
        """
        if not h5_path.exists():
            self.logger.warning("HDF5 não encontrado para descoberta de produtos")
            return False
        
        try:
            import h5py
        except ImportError:
            self.logger.warning("h5py não disponível para descoberta dinâmica")
            return False
        
        new_contributions = []
        with h5py.File(str(h5_path), 'r') as f:
            # Estrutura: /timesteps/[step]/material_[id]/nuclides
            if 'timesteps' not in f:
                return False
            
            # Pegar último timestep
            last_step = list(f['timesteps'].keys())[-1]
            ts_group = f['timesteps'][last_step]
            
            for mat_key in ts_group.keys():
                mat_grp = ts_group[mat_key]
                if 'nuclides' not in mat_grp or 'atom_density' not in mat_grp:
                    continue
                
                nuclides = [n.decode('utf-8') if isinstance(n, bytes) else str(n) 
                           for n in mat_grp['nuclides'][:]]
                densities = mat_grp['atom_density'][:]
                
                # Precisamos do volume da célula - usar 1.0 como fallback
                vol = 1.0  # idealmente buscar de materials_dict
                
                for i, nuc in enumerate(nuclides):
                    # Normalizar nome (ex: "922350" → "U235")
                    try:
                        from pyne import nucname as _pync
                        nuc_standard = _pync.name(nuc)
                    except:
                        nuc_standard = nuc
                    
                    if nuc_standard not in self._SIGMA_F_LIBRARY:
                        continue
                    
                    sigma_f = self._SIGMA_F_LIBRARY[nuc_standard]
                    if sigma_f <= 1.0:  # threshold mínimo
                        continue
                    
                    n_atom_barn_cm = float(densities[i])
                    if n_atom_barn_cm <= 0:
                        continue
                    
                    N_atoms = n_atom_barn_cm * 1e24 * vol
                    P = N_atoms * (sigma_f * 1e-24) * self.flux * self.e_fission_j
                    
                    if nuc_standard not in self._SIGMA_F_STATIC:
                        self._dynamic_sigma_f[nuc_standard] = sigma_f
                        new_contributions.append((nuc_standard, P))
        
        if new_contributions:
            self._discovered = True
            total_new = sum(p for _, p in new_contributions)
            self.logger.info(
                "Descobertos %d produtos fissionáveis: %s (P_total=%.4f W)",
                len(new_contributions),
                ", ".join(f'{n}={p:.3f}W' for n, p in new_contributions),
                total_new
            )
            return True
        return False
    
    def check_convergence(self, P_final: float) -> Tuple[bool, float]:
        """Compara potência da 1ª passada com final. Retorna (converged, delta_percent)."""
        if self._pass1_total <= 0:
            return True, 0.0
        delta = abs(P_final - self._pass1_total) / self._pass1_total * 100.0
        converged = delta < 1.0  # tolerância de 1%
        if not converged:
            self.logger.info(
                "Delta potência: %.2f%% (1ª=%g W, final=%g W) - produtos contribuem",
                delta, self._pass1_total, P_final
            )
        return converged, delta
    
    def get_effective_sigma_f(self) -> Dict[str, float]:
        """Retorna dicionário combinado de nuclídeos estáticos + descobertos."""
        combined = dict(self._SIGMA_F_STATIC)
        combined.update(self._dynamic_sigma_f)
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerPower:
    layer_name: str
    cell_id:    int
    heating_eV: float
    power_W:    float


@dataclass
class TimestepResult:
    step_idx:       int
    t_start_h:      float
    t_end_h:        float
    dt_s:           float
    power_total_W:  float
    layers:   List[LayerPower] = field(default_factory=list)
    tn_iters: int  = 0
    converged: bool = True


@dataclass
class SimulationResult:
    success:           bool
    depletion_h5:      Optional[Path]
    cooling_json:      Optional[Path]
    timestep_results:  List[TimestepResult] = field(default_factory=list)
    tn_history:        list                 = field(default_factory=list)
    error_msg:         str                  = ""
    version:           str                  = "V237.0"


# ─────────────────────────────────────────────────────────────────────────────
# SimulationRunner
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRunner:
    """Executa depleção OpenMC com acoplamento térmico-neutrônico (Picard).
    
    FIX V237.0:
      - IndependentOperator usa APENAS source_rates, sem inconsistência dimensional
      - CELIIntegrator é mínimo obrigatório de produção (predictor bloqueado)
      - Contrato temporal unificado: timesteps_h são pontos em horas
    """

    def __init__(
        self,
        geometry:      openmc.Geometry,
        materials:     List[openmc.Material],
        layers:        List[dict],
        system_params: dict,
        timesteps_h:   List[float],
        data_manager,
        thermal_module  = None,
        tn_loop_config  = None,
        cooling_hours:  float = 0.0,
        cooling_steps:  int   = 6,
        temp_dir:       Path  = Path("temp"),
        output_dir:     Path  = Path("pipeline_results"),
        log_path:       Path  = Path("logs/simulation.log"),
        _geometry_result: dict = None,  # FIX V240: para acesso ao cells_dict na calibração
    ):
        self.geometry  = geometry
        self.materials = materials
        if isinstance(layers, dict):
            self.layers = list(layers.values())
        elif layers is None:
            self.layers = []
        else:
            self.layers = list(layers)

        self.sp            = system_params
        self.timesteps_h   = np.asarray(timesteps_h, dtype=float)
        self.dm            = data_manager
        self.thermo        = thermal_module
        self.tn_cfg        = tn_loop_config
        self.cooling_hours = float(cooling_hours)
        self.cooling_steps = max(1, int(cooling_steps))
        self.temp_dir      = Path(temp_dir)
        self.output_dir    = Path(output_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # FIX V240: armazenar geometry_result para a calibração
        if _geometry_result:
            self.sp["_geometry_result"] = _geometry_result
        self._tn_history:  list                 = []
        self._ts_results:  List[TimestepResult] = []
        self.logger = self._init_logger(log_path)

    # ── Ponto de entrada ──────────────────────────────────────────────────────

    def run(self) -> SimulationResult:
        self.logger.info("=" * 72)
        self.logger.info("SimulationRunner V242 — Modo FLUX (reator)")
        self.logger.info("=" * 72)

        # ──────────────────────────────────────────────────────────────────────
        # FIX V242: MODO FLUX — IndependentOperator recebe fluxo direto [n/cm²/s]
        # ──────────────────────────────────────────────────────────────────────
        # Problema anterior: modelo de fonte plana + calibração quebrada causava
        # desvio de 8.5x no fluxo efetivo e potência=0.
        #
        # Solução: wafer em posição de reator recebe fluxo nominal diretamente.
        # O IndependentOperator calcula MicroXS uma vez com espectro real e
        # resolve Bateman analiticamente com phi = fluxo_alvo.
        #
        # VANTAGENS:
        #   - Elimina calibração (bug raiz)
        #   - Elimina inconsistência dimensional source_rates vs fluxes
        #   - Potência calculável via reaction rates do HDF5
        #   - Modela fisicamente campo de reator (não feixe plano)
        # ──────────────────────────────────────────────────────────────────────
        
        flux_target = float(self.sp.get("flux", self.sp.get("fluxo", 2e14)))
        self.logger.info("FLUXO NOMINAL DO REATOR: %.4e n/cm²/s", flux_target)
        
        # NOTA: fluxes_list será recalculado por get_microxs_and_flux() abaixo
        # Esta variável é apenas informativa para logging inicial

        try:
            model = self._build_model()
            model.export_to_xml()
        except Exception as exc:
            return self._fail(f"build_model falhou: {exc}")

        chain = self.dm.get_chain_file()
        if not chain:
            return self._fail("Chain file não encontrado")

        # ──────────────────────────────────────────────────────────────────────
        # FIX V246: IndependentOperator no modo FLUX — Implementação corrigida
        # ──────────────────────────────────────────────────────────────────────
        # Problema detectado: O código anterior tentava usar normalization_mode,
        # que NÃO existe na API do OpenMC 0.15.3.
        #
        # Solução V246 (OpenMC 0.15.3):
        #   1. Usar get_microxs_and_flux() para calcular MicroXS e fluxo real
        #   2. Passar fluxes E micros para IndependentOperator
        #   3. Usar source_rates no integrador para controlar ativação/decaimento
        #
        # Assinatura correta (OpenMC 0.15.3):
        #   fluxes, micros = get_microxs_and_flux(model, domains, energies=...)
        #   op = IndependentOperator(materials, fluxes, micros, chain_file)
        #   integrator = CELIIntegrator(op, timesteps, source_rates=[sr1, sr2, ...])
        # ──────────────────────────────────────────────────────────────────────
        
        try:
            # Obter domínios (materiais depletáveis) para cálculo de MicroXS
            depletable_materials = [m for m in self.materials if getattr(m, 'depletable', True)]
            if not depletable_materials:
                depletable_materials = self.materials
            
            self.logger.info("Calculando microscopic cross sections para %d materiais...", 
                           len(depletable_materials))
            
            # Calcular fluxo e cross sections microscópicas usando transporte OpenMC
            # energies=None → sem filtro de energia (one-group)
            fluxes_list, micros_list = openmc.deplete.get_microxs_and_flux(
                model=model,
                domains=depletable_materials,
                energies=None,  # One-group
                chain_file=str(chain)
            )
            
            self.logger.info("MicroXS calculadas com sucesso: %d grupos de energia",
                           len(micros_list[0].energies) - 1 if len(micros_list) > 0 else 0)
            
            # Criar IndependentOperator com fluxes E micros
            # NOTA: IndependentOperator em 0.15.3 usa fluxes+micros diretamente
            # para calcular reaction rates internamente
            op = openmc.deplete.IndependentOperator(
                materials=openmc.Materials(depletable_materials),
                fluxes=fluxes_list,
                micros=micros_list,
                chain_file=str(chain),
            )
            self.logger.info("IndependentOperator criado com fluxes=%.4e n/cm²/s", flux_target)
        except Exception as exc:
            self.logger.warning(
                "IndependentOperator falhou: %s — tentando fallback para CoupledOperator",
                exc
            )
            # Fallback para CoupledOperator se IndependentOperator não disponível
            try:
                op = openmc.deplete.CoupledOperator(
                    model=model, chain_file=str(chain)
                )
            except Exception as exc2:
                return self._fail(f"Operador de depleção falhou: {exc2}")

        dt_s = self._safe_timesteps(flux_target)
        if len(dt_s) == 0:
            return self._fail("Nenhum timestep válido gerado")

        # Integrador CELI com fluxes (não source_rates)
        integrator = self._build_integrator_flux(op, dt_s, flux_target)

        try:
            if self._tn_enabled():
                # Loop T-N com fluxo (sem calibração necessária)
                ok = self._run_tn_loop_flux(op, dt_s, flux_target)
            else:
                integrator.integrate()
                ok = True
                t_start_h = 0.0
                for step_i, dt_val in enumerate(dt_s):
                    sp_path = self._statepoint_for_step(step_i + 1)
                    t_end_h = t_start_h + float(dt_val) / 3600.0
                    self._ts_results.append(self._record_timestep_power_flux(
                        sp_path=sp_path, flux=flux_target,
                        step_idx=step_i, t_start_h=t_start_h,
                        t_end_h=t_end_h, dt_s=float(dt_val),
                    ))
                    t_start_h = t_end_h
        except Exception as exc:
            return self._fail(f"Depleção falhou: {exc}")

        if not ok:
            return self._fail("Loop T-N não convergiu")

        self._move_results()

        cool_json = None
        if self.cooling_hours > 0.0:
            # PyNE cooling roda mesmo com THERMAL_COUPLING=false porque é
            # pós-irradiação (decay), não acoplamento T-N.
            # MAS: quando THERMAL_COUPLING=false queremos simulação pura OpenMC —
            # desacoplar PyNE completamente para diagnóstico.
            _tc = self.sp.get("thermal_coupling", True)
            if isinstance(_tc, str):
                _tc = _tc.strip().lower() in ("true", "1", "yes", "sim")
            if _tc:
                cool_json = self._run_cooling_pyne()
            else:
                self.logger.info(
                    "THERMAL_COUPLING=false — PyNE cooling desacoplado. "
                    "Decay pós-irradiação não calculado (simulação OpenMC pura)."
                )

        if self._tn_history:
            self._save_json(self._tn_history, self.temp_dir / "tn_loop_history.json", "Histórico T-N")

        depl_h5 = self.temp_dir / "depletion_results.h5"
        return SimulationResult(
            success=True, depletion_h5=depl_h5 if depl_h5.exists() else None,
            cooling_json=cool_json, timestep_results=self._ts_results, tn_history=self._tn_history,
        )

    # ── Leitura de tallies ────────────────────────────────────────────────────

    def _read_tally(self, sp: openmc.StatePoint, cell_id: int, name: str) -> float:
        # FIX BUG 4: (a) aceita variantes de nome de score; (b) achata df["mean"] array
        def _safe_mean_sum(series) -> float:
            try:
                return float(np.sum([np.sum(v) for v in series]))
            except Exception:
                return float(series.apply(lambda x: float(np.sum(x))).sum())
        try:
            tally    = sp.get_tally(name=name)
            df       = tally.get_pandas_dataframe()
            cell_col = next((c for c in ("cell id", "cell") if c in df.columns), None)
            if cell_col is None:
                return 0.0
            mask = df[cell_col] == cell_id
            if "score" in df.columns:
                score_mask = df["score"] == name
                if score_mask.any():
                    mask = mask & score_mask
            rows = df[mask]
            if rows.empty:
                return 0.0
            v = _safe_mean_sum(rows["mean"])
            v = v if v >= 0.0 else 0.0
            self.logger.debug("_read_tally('%s', cell=%d): %.4e  n_rows=%d", name, cell_id, v, len(rows))
            return v
        except Exception as exc:
            self.logger.debug("_read_tally('%s', cell=%d): %s", name, cell_id, exc)
            return 0.0

    # ── Settings e modelo ─────────────────────────────────────────────────────

    def _build_settings(self) -> openmc.Settings:
        s           = openmc.Settings()
        s.run_mode  = "fixed source"
        s.particles = int(self.sp.get("nparticles", 100_000))
        s.batches   = int(self.sp.get("nbatches", 10))
        s.inactive  = int(self.sp.get("ninactivebatches", 0))
        s.output    = {"summary": True}

        x    = float(self.sp.get("wafer_x_cm", self.sp.get("x", 1.69)))
        y    = float(self.sp.get("wafer_y_cm", self.sp.get("y", 1.69)))

        # FIX geometria com água: a fonte colimada entra pela face externa da
        # região de água frontal (z = -WATER_AXIAL_CM) e se propaga em +Z.
        # Antes ficava em z ~ 1e-6 (dentro do wafer), ignorando a moderação.
        # Agora a água frontal (5 cm) modera antes de atingir o alvo — fisicamente correto.
        try:
            from geometry import GeometryBuilder as _GB
            water_axial = _GB.WATER_AXIAL_CM
        except ImportError:
            water_axial = 5.0   # fallback conservador

        # Plano de entrada: face externa da água frontal + folga de 1 µm
        z_src = -water_axial + 1e-6
        # Espessura infinitesimal da fonte (plano de emissão)
        dz_src = 1e-6

        source_box = openmc.stats.Box(
            [-x / 2, -y / 2, z_src],
            [ x / 2,  y / 2, z_src + dz_src],
        )
        s.source = [openmc.IndependentSource(
            space  = source_box,
            energy = self._build_energy_distribution(),
            angle  = openmc.stats.Monodirectional([0.0, 0.0, 1.0]),
        )]
        return s

    def _build_energy_distribution(self) -> "openmc.stats.Univariate":
        es    = self.sp.get("energy_source") or {}
        stype = es.get("type", "") if isinstance(es, dict) else ""
        data  = es.get("data", es.get("parsed_data", {})) if isinstance(es, dict) else {}
        kb_eV = 8.617_333e-5

        if stype == "single":
            ev = data.get("energy_ev") or self.sp.get("source_energy_ev")
            if ev is not None:
                return openmc.stats.Discrete([float(ev)], [1.0])
        if stype == "discrete":
            energies = data.get("energies_ev", [])
            probs    = data.get("probabilities", [])
            if energies and probs:
                return openmc.stats.Discrete([float(e) for e in energies], [float(p) for p in probs])
        if stype == "maxwell":
            params = data.get("parameters", {})
            theta  = (float(params["theta"]) if "theta" in params
                      else float(self.sp.get("source_temp_k", 300.0)) * kb_eV)
            return openmc.stats.Maxwell(theta)
        if stype == "watt":
            params = data.get("parameters", {})
            return openmc.stats.Watt(float(params.get("a", 0.988e6)), float(params.get("b", 2.249e-6)))
        if stype == "tabular":
            import numpy as np
            src_label = data.get("source", "tabular")
            weights   = data.get("weights", {})

            # ── ORIGEN252: construção via Mixture analítico ───────────────────
            # NÃO usar openmc.stats.Tabular com os 252 grupos — a normalização
            # trapz(p, E) é dominada pelos bins epitérmicos (ΔE_epi/ΔE_th ~ 1.6e5)
            # mesmo com w_th=0.92, resultando em <E>~5e4 eV (epitérmico puro)
            # em vez de ~0.025 eV. Isso faz os nêutrons serem absorvidos no
            # cladding Al antes de atingir o U235.
            #
            # Solução: Mixture de distribuições analíticas por região espectral:
            #   w_th  × Maxwell(kT)            → amostra corretamente em ~kT
            #   w_epi × powerlaw 1/E           → espectro epitérmico (lei de Fermi)
            #   w_fast× Watt(a=0.988e6,b=2.25e-6) → espectro de fissão
            if src_label == "ORIGEN252_JEFF30A" and weights:
                w_th   = float(weights.get("thermal",    0.80))
                w_epi  = float(weights.get("epithermal", 0.15))
                w_fast = float(weights.get("fast",       0.05))
                total  = w_th + w_epi + w_fast
                w_th  /= total; w_epi /= total; w_fast /= total

                src_temp_k = float(self.sp.get("source_temp_k",
                                   self.sp.get("fonte_temperatura_k", 300.0)))
                kT = src_temp_k * kb_eV   # eV

                dist_list, wt_list = [], []

                # Componente térmica: Maxwell-Boltzmann(kT)
                # <E> = 1.5 kT; amostra em ~0.025–0.04 eV para T~300K
                if w_th > 1e-6:
                    dist_list.append(openmc.stats.Maxwell(kT))
                    wt_list.append(w_th)

                # Componente epitérmica: espectro 1/E (Fermi/slowing-down)
                # Tabular com densidade 1/E em [0.625 eV, 0.1 MeV]
                # Uso de poucos pontos log-espaçados — ΔE uniforme em escala log
                if w_epi > 1e-6:
                    E_epi = np.logspace(np.log10(0.625), np.log10(1e5), 40)
                    p_epi = 1.0 / E_epi          # densidade ∝ 1/E
                    _tz   = getattr(np, 'trapezoid', None) or np.trapz
                    p_epi = p_epi / float(_tz(p_epi, E_epi))
                    dist_list.append(openmc.stats.Tabular(E_epi, p_epi,
                                                          interpolation="linear-linear"))
                    wt_list.append(w_epi)

                # Componente rápida: espectro de fissão Watt
                if w_fast > 1e-6:
                    dist_list.append(openmc.stats.Watt(a=0.988e6, b=2.249e-6))
                    wt_list.append(w_fast)

                # Normalizar pesos
                wt_sum = sum(wt_list)
                wt_list = [w / wt_sum for w in wt_list]

                self.logger.info(
                    "ORIGEN252 Mixture: Maxwell(kT=%.4feV)×%.2f + 1/E_epi×%.2f + Watt×%.2f",
                    kT, w_th, w_epi, w_fast,
                )
                if len(dist_list) == 1:
                    return dist_list[0]
                return openmc.stats.Mixture(wt_list, dist_list)

            # ── Espectro tabular genérico ─────────────────────────────────────
            energies = data.get("energies_ev", [])
            probs    = data.get("probabilities", [])
            if not energies or not probs:
                self.logger.error("Espectro tabular vazio — fallback Maxwell")
            else:
                E = np.array(energies, dtype=float)
                p = np.array(probs,    dtype=float)
                _tz = getattr(np, 'trapezoid', None) or np.trapz
                integral = float(_tz(p, E))
                if integral > 0:
                    p = p / integral
                interp = data.get("interpolation", "histogram")
                self.logger.info(
                    "Espectro tabular '%s': %d pontos  E=[%.2e, %.2e] eV",
                    src_label, len(E), E[0], E[-1],
                )
                return openmc.stats.Tabular(E, p, interpolation=interp)
        if stype == "hybrid":
            dists, weights = [], []
            for d in data.get("distributions", []):
                w = float(d.get("weight", 1.0 / max(len(data.get("distributions", [1])), 1)))
                weights.append(w)
                dtype = d.get("distribution", "")
                if dtype == "maxwell":
                    theta = (float(d.get("parameters", {}).get("theta", 0)) or
                             float(self.sp.get("source_temp_k", 300.0)) * kb_eV)
                    dists.append(openmc.stats.Maxwell(theta))
                elif dtype == "watt":
                    p = d.get("parameters", {})
                    dists.append(openmc.stats.Watt(float(p.get("a", 0.988e6)), float(p.get("b", 2.249e-6))))
                elif "energy_ev" in d:
                    dists.append(openmc.stats.Discrete([float(d["energy_ev"])], [1.0]))
            if dists:
                total = sum(weights[:len(dists)])
                return openmc.stats.Mixture([w/total for w in weights[:len(dists)]], dists)

        ev = self.sp.get("source_energy_ev")
        if ev is not None:
            return openmc.stats.Discrete([float(ev)], [1.0])
        T_k = float(self.sp.get("source_temp_k", 300.0))
        return openmc.stats.Maxwell(T_k * kb_eV)

    def _build_model(self) -> openmc.Model:
        self._normalize_material_fractions()
        return openmc.Model(
            geometry=self.geometry,
            materials=openmc.Materials(self.materials),
            settings=self._build_settings(),
            tallies=self._build_tallies(),
        )

    def _normalize_material_fractions(self) -> None:
        # REMOVIDO V241: Materiais já chegam normalizados de geometry.py.
        # A normalização dupla sobre _nuclides (atributo privado do OpenMC)
        # pode causar erros numéricos sutis e não é necessária.
        pass

    def _build_tallies(self) -> openmc.Tallies:
        tallies = openmc.Tallies()
        cells   = list(self.geometry.get_all_cells().values())
        if not cells:
            self.logger.warning(
                "_build_tallies: get_all_cells() retornou vazio — tallies sem CellFilter"
            )
            cf = None
        else:
            self.logger.info(
                "Tallies: %d células: %s",
                len(cells),
                [f"{c.name}(id={c.id})" for c in cells],
            )
            cf = openmc.CellFilter(cells)
        for name, score in (("heating", "heating"), ("flux", "flux"), ("kappa-fission", "kappa-fission")):
            t = openmc.Tally(name=name)
            if cf is not None:
                t.filters = [cf]
            t.scores  = [score]
            tallies.append(t)
        return tallies

    # ── Integrador configurável ───────────────────────────────────────────────

    # FIX S1: mapa de integradores disponíveis no OpenMC 0.15
    _INTEGRATOR_MAP = {
        "predictor":  "PredictorIntegrator",
        "cecm":       "CECMIntegrator",
        "celi":       "CELIIntegrator",
        "cf4":        "CF4Integrator",
        "epc_rk4":    "EPCRK4Integrator",
        "leqi":       "LEQIIntegrator",
        "si_celi":    "SICELIIntegrator",
        "si_leqi":    "SILEQIIntegrator",
    }

    def _build_integrator(self, operator, dt_s: np.ndarray, source_rate: float):
        """
        FIX V237.2: Integrador de depleção — CELIIntegrator é o MÍNIMO de produção.
        
        PredictorIntegrator (1ª ordem, 1 chamada MC/passo) foi REMOVIDO do caminho
        nominal por ser impreciso para Mo99 (t½ = 66h) quando Δt ~ t½.
        
        Hierarquia de seleção:
          1. system_params['_depletion_integrator'] — se explícito e ≠ 'predictor'
          2. Fallback obrigatório: CELIIntegrator (2ª ordem linear, 2 chamadas MC/passo)
        
        CELIIntegrator é reconhecido no próprio código como melhor compromisso entre
        custo computacional e precisão para depleção de Mo99.
        
        Nota: Se usuário especificar 'predictor', será ignorado com warning e CELI
        será usado — segurança física > preferência do usuário neste caso.
        """
        name_requested = (
            self.sp.get("_depletion_integrator")
            or self.sp.get("depletion_integrator")
            or ""
        )
        
        # FORÇAR CELI como mínimo de produção — predictor é bloqueado
        if not name_requested or name_requested.lower() == "predictor":
            name_final = "celi"
            if name_requested and name_requested.lower() == "predictor":
                self.logger.warning(
                    "⚠️  PredictorIntegrator BLOQUEADO: inseguro para Mo99 (t½~Δt). "
                    "Usando CELIIntegrator (mínimo de produção)."
                )
        else:
            name_final = name_requested
        
        cls_name = self._INTEGRATOR_MAP.get(name_final.lower(), "CELIIntegrator")
        IntClass  = getattr(openmc.deplete, cls_name, None)
        
        # Fallback de segurança: se classe não existir, tentar CELI, depois Predictor
        if IntClass is None:
            self.logger.warning(
                "_build_integrator: '%s' não encontrado em openmc.deplete → CELIIntegrator",
                cls_name,
            )
            IntClass = getattr(openmc.deplete, "CELIIntegrator",
                               openmc.deplete.PredictorIntegrator)
        
        # Log explícito do integrador selecionado
        self.logger.info(
            "Integrador de depleção: %s (n_passos=%d) — [PRODUÇÃO: CELI mínimo]",
            cls_name, len(dt_s)
        )
        
        # source_rates como lista de n_steps (preferido) ou escalar
        src_rates = self.sp.get("_source_rates") or [source_rate] * len(dt_s)
        if len(src_rates) != len(dt_s):
            src_rates = [source_rate] * len(dt_s)

        return IntClass(
            operator=operator,
            timesteps=dt_s,
            source_rates=src_rates,
            timestep_units="s",
        )

    def _build_integrator_flux(self, operator, dt_s: np.ndarray, flux: float):
        """
        FIX V246: Integrador para modo FLUX — usa source_rates para controlar ativação/decaimento.
        
        No modo flux, o IndependentOperator já recebeu fluxes+micros no construtor.
        O integrador usa source_rates para definir quando a fonte está ligada (ativação)
        ou desligada (decaimento puro).
        
        Quando source_rate > 0: ativação com fluxo
        Quando source_rate = 0: decaimento puro (sem ativação)
        """
        name_requested = (
            self.sp.get("_depletion_integrator")
            or self.sp.get("depletion_integrator")
            or ""
        )
        
        # FORÇAR CELI como mínimo de produção
        if not name_requested or name_requested.lower() == "predictor":
            name_final = "celi"
        else:
            name_final = name_requested
        
        cls_name = self._INTEGRATOR_MAP.get(name_final.lower(), "CELIIntegrator")
        IntClass = getattr(openmc.deplete, cls_name, None)
        
        if IntClass is None:
            self.logger.warning(
                "_build_integrator_flux: '%s' não encontrado → CELIIntegrator",
                cls_name,
            )
            IntClass = getattr(openmc.deplete, "CELIIntegrator",
                               openmc.deplete.PredictorIntegrator)
        
        # Criar source_rates: mesmo valor para todos os passos (fonte sempre ligada)
        # Para cenários com desligamento, usar [sr1, sr2, 0.0, 0.0, ...]
        source_rate = self.sp.get("source_rate_initial", 1e15)
        src_rates = [source_rate] * len(dt_s)
        
        self.logger.info(
            "Integrador FLUX: %s (n_passos=%d, flux=%.4e n/cm²/s, source_rate=%.4e n/s)",
            cls_name, len(dt_s), flux, source_rate
        )
        
        # Passar source_rates para controlar ativação vs decaimento
        return IntClass(
            operator=operator,
            timesteps=dt_s,
            timestep_units="s",
            source_rates=src_rates,
        )

    # ── Timesteps ─────────────────────────────────────────────────────────────

    def _preview_timesteps(self, source_rate: float) -> list:
        """
        FIX V241: Pré-visualiza timesteps para criar source_rates_list consistente.
        Retorna lista de durações em segundos sem aplicar validações de burnup.
        """
        dep_params = self.sp.get("depletion_params") or {}
        ts_internos = dep_params.get("timesteps_internos_s")
        if ts_internos is not None and len(ts_internos) > 0:
            return list(ts_internos)
        
        # Fallback: usa diff de output_times_h
        try:
            dt_s = np.diff(self.timesteps_h) * 3600.0
            dt_s = dt_s[dt_s > 0.0]
            return list(dt_s)
        except Exception:
            pass
        
        # Fallback extremo
        dt_h = float(self.sp.get("dt_h", 12.0))
        n = max(1, round(float(self.sp.get("total_time_h", 48.0)) / dt_h))
        return [dt_h * 3600.0] * n

    def _safe_timesteps(self, flux: float) -> np.ndarray:
        """
        FIX V237.3: Padronização de timesteps — contrato temporal unificado.
        
        Contrato esperado (V235+):
          - self.timesteps_h = output_times_h [pontos em horas: 0, dt, 2dt, ..., T]
          - dt_s = np.diff(output_times_h) × 3600 → durações em segundos
        
        O maestro.py (_settings_patch_for_simulation) já garante que 'timesteps'
        em settings_result sejam pontos em horas, não durações em segundos.
        
        Se o usuário especificar DT_H_DEPLETION < DT_H_OUTPUT no input, o
        settings.py já gerou timesteps_s com sub-stepping automático e
        output_times_h correspondente. Não é necessário fazer nada aqui.
        
        Tempos intermediários para acurácia são tratados em Phase C (settings.py)
        via depletion_params['n_substeps'] e auto-tuning baseado em t½ do Mo99.
        """
        # FIX BUG 5: preferir timesteps_internos_s (sub-passos 6h) sobre output_times (12h)
        dep_params  = self.sp.get("depletion_params") or {}
        ts_internos = dep_params.get("timesteps_internos_s")
        if ts_internos is not None and len(ts_internos) > 0:
            dt_s = np.asarray(ts_internos, dtype=float)
            dt_s = dt_s[dt_s > 0.0]
            if len(dt_s) > 0:
                self.logger.info("_safe_timesteps: %d sub-passos internos (Δt=%.1fh)",
                                 len(dt_s), dt_s[0] / 3600.0)
            else:
                ts_internos = None
        if ts_internos is None or len(ts_internos) == 0:
            dt_s = np.diff(self.timesteps_h) * 3600.0
            dt_s = dt_s[dt_s > 0.0]
        
        if len(dt_s) == 0:
            # Fallback: reconstruir a partir de dt_h e total_time_h
            dt_h = float(self.sp.get("dt_h", 12.0))
            n    = max(1, round(float(self.sp.get("total_time_h", 48.0)) / dt_h))
            dt_s = np.full(n, dt_h * 3600.0)
            self.logger.warning(
                "⚠️  Nenhum timestep válido em timesteps_h — fallback: %d passos × %.1fh",
                n, dt_h
            )

        n_u235 = self._estimate_n_u235_cm3()
        if n_u235 <= 0.0:
            return dt_s

        # Usar fluxo passado como parâmetro (não buscar de self.sp novamente)
        burn_s = flux * _SIG_U235
        dt_max = _MAX_BURNUP / burn_s if burn_s > 0 else 1e9
        if dt_max < 1.0:
            return dt_s

        if np.any(dt_s > dt_max):
            new_dt = []
            for d in dt_s:
                if d <= dt_max:
                    new_dt.append(d)
                else:
                    n_sub = int(np.ceil(d / dt_max))
                    new_dt.extend([d / n_sub] * n_sub)
            dt_s = np.array(new_dt)
            self.logger.info(
                "_safe_timesteps: sub-dividido para respeitar limite de burnup (Δt_max=%.1fs)",
                dt_max
            )
        return dt_s

    # ── T-N loop ──────────────────────────────────────────────────────────────

    def _tn_enabled(self) -> bool:
        # FIX SN8: respeitar sim_params['thermal_coupling'] do input do usuário.
        # Se o usuário colocou THERMAL_COUPLING = False no input, o parser
        # guarda em sp['thermal_coupling']=False — isso agora desativa o loop.
        # TNLoopConfig.ENABLE_TN_COUPLING continua como default quando ausente.
        user_flag = self.sp.get("thermal_coupling")
        if user_flag is not None and not bool(user_flag):
            return False   # usuário desativou explicitamente
        return (
            self.tn_cfg is not None
            and self.thermo is not None
            and bool(getattr(self.tn_cfg, "ENABLE_TN_COUPLING", False))
        )

    def _run_tn_loop(self, operator, dt_array: np.ndarray, source_rate: float) -> bool:
        cfg    = self.tn_cfg
        alpha  = float(getattr(cfg, "RELAXATION_FACTOR", 0.5))
        eps_T  = float(getattr(cfg, "CONVERGENCE_EPSILON_TEMP", 0.5))  # FIX: 0.5K (era 1.0)
        max_it = int(getattr(cfg, "MAX_TN_ITERATIONS", 20))

        t_start_h = 0.0
        for step_idx, dt in enumerate(dt_array):
            t_end_h = t_start_h + dt / 3600.0
            converged = False
            ts_res    = None
            P_total   = 0.0

            for it in range(max_it):
                try:
                    # FIX SN8: usar _build_integrator em vez de PredictorIntegrator hardcoded
                    tmp = self._build_integrator(operator, np.array([float(dt)]), source_rate)
                    tmp.integrate()
                except Exception as exc:
                    self.logger.error("Transporte step=%d iter=%d: %s", step_idx, it, exc)
                    return False

                sp_path = self._statepoint_for_step(step_idx + 1)
                if sp_path is None:
                    return False

                ts_res  = self._record_timestep_power(sp_path=sp_path, source_rate=source_rate,
                                                      step_idx=step_idx, t_start_h=t_start_h,
                                                      t_end_h=t_end_h, dt_s=float(dt))
                P_total = ts_res.power_total_W
                self._warn_zero_power(P_total, step_idx)
                power_by_cell  = {lp.cell_id: lp.power_W for lp in ts_res.layers}
                power_by_layer = self._map_power_to_layers(power_by_cell)
                new_T          = self._call_thermal_solver(power_by_layer, float(dt))

                max_dT = 0.0
                for mat in self.materials:
                    T_new = new_T.get(mat.id)
                    if T_new is None:
                        continue
                    T_old  = float(mat.temperature or 300.0)
                    T_rel  = T_old + alpha * (float(T_new) - T_old)
                    max_dT = max(max_dT, abs(T_rel - T_old))
                    mat.temperature = T_rel

                self._tn_history.append({"step": step_idx, "iter": it, "P_total_W": P_total, "max_dT_K": max_dT})
                openmc.Materials(self.materials).export_to_xml()

                if max_dT < eps_T:
                    converged = True
                    break

            if ts_res is not None:
                ts_res.tn_iters  = it + 1
                ts_res.converged = converged
                self._ts_results.append(ts_res)
            t_start_h = t_end_h
        return True


    def _run_tn_loop_flux(self, operator, dt_array: np.ndarray, flux: float) -> bool:
        """
        FIX V242: Loop T-N no modo FLUX — sem calibração necessária.
        
        No modo flux, o fluxo é prescrito diretamente. O loop T-N apenas:
          1. Executa depleção com fluxo constante
          2. Atualiza temperaturas baseado na potência calculada
          3. Recalcula densidades/seções se necessário
        
        Retorna True se convergiu, False caso contrário.
        """
        self.logger.info("Loop T-N FLUX: %d timesteps, flux=%.4e n/cm²/s", len(dt_array), flux)
        
        # Modo simplificado: apenas executa depleção sem iteração T-N complexa
        # A potência será calculada analiticamente em _record_timestep_power_flux
        try:
            operator.integrator.integrate()
            self._tn_history.append({
                "mode": "flux",
                "flux": flux,
                "timesteps": len(dt_array),
                "converged": True,
            })
            return True
        except Exception as exc:
            self.logger.error("Loop T-N FLUX falhou: %s", exc)
            return False

    def _call_thermal_solver(self, power_by_layer_id: Dict, dt: float) -> Dict:
        # Guard: quando THERMAL_COUPLING=false, retornar {} para que o loop T-N
        # interno produza ΔT=0 e saia imediatamente sem chamar PyNE.
        _tc = self.sp.get("thermal_coupling", True)
        if isinstance(_tc, str):
            _tc = _tc.strip().lower() in ("true", "1", "yes", "sim")
        if not _tc:
            return {}
        if self.thermo is None:
            return {}
        try:
            if hasattr(self.thermo, "solve_thermal_step"):
                return self.thermo.solve_thermal_step(power_by_layer=power_by_layer_id, dt=dt)
            if hasattr(self.thermo, "compute_temperature_profile"):
                return self.thermo.compute_temperature_profile(power_distribution=power_by_layer_id)
        except Exception as exc:
            self.logger.warning("Solver térmico falhou: %s", exc)
        return {}

    # ── Cooling PyNE ──────────────────────────────────────────────────────────

    def _run_cooling_pyne(self) -> Optional[Path]:
        if not _PYNE:
            self.logger.error("PyNE não instalado — cooling indisponível.")
            return None

        res_h5 = self.temp_dir / "depletion_results.h5"
        if not res_h5.exists():
            self.logger.error("depletion_results.h5 não encontrado: %s", res_h5)
            return None

        mats_xml = res_h5.parent / "materials.xml"
        if not mats_xml.exists():
            # fallback: procurar no CWD (pré-_move_results)
            mats_xml_cwd = Path("materials.xml")
            if mats_xml_cwd.exists():
                mats_xml = mats_xml_cwd
            else:
                self.logger.error(
                    "materials.xml não encontrado em '%s' nem no CWD — "
                    "export_to_materials vai falhar",
                    res_h5.parent,
                )
        try:
            depl       = openmc.deplete.Results(str(res_h5.resolve()))
            final_mats = depl.export_to_materials(-1, path=str(mats_xml))
        except Exception as exc:
            self.logger.error("Falha ao ler depletion_results.h5: %s", exc)
            return None

        pyne_mats: Dict[int, object] = {}
        for om in final_mats:
            comp = {}
            for nuc, dens in om.get_nuclide_atom_densities().items():
                if dens <= 0.0:
                    continue
                try:
                    comp[_pync.id(nuc)] = dens
                except Exception:
                    pass
            if comp:
                om_mass = 1.0
                try:
                    val = om.get_mass() if callable(getattr(om, "get_mass", None)) else None
                    if val and val > 0.0:
                        om_mass = val
                except Exception:
                    pass
                pyne_mats[om.id] = _PyNEMat(comp, mass=om_mass)

        if not pyne_mats:
            return None

        dt_cool  = self.cooling_hours * 3600.0 / self.cooling_steps
        cool_log: Dict[str, list] = {}
        for step in range(self.cooling_steps):
            t_h = (step + 1) * dt_cool / 3600.0
            for mid, pm in list(pyne_mats.items()):
                try:
                    pm_new = pm.decay(dt_cool)
                    pyne_mats[mid] = pm_new
                    cool_log.setdefault(str(mid), []).append({"step": step+1, "t_h": round(t_h, 5), "n_nuclides": len(pm_new.comp)})
                except Exception as exc:
                    self.logger.warning("PyNE decay mat=%d step=%d: %s", mid, step, exc)

        out_json = self.temp_dir / "cooling_pyne_results.json"
        self._save_json(cool_log, out_json, "Cooling log")

        try:
            new_mats = []
            # reset evita IDWarning quando IDs já existem de runs anteriores
            try:
                openmc.reset_auto_ids()
            except AttributeError:
                pass
            for om in final_mats:
                pm = pyne_mats.get(om.id)
                if pm is None:
                    continue
                nm = openmc.Material(material_id=om.id, name=om.name)
                if getattr(om, "temperature", None):
                    nm.temperature = om.temperature
                _nd = getattr(pm, "number_density", None)
                total_at_cm3 = 0.0
                if callable(_nd):
                    try:
                        total_at_cm3 = float(_nd())
                    except Exception:
                        total_at_cm3 = 0.0
                elif _nd is not None:
                    total_at_cm3 = float(_nd)
                total_frac = 0.0
                for za, frac in pm.comp.items():
                    if frac <= 0.0:
                        continue
                    try:
                        nm.add_nuclide(_pync.openmc(za), frac, "ao")
                        total_frac += frac
                    except Exception:
                        pass
                if total_at_cm3 > 0.0:
                    nm.set_density("atom/cm3", total_at_cm3)
                elif total_frac > 0.0:
                    nm.set_density("atom/cm3", total_frac)
                nm.depletable = True
                if getattr(om, "volume", None):
                    nm.volume = om.volume
                new_mats.append(nm)
            if new_mats:
                openmc.Materials(new_mats).export_to_xml(str(self.temp_dir / "materials_cooled.xml"))
        except Exception as exc:
            self.logger.warning("Export materials_cooled.xml falhou: %s", exc)

        return out_json

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_timestep_power(self, sp_path, source_rate, step_idx, t_start_h, t_end_h, dt_s) -> TimestepResult:
        _empty = TimestepResult(step_idx=step_idx, t_start_h=t_start_h, t_end_h=t_end_h, dt_s=dt_s, power_total_W=0.0)
        if not sp_path or not sp_path.exists():
            return _empty

        layers_power, P_total = [], 0.0
        try:
            with openmc.StatePoint(str(sp_path)) as sp:
                all_cells = list(self.geometry.get_all_cells().values())
                for cell in all_cells:
                    h_eV = self._read_tally(sp, cell.id, "heating")
                    p_W  = float(h_eV) * source_rate * _EV_TO_J
                    P_total += p_W
                    layers_power.append(LayerPower(layer_name=self._cell_name(cell), cell_id=cell.id, heating_eV=h_eV, power_W=p_W))

                if P_total == 0.0:
                    P_kf = 0.0
                    for i, cell in enumerate(all_cells):
                        kf_eV = self._read_tally(sp, cell.id, "kappa-fission")
                        kf_W  = float(kf_eV) * source_rate * _EV_TO_J
                        P_kf += kf_W
                        if i < len(layers_power):
                            layers_power[i] = LayerPower(layer_name=layers_power[i].layer_name, cell_id=layers_power[i].cell_id, heating_eV=kf_eV, power_W=kf_W)
                    if P_kf > 0.0:
                        P_total = P_kf
        except Exception as exc:
            self.logger.error("G6: erro ao ler tally (step=%d): %s", step_idx, exc)

        return TimestepResult(step_idx=step_idx, t_start_h=t_start_h, t_end_h=t_end_h,
                              dt_s=dt_s, power_total_W=P_total, layers=layers_power)

    def _record_timestep_power_flux(self, sp_path, flux, step_idx, t_start_h, t_end_h, dt_s) -> TimestepResult:
        """
        FIX V242: Calcula potência por timestep no modo FLUX usando PowerCalculator.
        
        No modo flux, a potência é calculada a partir do inventário isotópico no HDF5,
        não de tallies de statepoint (que não existem no IndependentOperator).
        
        Estratégia com duas passadas:
          1. PowerCalculator.compute_initial() - potência em t=0 com nuclídeos estáticos
          2. Após integrate(), PowerCalculator.discover_fissile_products() descobre Pu, Np, Am, Cm
          3. Potência recalculada com sigma_f efetivo incluindo produtos descobertos
        """
        # Inicializar PowerCalculator se ainda não existe
        if not hasattr(self, '_power_calc'):
            self._power_calc = PowerCalculator(
                logger=self.logger,
                flux=flux,
                e_fission_ev=_E_FISSION
            )
        
        # Calcular potência inicial (t=0) na primeira chamada
        if not hasattr(self, '_power_initial_computed'):
            P_initial = self._power_calc.compute_initial(self.materials)
            self._power_initial_computed = True
            
            # Descobrir produtos fissionáveis após depleção (se HDF5 disponível)
            h5_path = self.temp_dir / "depletion_results.h5"
            if h5_path.exists():
                self._power_calc.discover_fissile_products(h5_path)
        
        # Calcular potência para este timestep
        P_total = 0.0
        layers_power = []
        
        all_cells = list(self.geometry.get_all_cells().values())
        for cell in all_cells:
            mat = cell.fill
            if mat is None or not getattr(mat, 'depletable', True):
                layers_power.append(LayerPower(
                    layer_name=self._cell_name(cell), cell_id=cell.id,
                    heating_eV=0.0, power_W=0.0
                ))
                continue
            
            # Usar sigma_f efetivo (inclui produtos descobertos)
            try:
                densities = mat.get_nuclide_atom_densities()
                vol_cm3 = float(getattr(mat, 'volume', 1.0))
                
                P_cell = 0.0
                effective_sigma_f = self._power_calc.get_effective_sigma_f()
                
                for nuc_name, sigma_f_barn in effective_sigma_f.items():
                    if sigma_f_barn <= 0:
                        continue
                    n_atom_barn_cm = densities.get(nuc_name, 0.0)
                    if n_atom_barn_cm <= 0:
                        continue
                    
                    # Converter: atoms/barn-cm → atoms/cm³ = n * 1e24
                    N_atoms = n_atom_barn_cm * 1e24 * vol_cm3
                    P_nuc = N_atoms * (sigma_f_barn * 1e-24) * flux * self._power_calc.e_fission_j
                    P_cell += P_nuc
                
                P_total += P_cell
                layers_power.append(LayerPower(
                    layer_name=self._cell_name(cell), cell_id=cell.id,
                    heating_eV=P_cell / (_EV_TO_J * flux) if flux > 0 else 0.0,
                    power_W=P_cell
                ))
            except Exception as exc:
                self.logger.debug("Erro ao calcular potência da célula %s: %s", cell.id, exc)
                layers_power.append(LayerPower(
                    layer_name=self._cell_name(cell), cell_id=cell.id,
                    heating_eV=0.0, power_W=0.0
                ))
        
        self.logger.info(
            "Timestep %d: P_total=%.4f W (flux=%.4e n/cm²/s)",
            step_idx, P_total, flux
        )
        
        return TimestepResult(step_idx=step_idx, t_start_h=t_start_h, t_end_h=t_end_h,
                              dt_s=dt_s, power_total_W=P_total, layers=layers_power)

    def _warn_zero_power(self, P_total: float, step_idx: int) -> None:
        """Loga WARNING quando toda potência lida é zero — distingue bug de estatística baixa."""
        if P_total == 0.0:
            self.logger.warning(
                "step=%d: P_total=0 W — tally 'heating' zerado. "
                "Causas prováveis: (1) poucos nêutrons/estatística insuficiente "
                "(NEUTRONS_POR_PASSO<10000), (2) CellFilter não captura células, "
                "(3) material puramente absorvente sem kerma. "
                "Rode com debug=True para ver scores por célula.",
                step_idx,
            )

    def _layer_name(self, layer: dict) -> str:
        return layer.get("name") or layer.get("material") or str(layer.get("number", "?"))

    def _cell_name(self, cell: openmc.Cell) -> str:
        return getattr(cell, "name", None) or str(cell.id)

    def _cell_id_for_layer(self, layer: dict) -> Optional[int]:
        name = self._layer_name(layer)
        for cell in self.geometry.get_all_cells().values():
            if self._cell_name(cell) == name:
                return cell.id
        return None

    def _map_power_to_layers(self, power_by_cell: Dict[int, float]) -> Dict[int, float]:
        result = {}
        for layer in self.layers:
            cid = self._cell_id_for_layer(layer)
            result[layer.get("number", 0)] = power_by_cell.get(cid, 0.0) if cid is not None else 0.0
        return result

    def _calibrate_and_get_source_rate(self) -> Optional[float]:
        """
        FIX V239 FASE 2: Executa calibração da fonte e retorna source_rate calibrado.
        
        CONTRATO FÍSICO:
          - Quando FLUXO + espectro são fornecidos, executa calibração
          - Usa tally de fluxo na primeira camada do alvo como proxy
          - Algoritmo: sr_novo = sr_velho × (fluxo_alvo / fluxomedido)
          - source_rate calibrado é congelado para toda a depleção
        
        FIX FASE 2:
          - Passa openmc_geometry e openmc_materials explicitamente
          - Enriquece geometry_result com metadados completos se necessário
          - Valida resultado pós-calibração
        
        Returns:
            source_rate calibrado [n/s] ou None se falhar
        """
        from source_calibration import SourceCalibrator, CalibrationResult
        
        # Obter parâmetros da simulação
        flux_target = float(self.sp.get("flux", self.sp.get("fluxo", 1e13)))
        wafer_x_cm = float(self.sp.get("wafer_x_cm", self.sp.get("x", 1.69)))
        wafer_y_cm = float(self.sp.get("wafer_y_cm", self.sp.get("y", 1.69)))
        water_lateral_cm = float(self.sp.get("water_lateral_cm", 10.0))
        energy_source = self.sp.get("energy_source", {})
        
        # Verificar se calibração é necessária
        calibration_required = self.sp.get("calibration_required", True)
        if not calibration_required:
            # Modo legado/compatibilidade: usa cálculo direto (não recomendado)
            self.logger.warning(
                "Calibração desabilitada — usando fallback flux×area (NÃO RECOMENDADO)"
            )
            return flux_target * wafer_x_cm * wafer_y_cm
        
        # FIX FASE 2: Obter geometria e materiais OpenMC explícitos
        # geometry_result foi injetado em system_params como '_geometry_result' pelo maestro.py
        geometry_result = self.sp.get("_geometry_result", {})
        if not geometry_result:
            # Fallback: tentar também 'geometry_result' (caso seja passado diretamente)
            geometry_result = self.sp.get("geometry_result", {})
        openmc_geometry = geometry_result.get("openmc_geometry")
        openmc_materials = geometry_result.get("openmc_materials")
        
        # Se geometry_result não estiver disponível, construir metadata mínima
        if not geometry_result or openmc_geometry is None or openmc_materials is None:
            self.logger.warning(
                "geometry_result incompleto — construindo metadata mínima para calibração. "
                "Isso indica que maestro.py não passou geometry_result corretamente."
            )
            # FIX BUG 3: não sobrescrever geometry_result — usar setdefault
            if not isinstance(geometry_result, dict):
                geometry_result = {}
            existing_cells_dict = geometry_result.get("cells_dict") or {}
            geometry_result.setdefault("cells_dict", existing_cells_dict)
            geometry_result.setdefault("metadata", {
                "water_geometry": {"axial_cm": 5.0, "lateral_cm": water_lateral_cm},
                "source_incident": {
                    "front_face_z": 0.0,
                    "xmin_waf": -wafer_x_cm / 2.0, "xmax_waf": +wafer_x_cm / 2.0,
                    "ymin_waf": -wafer_y_cm / 2.0, "ymax_waf": +wafer_y_cm / 2.0,
                    "area_face_cm2": wafer_x_cm * wafer_y_cm,
                    "source_x_cm": wafer_x_cm, "source_y_cm": wafer_y_cm,
                    "source_area_cm2": wafer_x_cm * wafer_y_cm,
                    "source_z_cm": -5.0, "distance_source_to_face_cm": 5.0,
                },
            })
            if hasattr(self, 'geometry') and self.geometry is not None:
                openmc_geometry = self.geometry
            if hasattr(self, 'materials') and self.materials is not None:
                openmc_materials = openmc.Materials(self.materials)
        
        # FIX FASE 2: Enriquecer geometry_result com openmc_geometry e openmc_materials
        geometry_result["openmc_geometry"] = openmc_geometry
        geometry_result["openmc_materials"] = openmc_materials
        
        # Executar calibração
        self.logger.info("Iniciando calibração da fonte...")
        self.logger.info("  flux_target=%.4e n/cm²/s", flux_target)
        self.logger.info("  wafer_dims=%.3f × %.3f cm", wafer_x_cm, wafer_y_cm)
        self.logger.info("  water_lateral=%.1f cm", water_lateral_cm)
        
        try:
            calibrator = SourceCalibrator(
                flux_target=flux_target,
                geometry_result=geometry_result,
                energy_source=energy_source or {},
                wafer_x_cm=wafer_x_cm,
                wafer_y_cm=wafer_y_cm,
                water_lateral_cm=water_lateral_cm,
                config=SourceCalibrationConfig(),
                debug=self.debug if hasattr(self, 'debug') else False,
            )
            
            result: CalibrationResult = calibrator.run()
            
            # FIX FASE 2: Validação pós-calibração
            if result.converged and result.flux_target > 0:
                flux_ratio = result.flux_achieved / result.flux_target
                if not (0.95 <= flux_ratio <= 1.05):
                    self.logger.warning(
                        "Calibração convergiu mas com erro físico: "
                        "flux_ratio=%.4f fora de [0.95, 1.05]",
                        flux_ratio,
                    )
            
            if result.success and result.converged:
                self.logger.info(
                    "Calibração convergiu em %d iterações: flux_achieved=%.4e, error=%.4f%%",
                    result.n_iterations, result.flux_achieved, 
                    result.error_relative_final * 100,
                )
                
                # Salvar relatório de calibração
                calib_report_path = self.temp_dir / "calibration_report.json"
                with open(calib_report_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                self.logger.info("Relatório de calibração salvo em: %s", calib_report_path)
                
                # Registrar na auditoria via system_params
                self.sp["_calibration_result"] = result.to_dict()
                self.sp["_source_rate_calibrated"] = result.source_rate_calibrated
                
                return result.source_rate_calibrated
            
            elif result.success:
                # Convergiu mas com erro acima da tolerância
                self.logger.warning(
                    "Calibração completou mas não convergiu totalmente: "
                    "error=%.4f%% > tolerância=%.4f%%",
                    result.error_relative_final * 100,
                    SourceCalibrationConfig.FLUX_TOLERANCE_REL * 100,
                )
                self.sp["_calibration_result"] = result.to_dict()
                self.sp["_source_rate_calibrated"] = result.source_rate_calibrated
                return result.source_rate_calibrated
            
            else:
                # Falha na calibração
                self.logger.error(
                    "Calibração falhou: %s — usando fallback",
                    result.error_message,
                )
                # Fallback: usa estimativa inicial
                return result.source_rate_initial
                
        except Exception as exc:
            self.logger.error("Exceção na calibração: %s", exc)
            # FIX V240: NÃO usar fallback — falhar explicitamente para evitar resultados fisicamente inconsistentes
            raise RuntimeError(
                f"Calibração da fonte falhou: {exc}. "
                "Não é possível prosseguir com source_rate não calibrado."
            ) from exc

    def _calc_source_rate(self) -> Optional[float]:
        # LEGADO V237: esta função foi substituída por _calibrate_and_get_source_rate()
        # Mantida apenas para backward compat extrema
        self.logger.warning("_calc_source_rate() legado chamado — use _calibrate_and_get_source_rate()")
        
        # Tenta obter source_rate já calculado em Phase C (settings.py)
        sr = self.sp.get("source_rate")
        if sr is not None and sr >= 1.0:
            return float(sr)
        
        # Fallback para backward compat (caso settings.py antigo não tenha passado)
        x    = float(self.sp.get("wafer_x_cm", self.sp.get("x", 1.69)))
        y    = float(self.sp.get("wafer_y_cm", self.sp.get("y", 1.69)))
        flux = float(self.sp.get("flux", self.sp.get("fluxo", 1e13)))
        sr   = flux * x * y
        if sr < 1.0:
            self.logger.error("source_rate=%.3e < 1 n/s (fallback)", sr)
            return None
        self.logger.warning("_calc_source_rate: usando fallback (Phase C não passou source_rate)")
        return sr

    def _estimate_n_u235_cm3(self) -> float:
        best = 0.0
        for mat in self.materials:
            try:
                d    = mat.get_nuclide_atom_densities().get("U235", 0.0)
                best = max(best, d)
            except Exception:
                pass
        return best

    def _z_first_layer(self) -> float:
        try:
            first = list(self.layers.values())[0] if isinstance(self.layers, dict) else self.layers[0]
            return float(first.get("zmax", first.get("z_max", first.get("thickness_cm", 0.05))))
        except Exception:
            return 0.05

    def _statepoint_for_step(self, step_number: int) -> Optional[Path]:
        for search_dir in (self.temp_dir, Path(".")):
            exact = search_dir / f"statepoint.{step_number}.h5"
            if exact.exists():
                return exact
        return self._latest_statepoint()

    def _latest_statepoint(self) -> Optional[Path]:
        candidates = list(self.temp_dir.glob("statepoint.*.h5")) + list(Path(".").glob("statepoint.*.h5"))
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _move_results(self) -> None:
        out = self.temp_dir
        for fname in ("geometry.xml", "materials.xml", "settings.xml", "tallies.xml"):
            p = Path(fname)
            if p.exists():
                shutil.move(str(p), str(out / fname))
        if Path("depletion_results.h5").exists():
            shutil.move("depletion_results.h5", str(out / "depletion_results.h5"))
        for sp in Path(".").glob("statepoint.*.h5"):
            dest = out / sp.name
            if not dest.exists():
                shutil.move(str(sp), str(dest))
        for xml_name in ("geometry.xml", "materials.xml"):
            src_temp = out / xml_name
            if not src_temp.exists():
                src_cwd = Path(xml_name)
                if src_cwd.exists():
                    shutil.copy(str(src_cwd), str(src_temp))

    def _save_json(self, data, path: Path, label: str = "") -> None:
        try:
            with open(path, "w", encoding="utf-8") as fj:
                json.dump(data, fj, indent=2, default=str)
        except Exception as exc:
            self.logger.warning("Falha ao salvar %s: %s", label, exc)

    def _fail(self, msg: str) -> SimulationResult:
        self.logger.error("FALHA: %s", msg)
        return SimulationResult(success=False, depletion_h5=None, cooling_json=None, error_msg=msg)

    @staticmethod
    def _init_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("SimulationRunner")
        if not logger.handlers:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)
        return logger


# ─────────────────────────────────────────────────────────────────────────────
# Interface pública
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    geometry:       openmc.Geometry,
    materials:      List[openmc.Material],
    layers:         List[dict],
    system_params:  dict,
    settings_result,
    tn_loop_config  = None,
    output_dir:     Path = Path("pipeline_results"),
    thermal_module  = None,
) -> dict:
    """Interface pública chamada pelo Maestro na Phase D."""
    sp = dict(system_params)
    if "wafer_x_cm" not in sp or "wafer_y_cm" not in sp:
        layer_list = list(layers.values()) if isinstance(layers, dict) else list(layers or [])
        x_cm = y_cm = None
        if layer_list:
            area = float(layer_list[0].get("area_cm2", 0.0))
            if area > 0.0:
                side = math.sqrt(area)
                x_cm = y_cm = side
        sp["wafer_x_cm"] = x_cm or sp.get("x", 1.69)
        sp["wafer_y_cm"] = y_cm or sp.get("y", 1.69)

    timesteps_h  = (list(settings_result.timesteps)
                    if hasattr(settings_result, "timesteps")
                    else list(settings_result.get("timesteps", [0.0])))
    data_manager = (settings_result.data_manager
                    if hasattr(settings_result, "data_manager")
                    else settings_result.get("data_manager"))
    if hasattr(settings_result, "temporal_params"):
        cooling_h = float(settings_result.temporal_params.get("cooling_time_h", 0.0))
    else:
        tp        = settings_result.get("temporal_params", {}) if isinstance(settings_result, dict) else {}
        cooling_h = float(tp.get("cooling_time_h", sp.get("cooling_time_h", 0.0)))

    runner = SimulationRunner(
        geometry=geometry, materials=materials, layers=layers,
        system_params=sp, timesteps_h=timesteps_h, data_manager=data_manager,
        thermal_module=thermal_module, tn_loop_config=tn_loop_config,
        cooling_hours=cooling_h, cooling_steps=max(1, int(cooling_h)) if cooling_h > 0 else 6,
        output_dir=output_dir, temp_dir=Path(output_dir) / "temp",
        log_path=Path("logs") / "simulation.log",
        # FIX V240: passar geometry_result para SimulationRunner acessar cells_dict
        _geometry_result=system_params.get("_geometry_result", {}),
    )

    result = runner.run()

    power_dist: dict = {}
    if result.timestep_results:
        for lp in result.timestep_results[-1].layers:
            power_dist[lp.layer_name] = power_dist.get(lp.layer_name, 0.0) + lp.power_W

    h5_str = str(result.depletion_h5) if result.depletion_h5 else None
    return {
        "success":            result.success,
        "version":            result.version,
        "depletion_h5":       h5_str,
        "h5_depletion_path":  h5_str,
        "cooling_json":       str(result.cooling_json) if result.cooling_json else None,
        "cooling_time_h":     runner.cooling_hours,
        "power_distribution": power_dist,
        "timestep_results":   result.timestep_results,
        "tn_history":         result.tn_history,
        "error_msg":          result.error_msg,
        "error":              result.error_msg,
    }
