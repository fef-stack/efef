# ==========================================
# 2. 물리 엔진 연산 (Cross-Coupled Physics 반영)
# ==========================================
q = 1.6e-19
k_B = 1.38e-23
k_eV = 8.617e-5 # eV/K (Arrhenius equation 용)
eps_0 = 8.85e-14
eps_si = 11.7 * eps_0
eps_ox = 3.9 * eps_0
n_i = 1.5e10 
t_ox = 2e-7 
C_ox = eps_ox / t_ox

# [Coupling 1] N_A -> V_th 및 Q_dep 계산
phi_F = (k_B * T_K / q) * np.log(N_A / n_i)
Q_dep = np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) # 공핍층 전하량 (중요!)
W_dep = Q_dep / (q * N_A)
C_d = eps_si / W_dep

V_th0 = 0.4 + (2 * phi_F) + (Q_dep / C_ox)

# [Coupling 2] T & N_A -> Defect Generation (트랩 생성의 온도/도핑 의존성)
if stress_time > 0:
    # 1. HCI 메커니즘 (저온일수록, N_A가 높을수록 수평전계가 강해져 악화)
    # N_A가 1e17일 때를 기준으로 수평 전계 집중도 가중치 부여
    electric_field_factor = (N_A / 1e17) ** 0.5 
    hci_temp_factor = (300 / T_K) ** 1.5 # 저온에서 이동도 증가로 인한 가속화
    hci_trap = 2e10 * np.exp(1.0 * stress_vd) * (stress_time ** 0.5) * electric_field_factor * hci_temp_factor

    # 2. NBTI 메커니즘 (고온일수록 악화 - Arrhenius Model)
    E_a = 0.15 # Activation Energy (eV)
    bti_temp_factor = np.exp(-(E_a / k_eV) * (1/T_K - 1/300))
    bti_trap = 3e10 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5) * bti_temp_factor

    delta_trap_base = hci_trap + bti_trap
else:
    delta_trap_base = 0

# (UI 슬라이더 값 반영 로직은 기존과 동일)
N_it_total = 1e10 + (max(0.1, delta_trap_base*0.8/1e11) * 1e11) if 'Nit_slider' not in locals() else 1e10 + (Nit_slider * 1e11)
N_ot_total = (delta_trap_base*0.2/1e11 * 1e11) if 'Not_slider' not in locals() else (Not_slider * 1e11)

# 트랩 열화 반영
delta_Vth = (q * ((N_it_total - 1e10) + N_ot_total)) / C_ox
V_th_degraded = V_th0 + delta_Vth

SS_ideal = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * 1e10) / C_ox)
SS_degraded = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * N_it_total) / C_ox)

# I-V 및 Mobility 커브 연산
Vg_sweep = np.linspace(0.0, 1.5, 200)
Id_ideal, Id_degraded, mu_eff_array = [], [], []

for Vg in Vg_sweep:
    # [Coupling 3] N_A & Vg -> 수직 전계(E_eff) -> 표면 거칠기 산란(mu_sr)
    # Vg뿐만 아니라 도핑(Q_dep)이 수직 전계를 지배함
    Q_inv_ideal = C_ox * max(0, Vg - V_th0)
    Q_inv_degraded = C_ox * max(0, Vg - V_th_degraded)
    
    # Effective Field (E_eff) = (Q_dep + 0.5 * Q_inv) / eps_si (단위 보정 간략화)
    # 도핑이 높을수록 기본 E_eff가 매우 커짐
    E_eff_ideal = (Q_dep + 0.5 * Q_inv_ideal) / eps_si
    E_eff_degraded = (Q_dep + 0.5 * Q_inv_degraded) / eps_si

    # Matthiessen's Rule 이동도
    mu_ph = 300 * ((300 / T_K) ** 1.5)
    
    # E_eff에 기반한 엄밀한 Surface Roughness (상수 1e-12는 스케일링 팩터)
    mu_sr = 1000 / (1 + (E_eff_degraded * 1e-5)**2) 
    
    mu_coulomb = 1e16 / max(1e10, N_it_total + N_ot_total)
    
    mu_eff = 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb)
    mu_eff_array.append(mu_eff)
    
    # 이상적 커브
    if Vg < V_th0:
        Id_ideal.append(1e-12 * 10 ** ((Vg - V_th0) / SS_ideal))
    else:
        # Mobility도 이상적 상태(초기상태)의 값을 별도로 적용하는 것이 더 정확함
        mu_eff_ideal = 1 / (1/mu_ph + 1/(1000 / (1 + (E_eff_ideal * 1e-5)**2)) + 1/(1e16 / 1e10))
        Id_ideal.append(0.5 * mu_eff_ideal * C_ox * 5 * ((Vg - V_th0) ** 2) + 1e-12)
        
    # 열화 커브
    if Vg < V_th_degraded:
        Id_degraded.append(1e-12 * 10 ** ((Vg - V_th_degraded) / SS_degraded))
    else:
        Id_degraded.append(0.5 * mu_eff * C_ox * 5 * ((Vg - V_th_degraded) ** 2) + 1e-12)
