import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # Quantile Regression을 위한 모듈
import matplotlib.font_manager as fm

# --- 1. 한글 폰트 설정 (필요시 수정) ---
try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 예시
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic' # macOS 예시
    except:
        plt.rcParams['font.family'] = 'NanumGothic' # Linux/Nanum 폰트 예시 (설치 필요)

plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# --- 2. 시계열 데이터 생성 ---
np.random.seed(42) # 재현성을 위해 시드 설정

n_days = 1000 # 시계열 데이터의 기간 (일)
time_index = np.arange(n_days)

# 2.1. 시장 변동성 프록시 (독립변수 1): 시간이 지남에 따라 증가하는 경향
# 이는 실제 시장의 변동성(VIX 지수 등)을 단순화한 시뮬레이션입니다.
# 나중에 수익률의 표준편차를 결정하는 데 사용됩니다.
market_volatility_proxy = 0.0005 * time_index + 0.01 # 초기 0.01, 점차 증가

# 2.2. 평균 수익률 (매우 작은 양의 값으로 설정)
mean_return_base = 0.0005 # 일일 평균 수익률 0.05%

# 2.3. 수익률 생성 (이분산성 포함)
# 수익률의 표준편차가 market_volatility_proxy에 따라 변동하도록 설정
# 즉, market_volatility_proxy가 높을수록 수익률의 변동폭이 커집니다.
# 이는 이분산성(heteroscedasticity)을 모방합니다.
returns = np.random.normal(loc=mean_return_base, scale=market_volatility_proxy, size=n_days)

# 2.4. 과거 수익률 (독립변수 2): VaR 모델에 사용할 설명 변수
# 과거 수익률이 현재 VaR에 영향을 미칠 수 있다는 가정 (모멘텀/회귀 효과)
lagged_returns = pd.Series(returns).shift(1)

# 데이터프레임으로 결합
data = pd.DataFrame({
    'returns': returns,
    'lagged_returns': lagged_returns,
    'market_volatility_proxy': market_volatility_proxy,
    'time': time_index
}).dropna() # lagged_returns의 NaN 값 제거

# --- 3. Quantile Regression을 이용한 VaR 산출 ---

# VaR 신뢰수준 설정 (예: 99% VaR -> 수익률 분포의 하위 1% 분위수)
quantile_for_VaR = 0.01 # 99% VaR에 해당하는 분위수 (1-0.99)

# VaR 모델 (수익률 = f(과거수익률, 시장변동성프록시))
# 'returns'는 종속변수, 'lagged_returns'와 'market_volatility_proxy'는 독립변수
# statsmodels의 QuantReg는 formula API를 지원하여 편리합니다.
formula = 'returns ~ lagged_returns + market_volatility_proxy'
model = smf.quantreg(formula, data)

# 0.01 분위수(VaR) 모델 추정
# 이 `fit` 함수가 비대칭적으로 가중된 절댓값 오차를 최소화하여 해당 분위수를 찾아줍니다.
result_qr_var = model.fit(q=quantile_for_VaR)

print(f"--- Quantile Regression for {quantile_for_VaR*100:.0f}% VaR ---")
print(result_qr_var.summary())
print("\n")

# 추정된 VaR 값 예측 (각 시점의 조건부 VaR)
# 예측된 수익률의 0.01 분위수 값이 곧 VaR (부호 반전 필요)
predicted_var_returns = result_qr_var.predict(data)
predicted_var = -predicted_var_returns # VaR는 손실 값이므로 양수로 표현하기 위해 음수 취함

# --- 4. VaR 시각화 ---
plt.figure(figsize=(15, 8))

# 실제 수익률 플롯
plt.plot(data['time'], data['returns'], label='일일 수익률', color='skyblue', alpha=0.7, linewidth=0.8)

# 추정된 VaR 라인 플롯
plt.plot(data['time'], -predicted_var, label=f'Quantile Regression {quantile_for_VaR*100:.0f}% VaR',
         color='red', linestyle='--', linewidth=2.5) # VaR는 손실이므로 음수 값으로 표시

# VaR 위반 (Exceedances) 시각화
# 실제 수익률이 VaR 선 아래로 내려가는 지점
exceedances = data[data['returns'] < -predicted_var]
plt.scatter(exceedances['time'], exceedances['returns'], color='purple', s=50, zorder=5, label='VaR 위반')

plt.title('Quantile Regression을 활용한 VaR 추정 (이분산성 데이터)', fontsize=18)
plt.xlabel('시간 (일)', fontsize=14)
plt.ylabel('수익률', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(data['returns'].min() * 1.1, data['returns'].max() * 1.1) # Y축 범위 조정
plt.tight_layout()
plt.show()

# --- 5. 간단한 백테스팅 (VaR 위반 횟수 및 비율) ---
total_observations = len(data)
num_exceedances = len(exceedances)
exceedance_rate = num_exceedances / total_observations

expected_exceedance_rate = quantile_for_VaR

print(f"--- VaR 백테스팅 결과 ({quantile_for_VaR*100:.0f}% VaR) ---")
print(f"총 관측치: {total_observations} 일")
print(f"VaR 위반 횟수: {num_exceedances} 회")
print(f"VaR 위반율: {exceedance_rate:.2%}")
print(f"기대 위반율: {expected_exceedance_rate:.2%}")

if abs(exceedance_rate - expected_exceedance_rate) < 0.005: # 오차 허용 범위
    print("VaR 위반율이 기대 위반율에 가깝습니다. 모델이 잘 작동하는 것으로 보입니다.")
else:
    print("VaR 위반율이 기대 위반율과 차이가 있습니다. 모델 재검토가 필요할 수 있습니다.")

# --- 추가: 다른 분위수의 계수 비교 (참고용) ---
# 이는 Quantile Regression이 각 분위수마다 다른 계수를 추정한다는 것을 보여줍니다.
# 예를 들어, 0.5 (중앙값) 분위수와 0.01 (VaR) 분위수의 계수를 비교
print("\n--- 계수 비교 (0.5 분위수 vs 0.01 분위수) ---")
result_qr_median = model.fit(q=0.5)
print(f"0.01 분위수 계수 (lagged_returns): {result_qr_var.params['lagged_returns']:.4f}")
print(f"0.50 분위수 계수 (lagged_returns): {result_qr_median.params['lagged_returns']:.4f}")
print(f"0.01 분위수 계수 (market_volatility_proxy): {result_qr_var.params['market_volatility_proxy']:.4f}")
print(f"0.50 분위수 계수 (market_volatility_proxy): {result_qr_median.params['market_volatility_proxy']:.4f}")
