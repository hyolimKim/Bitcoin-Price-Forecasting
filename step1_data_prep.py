# coding: utf-8
# 1. 필요한 라이브러리 임포트
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 머신러닝 및 PyTorch 라이브러리
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 유틸리티 함수 로드 (utils.py 파일이 필요합니다)
from utils import (
    load_bitcoin_data,
    create_features,
    prepare_data,
    LSTMModel,  # LSTM 모델 클래스
    train_pytorch_model,
    predict_pytorch_model,
    evaluate_model,
    plot_confusion_matrix,
    simulate_trading_strategy,
    calculate_buy_and_hold_return,
    compare_trading_strategies,
    plot_trading_results,
    device # cpu 또는 gpu
)

# 시각화 설정
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# 재현성을 위한 랜덤 시드 설정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("✅ 라이브러리 및 설정 완료!")
print(f"Using device: {device}")


# 2. 데이터 로드 및 피처 생성
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# 데이터 로드
btc_data = load_bitcoin_data(start_date=start_date, end_date=end_date)

# 피처(특성) 생성
lookback_days = 10  # 과거 10일치 데이터로 피처 생성
btc_features = create_features(btc_data.copy(), lookback_days=lookback_days)

print("\n✅ 데이터 로드 및 피처 생성 완료!")
print(f"데이터 기간: {btc_features.index.min()} ~ {btc_features.index.max()}")
print(f"생성된 피처 수: {len(btc_features.columns)}")
# btc_features.head() # 노트북에서 이 줄의 주석을 풀고 실행하여 데이터를 확인하세요.
