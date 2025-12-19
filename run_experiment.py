# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 유틸리티 함수 로드
from utils import (
    load_bitcoin_data,
    create_features,
    prepare_data,
    GRUModel,  # README에서 설명된 GRU 모델을 사용합니다.
    train_pytorch_model,
    predict_pytorch_model,
    evaluate_model,
    simulate_trading_strategy,
    calculate_buy_and_hold_return,
    compare_trading_strategies,
    plot_trading_results,
    print_trade_log,
    device
)

def create_sequences(X, y, sequence_length):
    """시계열 데이터를 시퀀스 형태로 변환"""
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X.iloc[i:i+sequence_length].values)
        y_sequences.append(y.iloc[i+sequence_length-1])
    return np.array(X_sequences), np.array(y_sequences)

def main():
    """메인 실행 함수"""
    print("--- 비트코인 예측 및 트레이딩 시뮬레이션 시작 ---")

    # --- 1. 데이터 로드 및 준비 ---
    print("\n[1/6] 데이터 로드 및 피처 생성...")
    btc_features = create_features(load_bitcoin_data('2020-01-01'))
    
    # 데이터 분할
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data(btc_features)

    # 스케일링
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=X_train_raw.columns)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), index=X_val_raw.index, columns=X_val_raw.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=X_test_raw.columns)
    
    # 시퀀스 데이터 생성 (README 기반 30일)
    SEQUENCE_LENGTH = 30
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)

    # PyTorch 텐서로 변환
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    # 데이터 로더 생성
    BATCH_SIZE = 64
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
    
    print("✅ 데이터 준비 완료")

    # --- 2. 모델 정의 ---
    print("\n[2/6] GRU 모델 정의...")
    INPUT_SIZE = X_train.shape[1]
    model = GRUModel(input_size=INPUT_SIZE).to(device)
    print(model)
    print("✅ 모델 정의 완료")

    # --- 3. 모델 학습 ---
    print("\n[3/6] 모델 학습 시작...")
    history = train_pytorch_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15)
    print("✅ 모델 학습 완료")

    # --- 4. 모델 평가 ---
    print("\n[4/6] 모델 성능 평가...")
    y_pred_probs, y_pred_labels = predict_pytorch_model(model, test_loader)
    
    # 평가를 위해 y_test의 길이를 예측 길이와 맞춤
    y_test_eval = y_test.iloc[SEQUENCE_LENGTH:]
    
    gru_eval = evaluate_model(y_test_eval, y_pred_labels, model_name="GRU Model")
    print("✅ 모델 평가 완료")

    # --- 5. 트레이딩 시뮬레이션 ---
    print("\n[5/6] 트레이딩 시뮬레이션 시작...")
    
    test_prices = btc_features['Close'].loc[y_test_eval.index]
    test_dates = test_prices.index
    
    # 전략 1: GRU 모델 예측 기반
    trading_results = simulate_trading_strategy(y_pred_labels, test_prices.values, test_dates, strategy_type='simple')

    # 전략 2: Buy and Hold (벤치마크)
    buy_and_hold_results = calculate_buy_and_hold_return(test_prices.values)

    print("✅ 시뮬레이션 완료")

    # --- 6. 결과 분석 및 비교 ---
    print("\n[6/6] 최종 결과 분석...")
    
    results_to_compare = {
        "GRU Model Strategy": trading_results,
        "Buy and Hold": {
            **buy_and_hold_results,
            'initial_capital': trading_results['initial_capital'],
            'final_value': buy_and_hold_results['final_value']
        }
    }
    
    compare_trading_strategies(results_to_compare)
    plot_trading_results(results_to_compare)
    print_trade_log(trading_results['trade_log'])
    print("\n--- 모든 작업 완료 ---")


if __name__ == '__main__':
    main()
