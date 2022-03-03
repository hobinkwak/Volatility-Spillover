## Volatility Spillover

### Overview
- S&P500과 BTC/USD 간의 Rolling 표본 변동성 전이지수 계산
    - Net Spillover [Diebold-Yilmaz (2012)](https://ideas.repec.org/p/koc/wpaper/1001.html)
- 일간 로그수익률에 대해 ARIMA-GARCH 모델을 통한 일간 변동성 산출
    - ARIMA 모형의 잔차에 대한 GARCH 모형의 조건부 변동성

### Model
- ARIMA-GARCH
    - ARIMA : 로그수익률에 대한 Mean Term
    - GARCH : 로그수익률에 대한 잔차 Term
- Model Selection
    - BIC 기준 파라미터 Grid Search


### Implementation
```python
main.py
```

### Requirements
- python >= 3.7
```
arch==5.1.0
finance-datareader==0.9.31
matplotlib
numpy==1.19.3
pandas==1.3.4
scikit-learn==1.0.2
scipy==1.7.0
seaborn==0.11.2
statsmodels==0.13.2
tqdm==4.62.3
wandb==0.12.9
```

### Team

- 김다정
> email : ohhappy12@korea.ac.kr
> 
> github : github.com/githubhapi1
- 곽호빈
> email : high0802@naver.com
> 
> github : github.com/hobinkwak
