# **Stock Analysis Dashboard**
This dashboard enables visualization and analysis of OHLCV-derived stock and ETF data.

### **Features**
**Features include:**
1. Single stock/ETF Time Series
2. Inter-stock Analytics 
3. Simple Regression

*Derived financial indicators include: Returns (Daily, Cumulative, Rolling), Moving Averages (Simple & Exponential), True Range (Daily & Average), Historical Value at Risk (VaR), Sharpe Ratio.*
*Risk-Free Rate derived from U.S. 10 Year Treasury Yield.*

## **Single Stock Time Series**
Time Series for derived financial indicators. 
Ticker symbols are used to represent stocks.
Date range value determines the date range for displayed metrics. 

**Displayable metrics include:**
1. Close
2. Returns (Daily)
3. Cumulative Returns
4. Rolling Returns
5. Historical Value at Risk at 95% & 99% Confidence (VaR)
6. Simple Moving Averages (SMA)
7. Exponential Moving Averages (EMA)
8. Average True Range (ATR)

*Window options for Rolling Returns, VaR, SMA, EMA, ATR: [5, 10, 21, 63, 126, 252]*
*All rolling metrics use trading day windows.*
*Bollinger Bands can be toggled for SMA.*

## **Inter-Stock Analytics**
Visual comparisons across up to 5 tickers across three analytical categories:

**Performance**
1. Cumulative Returns Time Series
2. Rolling Returns Time Series
3. Returns Histogram with Kernel Density Estimation (KDE)

**Risk & Volatility**
1. Rolling Volatility Time Series
2. Historical Value at Risk Time Series at 95% or 99% Confidence
3. Sharpe Ratio Time Series

**Volume & Liquidity**
1. Daily Volume Time Series
2. Volatility against Volume Scatter Plot

*Date range value determines the date range for displayed metrics. *
*Window options for Rolling Returns, Risk & Volatility, Volatility against Volume: [5, 10, 21, 63, 126, 252]*
*Windows are in trading days.*

## **Regression**
Single stock simple linear regressions using Ordinary Least Squares (OLS)

**Regressions include:**
1. Forward Rolling Returns against Relative Volume
2. Daily Beta Regression using CAPM
3. Daily Alpha Regression using CAPM

*CAPM = Capital Asset Pricing Model*

### **Forward Rolling Returns against Relative Volume**
Regressing for predictive power of volume for returns.
*Relative Volume and Forward Rolling Returns share the same volume to respect temporal causality.*

**Relative Volume:**
- Normalised across options
- Captures market spikes
- Adjusts for baseline activity

**Forward Rolling Returns:**
- Allows forecasting
- Dissipates noise over rolling window

**Regression Details:**
- Independent variable Relative Volume undergoes Box-Cox transformation for normality under simple linear regression
- Data binned (Size = 50) to smooth relationships
- Outlier detected using Interquartile Range (IQR) and winsorized for each bin
- Bin means are used as observations
- R-squared scores are displayed

### Daily Beta/Alpha Regression using CAPM
Regressing excess stock returns against excess market returns for Alpha and Beta in Capital Asset Pricing Model.
*Market Returns derived from S&P 500 Index*

Windows for Beta Regression: 126 to 1260 trading days (6 months to 5 years), unless constrained.
Windows for Alpha Regression: 21 to 504 trading days (1 month to 2 years), unless constrained.

**Displayed Metrics:**
- Alpha and Beta Estimates
- R-squared
- 95% Confidence Intervals (CI)

**Statistical Assumptions:**
- Normality and Linearity of Residuals
- Parameter Estimates based on Maximum Likelihood Estimators (MLE) and Unbiased Estimates from sampling distribution

For Alpha, Student's t-test for a single mean is conducted to test hypothesis H0: Alpha = 0.
*Alphas are rarely significant for individual stocks.*

*Windows here refer to the number of trading days worth of data to regress, not the return window.*
*Daily Stock and Market Returns used.*