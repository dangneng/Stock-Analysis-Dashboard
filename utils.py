try:
    import os
    os.chdir("/Users/dangnengang/Documents/self_initiated_projects/dashboard") # manually setting directory (ipynb and py incompatibility)
    import io
    from datetime import datetime
    import numpy as np
    import scipy.stats
    from scipy.stats import gaussian_kde, boxcox
    import pandas as pd
    import matplotlib.colors as mcolors
    from dash import Dash, html, dcc, Input, Output, State
    import plotly.graph_objects as go
    import sklearn
    print("Import successful")
except ImportError as e:
    print("Import failed")
    print(e)

#INFORMATION
"""
Earliest Year: 1962(Stocks = ge), 1999(ETFs)
Reference Date: 1960-01-01, Function to convert date: getdate()
Reference Year for Inflation: 1960
"""

# GLOBAL VARIABLES
windows = [5, 10, 21, 63, 126, 252]
reference_date = datetime(1960, 1, 1)
global_date_format = "%Y-%m-%d"
regression_bin_size = 50
# SYMBOLS
stock_symbols = [s[0:-7] for s in os.listdir(os.path.join(os.getcwd(), "datasets/stocks"))]
etf_symbols = [s[0:-7] for s in os.listdir(os.path.join(os.getcwd(), "datasets/etfs"))]
stock_symbols.sort()
etf_symbols.sort()
all_symbols = stock_symbols.copy()
all_symbols.extend(etf_symbols)
all_symbols.sort()

# BASIC FUNCTIONS
def getdate(date: str) -> int:
    return int((datetime.strptime(date, global_date_format) - reference_date).days)
def getpath(symbol: str) -> str:
    try:
        assert type(symbol) == str, "Symbol must be a string"
        if symbol.casefold() in stock_symbols:
            return os.getcwd() + "/datasets/stocks/" + symbol.casefold() + ".us.txt"      
        elif symbol.casefold() in etf_symbols:
            return os.getcwd() + "/datasets/etfs/" + symbol.casefold() + ".us.txt"
        else:
            raise AssertionError("Invalid symbol")
    except AssertionError as e:
        print(f"getpath: {e}, returned None")
        return None
def checkpath(path: str) -> bool:
    return os.path.commonpath([os.path.abspath(path), "/Users/dangnengang/Documents/self_initiated_projects/dashboard/datasets"]) == "/Users/dangnengang/Documents/self_initiated_projects/dashboard/datasets"
def prepare_spx(path: str) -> pd.DataFrame:
    try:
        if not isinstance(path, str):
            raise ValueError("Path must be a string")

        if not checkpath(path) or not os.path.exists(path) or not os.path.isfile(path):
            raise FileNotFoundError("Invalid path")
        
        if os.path.getsize(path) == 0:
            raise ValueError("File is empty")

        df = pd.read_csv(path)

        # changing column names to small letters
        df.rename(columns={col:col.casefold().replace(" ", "_") for col in df.columns}, inplace=True)

        # changing date to days after 1960-01-01
        df["original_date"] = pd.to_datetime(df.iloc[:,0], format=global_date_format)
        df.iloc[:,0] = df.iloc[:,0].apply(getdate).astype("int64")

        # since df insists on taking on default dtypes
        df = df.convert_dtypes()
        df = df.sort_values("date")
        for col in [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]:
            df[col] = df[col].astype(str(df[col].dtype).casefold())
        
        # returns
        df["return"] = df["adj_close"].pct_change()
        
        # slicing dataset based on date
        df = df[df["date"] >= 0]
        df.reset_index(drop=True, inplace=True)

        return df
    except Exception as e:
        print(f"prepare_spx: {e}, returned None")
        return None
spx = pd.read_csv(os.path.join(os.getcwd(), "datasets/spx_df.csv"))

# CPI
cpi = pd.read_csv(os.path.join(os.getcwd(), "datasets/us_cpi.csv"))
# changing format of yearmon from %d-%m-%Y to %m-%Y
cpi.loc[:,"yearmon"] = cpi.loc[:,"yearmon"].apply(lambda x: x[3:])
cpi.set_index("yearmon", inplace=True)

# US TREASURY
treasury = pd.read_csv(os.path.join(os.getcwd(), "datasets/us_treasury_yields_daily.csv"))
treasury.rename(columns={col:col.casefold() for col in treasury.columns}, inplace=True)
for col in treasury.columns:
    if "us" in col:
        treasury[col] = treasury[col].interpolate() # filling nulls
    if pd.api.types.is_numeric_dtype(treasury[col]):
        treasury[col] = treasury[col]/100 # converting to decimals
    if "us" in col and "y" in col:
        treasury[f"{col}_rfr"] = treasury[col].apply(lambda x: (x + 1)**(1/360) - 1) # annual to daily rates
treasury["date"] = treasury["date"].apply(getdate)

# PREPARE
def prepare_df(path: str) -> pd.DataFrame:
    try:
        if not isinstance(path, str):
            raise ValueError("Path must be a string")

        if not checkpath(path) or not os.path.exists(path) or not os.path.isfile(path):
            raise FileNotFoundError("Invalid path")
        
        if os.path.getsize(path) == 0:
            raise ValueError("File is empty")

        df = pd.read_csv(path)

        # changing columns to small letters
        df.rename(columns={col:col.casefold() for col in df.columns}, inplace=True)

        # changing date to days after 1960-01-01
        df["original_date"] = pd.to_datetime(df.iloc[:,0], format=global_date_format)
        df.iloc[:,0] = df.iloc[:,0].apply(getdate).astype("int64")

        # since df insists on taking on default dtypes
        df = df.convert_dtypes() # resets dtypes
        df = df.sort_values("date")
        for col in [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]:
            df[col] = df[col].astype(str(df[col].dtype).casefold()) # prevent nullable dtypes in pandas

        # returns
        df["return"] = df["close"].pct_change()
        df["cum_return"] = (1 + df["return"]).cumprod()
        for window in windows:
            df[f"rolling_return_{window}"] = (1 + df["return"]).rolling(window).apply(lambda x: x.prod() - 1)

        # SMA (simple moving averages), bollinger bands and EMA (exponential moving averages)
        for window in windows:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"std_{window}"] = df["close"].rolling(window).std()
            df[f"upper_band_{window}"] = df[f"sma_{window}"] + df[f"std_{window}"]
            df[f"lower_band_{window}"] = df[f"sma_{window}"] - df[f"std_{window}"]
            df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
        
        # volatility: true range and average true range
        buffer = []
        for i in df.index:
            high_low = df.loc[i, "high"]-df.loc[i, "low"]
            if i > 0:
                high_close = abs(df.loc[i, "high"] - df.loc[i-1, "close"])
                low_close = abs(df.loc[i, "low"] - df.loc[i-1, "close"])
                buffer.append(max(high_low, high_close, low_close))
            else:
                buffer.append(high_low)
        df["true_range"] = buffer
        for window in windows:
            df[f"average_tr_{window}"] = df["true_range"].rolling(window).mean()

        # risk: historical var and sharpe ratio
        for window in windows:
            for ci in [0.95, 0.99]:
                quantile = 1-ci
                df[f"var_{ci}_{window}"] = df["return"].rolling(window=window).quantile(q=quantile, interpolation="lower")
        df["rfr"] = pd.merge(left=df.loc[:, ["date", "return"]], right=treasury.loc[:, ["date", "us10y_rfr"]], on="date", how="left")["us10y_rfr"]
        for window in windows:
            df[f"return_mean_{window}"] = df["return"].rolling(window).mean()
            df[f"rfr_mean_{window}"] = df["rfr"].rolling(window).mean() 
            df[f"return_std_{window}"] = df["return"].rolling(window).std()
            df[f"sharpe_{window}"] = (df[f"return_mean_{window}"] - df[f"rfr_mean_{window}"]) / df[f"return_std_{window}"]
        
        return df
    
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"prepare_df: {e}, returned None")
        return None

def winsorize_bins_iqr(series: pd.Series) -> pd.Series:
    c1 = series.quantile(0.25)
    c3 = series.quantile(0.75)
    iqr = c3-c1
    upper = c3+1.5*iqr
    lower = c1-1.5*iqr
    return series.clip(lower=lower, upper=upper)

def prepare_return_volume_df(symbol: str, window: int) -> tuple[pd.DataFrame, float]:
    try:
        df = prepare_df(getpath(symbol))[["volume", "close"]]

        # preparing dataset
        df["rvol"] = df["volume"]/df["volume"].rolling(window=window).mean() # rolling volume
        df["forward_returns"] = (df["close"].shift(-window) / df["close"]) - 1 # forward rolling returns
        df = df[["rvol", "forward_returns"]].dropna()
        df = df[(df["rvol"] != 0)]

        # boxcox
        boxcox_rvol, boxcox_lambda = boxcox(df["rvol"])
        df["boxcox_rvol"] = list(boxcox_rvol)

        # binning
        bins = int(len(df) / regression_bin_size)
        df["boxcox_rvol_bins"] = pd.qcut(df["boxcox_rvol"], q=bins)

        # winsorization by IQR
        df["winsorized_boxcox_rvol"] = df.groupby("boxcox_rvol_bins")["boxcox_rvol"].transform(winsorize_bins_iqr)

        return df, boxcox_lambda
    except Exception as e:
        print(f"prepare_return_volume_df: {e}, returned None, None")
        return None, None

def prepare_beta_alpha_df(symbol: str) -> pd.DataFrame:
    try:
        df = spx[["date", "original_date", "return"]].rename(columns={"return":"market_return"}).set_index("date").join(prepare_df(getpath(symbol)).set_index("date")["return"]).dropna()
        df.reset_index(inplace=True)
        df["us10y_rfr"] = pd.merge(left=df, right=treasury.loc[:, ["date", "us10y_rfr"]], on="date", how="left")["us10y_rfr"]
        for col in ["return", "market_return"]:
            df[f"excess_{col}"] = df[col] - df["us10y_rfr"]
        return df
    except Exception as e:
        print(f"prepare_beta_alpha_df: {e}, returned None")
        return None
    
def create_dashboard():
    app = Dash()
    metric_columns = {
        "Close":"close", 
        "Return":"return", 
        "Cumulative Return":"cum_return", 
        "Rolling Return":"rolling_return_",
        "Historical Value at Risk (95% CI)":"var_0.95_",
        "Historical Value at Risk (99% CI)":"var_0.99_",
        "Simple Moving Average":"sma_", 
        "Exponential Moving Average":"ema_", 
        "Average True Range":"average_tr_"
        }
    # styles
    backgroundColor = "#e1e6eb"
    contentColor = "#ffffff"
    border = f"1px solid {backgroundColor}"
    colorway = ["#8b29c4", "#55d9af", "#f0b71d", "#24f281", "#ebeb1c", "#e33434", "#1504b5", "#e61cdf", "#0b5731", "#3d2911", "#263e47"]
    color = "#a7a9ab"
    h1_color = "#292d2e"
    h3_color = "#656c6e"
    font_family = "Trebuchet MS"
    smallfont = "12px"
    mediumfont = "15px"
    largefont = "25px"
    dropdown_style = {
        "backgroundColor":contentColor,
        "fontSize":smallfont,
        "color":color,
        "font-family":font_family,
        "border":border,
        "borderColor":backgroundColor,
        "borderRadius":"0px",
        "height":"50px"
    }
    input_style = {
        "backgroundColor":contentColor,
        "fontSize":smallfont,
        "color":color,
        "font-family":font_family,
        "border":border,
        "borderColor":backgroundColor,
        "borderRadius":"0px",
        "height":"80px"
    }
    big_graph_style = {
        "backgroundColor":contentColor,
        "fontSize":smallfont,
        "color":color,
        "font-family":font_family,
        "border":border,
        "borderColor":backgroundColor,
        "borderRadius":"0px",
        "height":"430px"
    }
    small_graph_style = {
        "backgroundColor":contentColor,
        "fontSize":smallfont,
        "color":color,
        "font-family":font_family,
        "border":border,
        "borderColor":backgroundColor,
        "borderRadius":"0px",
        "flex":1,
        "height":"100%",
        "overflow": "hidden"
    }
    regression_style = {
        "display":"flex", 
        "flexDirection": "column",
        "height":"800px",
        "width":"33%"
        }
    checklist_style = { # displaying checklist
    "display":"block", # not hidden
    "maxHeight":"100px",
    "overflowY":"scroll",
    "fontSize":smallfont,
    "border":border,
    "borderColor":backgroundColor,
    "backgroundColor":contentColor,
    "padding":"5px"
    }
    h1_style = {
        "font-family":font_family,
        "color":h1_color,
        "fontSize":largefont,
        "textAlign":"center"
    }
    h2_style = {
        "font-family":font_family,
        "color":h1_color,
        "fontSize":mediumfont,
        "textAlign":"center"
    }
    h3_style = {
        "font-family":font_family,
        "color":h3_color,
        "fontSize":smallfont,
        "textAlign":"center"
    }

    app.layout = html.Div(
        style={
            "backgroundColor":backgroundColor, 
            "color":color, 
            "font-family":font_family
        },
        children=[
            html.Div(
                # FLEX
                style={
                    "display":"flex",
                },
                children=[
                    # LEFT SIDE
                    html.Div(
                        style={
                            "display":"flex",
                            "flexDirection":"column",
                            "gap":"5px",
                            "width":"40%",
                            "padding":"10px"
                        },
                        children=[
                            html.H1(
                                id="time-series",
                                children="Single Stock Time Series",
                                style=h1_style
                            ),
                            # stock-selector
                            dcc.Dropdown(
                                id="stock-selector",
                                placeholder="Stock",
                                options=[stock.upper() for stock in all_symbols],
                                style=dropdown_style
                            ),
                            dcc.Store(id="df-store"),
                            
                            # date-range-picker
                            dcc.DatePickerRange(
                                id="date-range-picker",
                                display_format="DD-MM-YYYY",
                                clearable=True
                            ),

                            # metric-selector
                            dcc.Dropdown( 
                                id="metric-selector",
                                placeholder="Metric",
                                options=list(metric_columns.keys()),
                                value=["close"],
                                multi=True,
                                style=dropdown_style
                            ),

                            # rolling-window-container
                            html.Div(
                                id="rolling-window-container",
                                style={
                                    "display":"none", # hidden
                                    **checklist_style
                                },
                                children=[
                                    dcc.Checklist( 
                                        id="rolling-window-selector",
                                        options=[],
                                        value=[]
                                    ),
                                ]
                            ),
                            
                            # bollinger-band-checklist
                            dcc.Checklist(
                                id="bollinger-band-checklist",
                                options=["Show Bollinger Bands"],
                                style=checklist_style
                            ),

                            # metric-chart
                            dcc.Graph(
                                id="metric-chart",
                                style=big_graph_style
                            ),
                        ]
                    ),

                    # RIGHT SIDE
                    html.Div(
                        style={
                            "display":"flex",
                            "flexDirection":"column",
                            "gap":"5px",
                            "width":"60%",
                            "padding":"10px"
                        },
                        children=[
                            # inter-stock analytics
                            html.H1(
                                children="Inter-Stock Analytics",
                                style=h1_style
                            ),

                            #  compare-stock-selector
                            dcc.Dropdown(
                                id="compare-stock-selector",
                                placeholder="Compare Stocks (Maximum 5)",
                                options=[stock.upper() for stock in all_symbols],
                                multi=True,
                                style=dropdown_style
                            ),
                            dcc.Store(id="compare-df-store"),

                            # date-range-picker-2
                            dcc.DatePickerRange(
                                id="date-range-picker-2",
                                display_format="DD-MM-YYYY",
                                clearable=True
                            ),

                            # performance - cum return & rolling return & daily return/kde histogram
                            html.Hr(),
                            html.H2(
                                children="Performance",
                                style=h2_style
                            ),
                            html.Div(
                                style={"display":"flex", "height":"300px"},
                                children=[
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            # cum-return-graph
                                            html.H2(
                                                children="Cumulative Returns",
                                                style=h2_style
                                            ),
                                            dcc.Graph(
                                                id="cum-return-graph",
                                                style=small_graph_style
                                            ),
                                        ]
                                    ),

                                    # rolling-return-chart
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Rolling Returns",
                                                style=h2_style
                                            ),
                                            dcc.Dropdown(
                                                id="cum-return-rolling-window-selector",
                                                options=windows,
                                                placeholder="Rolling Window",
                                                style=dropdown_style
                                            ),
                                            dcc.Graph(
                                                id="rolling-return-chart",
                                                style=small_graph_style
                                            )
                                        ]
                                    ),

                                    # return-kde-histogram
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Returns KDE",
                                                style=h2_style
                                            ),
                                            dcc.Graph(
                                                id="return-kde-histogram",
                                                style=small_graph_style
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # risk/volatility - rolling volatility & VaR & sharpe
                            html.Hr(),
                            html.H2(
                                children="Risk & Volatility",
                                style=h2_style
                            ),
                            # risk-rolling-window-selector
                            dcc.Dropdown(
                                id="risk-rolling-window-selector",
                                options=windows,
                                placeholder="Rolling Window",
                                style=dropdown_style
                            ),
                            html.Div(
                                style={"display":"flex", "height":"300px"},
                                children=[
                                    # rolling-volatility-chart
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Rolling Volatility",
                                                style=h2_style
                                            ),
                                            dcc.Graph(
                                                id="rolling-volatility-chart",
                                                style=small_graph_style
                                            ),
                                        ]
                                    ),

                                    # rolling-var-chart
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Historical VaR",
                                                style=h2_style
                                            ),
                                            dcc.Dropdown(
                                                id="var-confidence-level-selector",
                                                options=["95%", "99%"],
                                                placeholder="Confidence Level",
                                                style=dropdown_style
                                            ),
                                            dcc.Graph(
                                                id="rolling-var-chart",
                                                style=small_graph_style
                                            )
                                        ]
                                    ),

                                    # sharpe-ratio-chart
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Sharpe Ratio",
                                                style=h2_style
                                            ),
                                            dcc.Graph(
                                                id="sharpe-ratio-chart",
                                                style=small_graph_style
                                            ),
                                        ]
                                    ),
                                ]
                            ),

                            # volume/liquidity - daily volume & volatility vs volume
                            html.Hr(),
                            html.H2(
                                children="Volume & Liquidity",
                                style=h2_style
                            ),
                            html.Div(
                                style={"display":"flex", "height":"300px"},
                                children=[
                                    # daily-volume-chart
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Daily Volume",
                                                style=h2_style
                                            ),
                                            dcc.Graph(
                                                id="daily-volume-chart",
                                                style=small_graph_style
                                            ),
                                        ]
                                    ),

                                    # rolling-var-chart
                                    html.Div(
                                        style={"display":"flex", "flexDirection":"column", "flex":"1", "padding":"0px"},
                                        children=[
                                            html.H2(
                                                children="Volatility Against Volume",
                                                style=h2_style
                                            ),
                                            dcc.Dropdown(
                                                id="volatility-volume-rolling-window-selector",
                                                options=windows,
                                                placeholder="Volatility Window",
                                                style=dropdown_style
                                            ),
                                            dcc.Graph(
                                                id="volatility-volume-chart",
                                                style=small_graph_style
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # regression
            html.Hr(),
            html.H1(
                children="Regression",
                style=h1_style
            ),
            html.Div(
                style={
                    "display":"flex",
                },
                children=[
                    # forward returns against relative volume
                    html.Div(
                        style=regression_style,
                        children=[
                            html.H2(
                                id="returns-volume-regression-title",
                                children="Forward Rolling Returns against Relative Volume",
                                style=h2_style
                            ),
                            dcc.Dropdown(
                                id="returns-volume-regression-stock",
                                placeholder="Stock",
                                options=[stock.upper() for stock in all_symbols],
                                style=dropdown_style
                            ),
                            dcc.Input(
                                id="returns-volume-regression-window",
                                type="number",
                                placeholder="  Window",
                                min=1,
                                max=300,
                                step=1,
                                style=input_style
                            ),
                            html.H3(
                                children="Window: Min 1, Max 300 unless constrained",
                                style=h3_style
                            ),
                            html.H3(
                                id="returns-volume-regression-r2",
                                children="R-squared: NA, Box-cox Lambda: NA",
                                style=h3_style
                            ),
                            dcc.Graph(
                                id="returns-volume-regression-graph",
                                style=small_graph_style
                            )
                        ]
                    ),
                    # beta regression
                    html.Div(
                        style=regression_style,
                        children=[
                            html.H2(
                                id="beta-regression-title",
                                children="Daily Beta Regression",
                                style=h2_style
                            ),
                            dcc.Dropdown(
                                id="beta-regression-stock",
                                placeholder="Stock",
                                options=[stock.upper() for stock in all_symbols],
                                style=dropdown_style
                            ),
                            dcc.Input(
                                id="beta-regression-window",
                                type="number",
                                placeholder="  Window",
                                min=126,
                                max=1260,
                                step=1,
                                style=input_style
                            ),
                            html.H3(
                                id="beta-regression-window-description",
                                children="Min window: NA, Max window: NA",
                                style=h3_style
                            ),
                            dcc.DatePickerSingle(
                                id="beta-regression-date",
                                placeholder="Date",
                                min_date_allowed=None,
                                max_date_allowed=None,
                                display_format="DD-MM-YYYY",
                                clearable=True
                            ),
                            html.H3(
                                id="beta-regression-date-description",
                                children="Min date: NA, Max date: NA",
                                style=h3_style
                            ),
                            dcc.Store(id="beta-regression-df"),
                            dcc.Store(id="beta-regression-dates"),
                            html.H3(
                                id="beta-regression-graph-description",
                                children="Daily Beta: NA, R-squared: NA",
                                style=h3_style
                            ),
                            html.H3(
                                id="beta-regression-graph-confidence",
                                children="Beta 95% CI: NA",
                                style=h3_style
                            ),
                            dcc.Graph(
                                id="beta-regression-graph",
                                style=small_graph_style
                            )
                        ]
                    ),
                    # alpha regression
                    html.Div(
                        style=regression_style,
                        children=[
                            html.H2(
                                id="alpha-regression-title",
                                children="Daily Alpha Regression",
                                style=h2_style
                            ),
                            dcc.Dropdown(
                                id="alpha-regression-stock",
                                placeholder="Stock",
                                options=[stock.upper() for stock in all_symbols],
                                style=dropdown_style
                            ),
                            dcc.Input(
                                id="alpha-regression-window",
                                type="number",
                                placeholder="  Window",
                                min=21,
                                max=504,
                                step=1,
                                style=input_style
                            ),
                            html.H3(
                                id="alpha-regression-window-description",
                                children="Min window: NA, Max window: NA",
                                style=h3_style
                            ),
                            dcc.DatePickerSingle(
                                id="alpha-regression-date",
                                placeholder="Date",
                                min_date_allowed=None,
                                max_date_allowed=None,
                                display_format="DD-MM-YYYY",
                                clearable=True
                            ),
                            html.H3(
                                id="alpha-regression-date-description",
                                children="Min date: NA, Max date: NA",
                                style=h3_style
                            ),
                            dcc.Store(id="alpha-regression-df"),
                            dcc.Store(id="alpha-regression-dates"),
                            html.H3(
                                id="alpha-regression-graph-description",
                                children="Daily Alpha: NA, R-squared: NA",
                                style=h3_style
                            ),
                            html.H3(
                                id="alpha-regression-graph-hypothesis",
                                children="Alpha 95% CI: NA, p-value for (H0: Alpha = 0): NA",
                                style=h3_style
                            ),
                            dcc.Graph(
                                id="alpha-regression-graph",
                                style=small_graph_style
                            )
                        ]
                    )      
                ]
            )  
        ]
        )
    
    # LEFT 
    # storing df and renaming title
    @app.callback(
            Output("df-store", "data"),
            Output("time-series", "children"),
            Input("stock-selector", "value")
    )
    def store_df(stock_name):
        if stock_name and stock_name.casefold() in all_symbols:
            return prepare_df(getpath(stock_name.casefold())).to_json(date_format="iso", orient="split"), f"{stock_name} Single Stock Time Series"
        else:
            return None, "Single Stock Time Series"
    # date-range-picker values
    @app.callback(
            Output("date-range-picker", "start_date"),
            Output("date-range-picker", "end_date"),
            Output("date-range-picker", "min_date_allowed"),
            Output("date-range-picker", "max_date_allowed"),
            Input("df-store", "data")
    )
    def date_range_picker_values(json_df):
        if not json_df:
            return None, None, None, None
        else:
            df = pd.read_json(io.StringIO(json_df), orient="split")[["original_date"]]
        earliest_date = df["original_date"].values[0]
        latest_date = df["original_date"].values[-1]
        return earliest_date, latest_date, earliest_date, latest_date
    # display rolling window checklist and options
    @app.callback(
            Output("rolling-window-container", "style"),
            Output("rolling-window-selector", "options"),
            Input("metric-selector", "value")
    )
    def display_rolling_window_checklist(metrics):
        if any(metric in ["Rolling Return", "Simple Moving Average", "Exponential Moving Average", "Average True Range", "Historical Value at Risk (95% CI)", "Historical Value at Risk (99% CI)"] for metric in metrics):
            options = []
            for metric in metrics:
                if metric in ["Rolling Return", "Simple Moving Average", "Exponential Moving Average", "Average True Range", "Historical Value at Risk (95% CI)", "Historical Value at Risk (99% CI)"]:
                    options.extend([f"{metric} - {window} Day Window" for window in windows])
            return checklist_style, options     
        style = checklist_style.copy()
        style['display'] = "none"
        return style, windows  
    # display bollinger band checklist
    @app.callback(
            Output("bollinger-band-checklist", "style"),
            Input("rolling-window-selector", "value")
    )
    def display_bollinger_band_checklist(metrics_and_windows):
        sma_present = False
        # checking if sma present
        for metric_and_window in metrics_and_windows:
            if "Simple Moving Average" in metric_and_window:
                sma_present = True
                break
        if sma_present:
            return checklist_style
        else:
            style = checklist_style.copy()
            style["display"] = "none"
            return style
    # updating metric chart
    @app.callback(
            Output("metric-chart", "figure"), # id="line_chart", parameter="figure"
            Input("metric-selector", "value"), # id="column_selector", parameter="value"
            Input("rolling-window-selector", "value"), # window
            Input("df-store", "data"), # df
            Input("bollinger-band-checklist", "value"), # y/n bollinger bands
            Input("date-range-picker", "start_date"), 
            Input("date-range-picker", "end_date")
    )
    def update_graph(metrics, metrics_and_windows, json_df, show_bb, start, end):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            legend_groupclick="togglegroup",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )

        # null handling
        if isinstance(metrics, str): # if only one column selected
            metrics = [metrics]
        if isinstance(metrics_and_windows, int):
            metrics_and_windows = [metrics_and_windows]
        if not json_df or not start or not end:
            return fig
        else: # reading and slicing df
            df = pd.read_json(io.StringIO(json_df), orient="split")
            df["original_date"] = pd.to_datetime(df["original_date"])
            df = df[(df["original_date"] >= pd.to_datetime(start)) & (df["original_date"] <= pd.to_datetime(end))] # slicing rows by date


        # adding metrics with windows
        count = 0
        for metric_and_window in metrics_and_windows:
            metric = next((metric for metric in ["Rolling Return", "Simple Moving Average", "Exponential Moving Average", "Average True Range", "Historical Value at Risk (95% CI)", "Historical Value at Risk (99% CI)"] if metric in metric_and_window))
            window = next((window for window in windows if f" {str(window)} " in metric_and_window))
            column = f"{metric_columns.get(metric)}{window}"
            if column in df.columns and metric in metrics:
                # sma and bollinger bands
                if "sma" in column and show_bb: 
                    upper_column = f"upper_band_{window}"
                    lower_column = f"lower_band_{window}"
                    legend_key = column.upper()

                    count += 1

                    # colors 
                    rgb_sma_color = mcolors.to_rgb(colorway[count%len(colorway)])
                    line_color = f"rgb({int(rgb_sma_color[0]*255)}, {int(rgb_sma_color[1]*255)}, {int(rgb_sma_color[2]*255)})"
                    fill_color = f"rgba({int(rgb_sma_color[0]*255)}, {int(rgb_sma_color[1]*255)}, {int(rgb_sma_color[2]*255)}, 0.2)"
                    
                    # sma
                    fig.add_scatter(x=df["original_date"], y=df[column], mode="lines", name=legend_key,
                                    line=dict(color=line_color),
                                    legendgroup=legend_key,
                                    hovertemplate="%{y:.4f}"
                                    )
                    
                    # bollinger bands
                    fig.add_scatter(x=df["original_date"], y=df[upper_column], mode="lines", name=upper_column.upper(), 
                                    line=dict(color=line_color, width=0), # hides line
                                    showlegend=False,
                                    legendgroup=legend_key,
                                    hovertemplate="%{y:.4f}"
                                    )
                    fig.add_scatter(x=df["original_date"], y=df[lower_column], mode="lines", name=lower_column.upper(),
                                    line=dict(color=line_color, width=0), # hides line
                                    showlegend=False,
                                    fill="tonexty",
                                    fillcolor=fill_color,
                                    legendgroup=legend_key,
                                    hovertemplate="%{y:.4f}"
                                    )

                # other metrics with windows
                else:
                    fig.add_scatter(x=df["original_date"], y=df[column], mode="lines", name=column.upper(), hovertemplate="%{y:.4f}")
                    count += 1

        # adding metrics without windows
        for metric in metrics:
            if metric not in ["Simple Moving Average", "Exponential Moving Average", "Average True Range", "Historical Value at Risk (95% CI)", "Historical Value at Risk (99% CI)"]:
                column = f"{metric_columns.get(metric)}"
                if column in df.columns:
                    fig.add_scatter(x=df["original_date"], y=df[column], mode="lines", name=column.upper(), hovertemplate="%{y:.4f}")
                    count += 1

        return fig

    # REGRESSION
    # FORWARD RETURNS AGAINST RELATIVE VOLUME
    # update title and max window
    @app.callback(
            Output("returns-volume-regression-title", "children"),
            Output("returns-volume-regression-window", "min"),
            Output("returns-volume-regression-window", "max"),
            Input("returns-volume-regression-stock", "value"),
    )
    def update_title_window(stock_name):
        if stock_name and stock_name.casefold() in all_symbols:
            df = prepare_df(getpath(stock_name))

            # checking maximum window length for 1000 observations (20 bins of bin size 50)
            max_window = int(((len(df) - 1000)/2) + 1)

            if max_window < 1: # not enough observations
                return f"{stock_name.upper()}: Forward Rolling Returns against Relative Volume (Stock not enough observations)", 0, 0
            elif max_window < 300: # if maximum window is lesser than default maximum 300
                return f"{stock_name.upper()}: Forward Rolling Returns against Relative Volume (Max Window: {max_window})", 1, max_window
            else:
                return f"{stock_name.upper()}: Forward Rolling Returns against Relative Volume (Max Window: 300)", 1, 300
        else:
            return "Forward Rolling Returns against Relative Volume", 1, 300
    # update returns-volume-regression-graph
    @app.callback(
            Output("returns-volume-regression-r2", "children"),
            Output("returns-volume-regression-graph", "figure"),
            Input("returns-volume-regression-stock", "value"),
            Input("returns-volume-regression-window", "value")
    )
    def update_returns_volume_regression_graph(stock_name, window):
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        score = "R-squared: NA, Box-cox Lambda: NA"
        if not stock_name:
            return score, fig
        elif stock_name.casefold() not in all_symbols:
            return score, fig
        elif not window:
            return score, fig
        elif window > int(((len(prepare_df(getpath(stock_name))) - 1000)/2) + 1) or window < 1:
            return score, fig
        
        df, boxcox_lambda = prepare_return_volume_df(stock_name, window)

        x = pd.DataFrame(df.groupby("boxcox_rvol_bins")["boxcox_rvol"].mean())
        y = pd.DataFrame(df.groupby("boxcox_rvol_bins")["forward_returns"].mean())

        # model
        model = sklearn.linear_model.LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        model_score = model.score(x, y)
        score = f"R-squared: {model_score:.5f}, Box-cox Lambda: {boxcox_lambda:.5f}"

        # graphing
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x.squeeze(), y=y.squeeze(),
            mode="markers",
            hovertemplate="Observation: %{y:.2f}, %{x:.2f}<extra></extra>",
            name="Observations"
        ))
        fig.add_trace(go.Scatter(
            x=x.squeeze(), y=y_pred.squeeze(),
            mode="lines",
            hovertemplate="OLS Line: %{y:.2f}, %{x:.2f}<extra></extra>",
            name="OLS Line"
        ))
        fig.update_layout(
            xaxis_title="Winsorized box-cox RVOL",
            yaxis_title="Forward rolling returns"
        )
        return score, fig

    # BETA REGRESSION
    # store df, dates and update title, min-max
    @app.callback(
            # storage
            Output("beta-regression-df", "data"),

            # title
            Output("beta-regression-title", "children"),

            # minx-max
            Output("beta-regression-window", "min"),
            Output("beta-regression-window", "max"),
            Output("beta-regression-date", "min_date_allowed"),
            Output("beta-regression-date", "max_date_allowed"),

            Input("beta-regression-stock", "value")
    )
    def beta_regression_1(stock_name):
        basic_title = "Daily Beta Regression"
        min = 126
        max = 1260
        min_date_allowed = None
        max_date_allowed = None
        if not stock_name:
            return None, basic_title, min, max, min_date_allowed, max_date_allowed
        elif not isinstance(stock_name, str):
            return None, basic_title, min, max, min_date_allowed, max_date_allowed
        stock_name = stock_name.casefold()
        if stock_name not in all_symbols:
            return None, basic_title, min, max, min_date_allowed, max_date_allowed

        df = prepare_beta_alpha_df(stock_name)
        dates = list(pd.to_datetime(df["original_date"], format=global_date_format).dt.date)
        title = f"{stock_name.upper()}: Daily Beta Regression ({len(dates)} observations)"

        # min, max window + dates
        if len(dates) < 126:
            min = 0
            max = 0
            title = f"{stock_name.upper()}: Daily Beta Regression (Not enough observations)"
        else:
            min_date_allowed = dates[125]
            max_date_allowed = dates[-1]

        if len(dates) < 1260 and len(dates) >= 126:
            max = len(dates)


        df = df.to_json(date_format="iso", orient="split")
        return df, title, min, max, min_date_allowed, max_date_allowed
    # update descriptions
    @app.callback(
            Output("beta-regression-window-description", "children"),
            Input("beta-regression-window", "min"),
            Input("beta-regression-window", "max")
    )
    def beta_regression_2(min_window, max_window):
        if min_window == 0 and max_window == 0:
            return "Min window: NA, Max window: NA"
        else:
            return f"Min window: {min_window}, Max window: {max_window}"
    @app.callback(
            Output("beta-regression-date-description", "children"),
            Input("beta-regression-date", "min_date_allowed"),
            Input("beta-regression-date", "max_date_allowed")
    )
    def beta_regression_3(min_date, max_date):
        if not min_date and not max_date:
            return "Min date: NA, Max date: NA"
        else:
            return f"Min date: {min_date}, Max date: {max_date}"
    # update graph
    @app.callback(
            Output("beta-regression-graph", "figure"),
            Output("beta-regression-graph-description", "children"),
            Output("beta-regression-graph-confidence", "children"),
            Input("beta-regression-window", "value"),
            Input("beta-regression-date", "date"),
            State("beta-regression-df", "data")
    )
    def beta_regression_4(window, date, json_df):
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        description = "Daily Beta: NA, R-squared: NA"
        confidence = "Beta 95% CI: NA"
        if not window or not date or not json_df:
            return fig, description, confidence
        
        df = pd.read_json(io.StringIO(json_df), orient="split")

        loop_attempts = 0
        while date not in list(df["original_date"]) and loop_attempts < 5:
            date = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            loop_attempts += 1

        if loop_attempts >= 5:
            description = "Daily Beta: NA, R-squared: NA (Invalid Date)"
            confidence = "Beta 95% CI: NA (Invalid Date)"
            return fig, description, confidence

        end_idx = df[df["original_date"] == date].index[0]
        start_idx = end_idx - window + 1

        if start_idx < 0:
            description = "Daily Beta: NA, R-squared: NA (Not enough observations)"
            confidence = "Beta 95% CI: NA (Not enough observations)"
            return fig, description, confidence

        df = df.iloc[start_idx:end_idx]

        x = df["excess_market_return"]
        x_df = pd.DataFrame(df["excess_market_return"])
        y = df["excess_return"]

        model = sklearn.linear_model.LinearRegression()
        model.fit(x_df, y)
        pred_y = model.predict(x_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=pred_y,
            mode="lines"
        ))
        fig.update_layout(
            xaxis_title="Excess Market Returns",
            yaxis_title="Excess Stock Returns"
        )

        beta = model.coef_[0]
        score = model.score(x_df, y)
        description = f"Daily Beta: {beta:.5f}, R-squared: {score:.5f}"

        # standard error of y: se_y
        residual = y-pred_y
        n = len(x)
        degree_of_freedom = n - 2 # 2dfs lost to estimating intercept and coefficient
        se_y = np.sum(residual**2)/degree_of_freedom
        # variance of x: variance_x
        x_mean = np.mean(x)
        variance_x = np.sum((x-x_mean)**2)/n
        # se_beta
        variance_beta = se_y/(n*variance_x)
        se_beta = np.sqrt(variance_beta)

        # confidence interval
        t_crit = scipy.stats.t.ppf(0.975, df=degree_of_freedom) # critical points of t-distribution at df=df and 95% CI
        ci_lower = beta - t_crit*se_beta
        ci_upper = beta + t_crit*se_beta

        confidence = f"Beta 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"

        return fig, description, confidence

    # ALPHA
    # store df, dates and update title, min-max
    @app.callback(
            # storage
            Output("alpha-regression-df", "data"),

            # title
            Output("alpha-regression-title", "children"),

            # minx-max
            Output("alpha-regression-window", "min"),
            Output("alpha-regression-window", "max"),
            Output("alpha-regression-date", "min_date_allowed"),
            Output("alpha-regression-date", "max_date_allowed"),

            Input("alpha-regression-stock", "value")
    )
    def alpha_regression_1(stock_name):
        basic_title = "Daily Alpha Regression"
        min = 21
        max = 504
        min_date_allowed = None
        max_date_allowed = None
        if not stock_name:
            return None, basic_title, min, max, min_date_allowed, max_date_allowed
        elif not isinstance(stock_name, str):
            return None, basic_title, min, max, min_date_allowed, max_date_allowed
        stock_name = stock_name.casefold()
        if stock_name not in all_symbols:
            return None, basic_title, min, max, min_date_allowed, max_date_allowed

        df = prepare_beta_alpha_df(stock_name)
        dates = list(pd.to_datetime(df["original_date"], format=global_date_format).dt.date)
        title = f"{stock_name.upper()}: Daily Alpha Regression ({len(dates)} observations)"

        # min, max window + dates
        if len(dates) < 21:
            min = 0
            max = 0
            title = f"{stock_name.upper()}: Daily Alpha Regression (Not enough observations)"
        else:
            min_date_allowed = dates[min-1]
            max_date_allowed = dates[-1]

        if len(dates) < 504 and len(dates) >= 21:
            max = len(dates)


        df = df.to_json(date_format="iso", orient="split")
        return df, title, min, max, min_date_allowed, max_date_allowed
    # update descriptions
    @app.callback(
            Output("alpha-regression-window-description", "children"),
            Input("alpha-regression-window", "min"),
            Input("alpha-regression-window", "max")
    )
    def alpha_regression_2(min_window, max_window):
        if min_window == 0 and max_window == 0:
            return "Min window: NA, Max window: NA"
        else:
            return f"Min window: {min_window}, Max window: {max_window}"
    @app.callback(
            Output("alpha-regression-date-description", "children"),
            Input("alpha-regression-date", "min_date_allowed"),
            Input("alpha-regression-date", "max_date_allowed")
    )
    def alpha_regression_3(min_date, max_date):
        if not min_date and not max_date:
            return "Min date: NA, Max date: NA"
        else:
            return f"Min date: {min_date}, Max date: {max_date}"
    # update graph
    @app.callback(
            Output("alpha-regression-graph", "figure"),
            Output("alpha-regression-graph-description", "children"),
            Output("alpha-regression-graph-hypothesis", "children"),
            Input("alpha-regression-window", "value"),
            Input("alpha-regression-date", "date"),
            State("alpha-regression-df", "data")
    )
    def alpha_regression_4(window, date, json_df):
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        description = "Daily Alpha: NA, R-squared: NA"
        hypothesis = "Alpha 95% CI: NA, p-value for (H0: Alpha = 0): NA"
        if not window or not date or not json_df:
            return fig, description, hypothesis
        
        df = pd.read_json(io.StringIO(json_df), orient="split")

        loop_attempts = 0
        while date not in list(df["original_date"]) and loop_attempts < 5:
            date = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            loop_attempts += 1
        
        if loop_attempts >= 5:
            description = "Daily Alpha: NA, R-squared: NA (Invalid Date)"
            hypothesis = "Alpha 95% CI: NA, p-value for (H0: Alpha = 0): NA (Invalid Date)"
            return fig, description, hypothesis

        end_idx = df[df["original_date"] == date].index[0]
        start_idx = end_idx - window + 1

        if start_idx < 0:
            description = "Daily Alpha: NA, R-squared: NA (Not enough observations)"
            hypothesis = "Alpha 95% CI: NA, p-value for (H0: Alpha = 0): NA (Not enough observations)"
            return fig, description, hypothesis

        df = df.iloc[start_idx:end_idx]

        x = df["excess_market_return"]
        x_df = pd.DataFrame(df["excess_market_return"])
        y = df["excess_return"]

        model = sklearn.linear_model.LinearRegression()
        model.fit(x_df, y)
        pred_y = model.predict(x_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=pred_y,
            mode="lines"
        ))
        fig.update_layout(
            xaxis_title="Excess Market Returns",
            yaxis_title="Excess Stock Returns"
        )

        alpha = model.intercept_
        score = model.score(x_df, y)
        description = f"Daily Alpha: {alpha:.5f}, R-squared: {score:.5f}"

        # test-statistic: t = alpha/SEalpha
        # values needed: se_y, variance_x, mean
        n = len(x)
        # mean
        x_mean = np.mean(x)
        # variance of x
        variance_x = np.sum((x-x_mean)**2)/n
        # se_y (unbiased not MLE)
        residual = y-pred_y
        degree_of_freedom = n - 2 # 2dfs lost to estimating intercept and coefficient
        se_y = np.sum(residual**2)/degree_of_freedom
        # SE (not SD, as CI is the interval of true parameter using the prediction of the sample, so we want to replicate the population)
        variance_alpha = (1/n + (x_mean**2)/(n*variance_x)) * se_y
        se_alpha = np.sqrt(variance_alpha)

        # confidence interval
        t_crit = scipy.stats.t.ppf(0.975, df=degree_of_freedom) # critical points of t-distribution at df=df and 95% CI
        ci_lower = alpha - t_crit*se_alpha
        ci_upper = alpha + t_crit*se_alpha

        # test-statistic
        t_alpha = alpha/se_alpha
        # p-value
        p_value = 2 * scipy.stats.t.sf(np.abs(t_alpha), df=degree_of_freedom)

        if p_value >= 0.05:
            hypothesis = [
                f"Alpha 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}],",
                html.Br(),
                f"p-value for (H0: Alpha = 0): {p_value:.4f} (Threshold c = 0.05, Null hypothesis accepted)"
            ]
        else:
            hypothesis = [
                f"Alpha 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}],",
                html.Br(),
                f"p-value for (H0: Alpha = 0): {p_value:.4f} (Threshold c = 0.05, Null hypothesis rejected)"
            ]

        return fig, description, hypothesis

    # RIGHT
    # limit stocks
    @app.callback(
            Output("compare-stock-selector", "options"), # whether updated stocks should pass
            Input("compare-stock-selector", "value") # updated stocks
    )
    def limit_stocks(stock_list):
        if not stock_list:   
            return [stock.upper() for stock in all_symbols]
        if isinstance(stock_list, str):
            stock_list = [stock_list]
        
        if len(stock_list) >= 5:
            return stock_list # limiting available options to current selection once value len reaches 5
        else:
            return [stock.upper() for stock in all_symbols] # else all options are available
    # date-range-picker-2 values
    @app.callback(
        Output("date-range-picker-2", "start_date"),
        Output("date-range-picker-2", "end_date"),
        Output("date-range-picker-2", "min_date_allowed"),
        Output("date-range-picker-2", "max_date_allowed"),
        Input("compare-stock-selector", "value")
    )
    def date_range_picker_2_values(stock_list):
        if isinstance(stock_list, str):
            stock_list = [stock_list]
        if not stock_list:
            return None, None, None, None
        if not 1 < len(stock_list) < 6: # 2-5
            return None, None, None, None

        # start and end dates
        start = max(prepare_df(getpath(stock.casefold()))["original_date"].min() for stock in stock_list)
        end = min(prepare_df(getpath(stock.casefold()))["original_date"].max() for stock in stock_list)
        return start, end, start, end
    # storing dfs
    @app.callback(
        # storing
        Output("compare-df-store", "data"),
        # inputs
        Input("compare-stock-selector", "value"),
        Input("date-range-picker-2", "start_date"),
        Input("date-range-picker-2", "end_date"),
    )
    def store_dfs(stock_list, start, end):
        if isinstance(stock_list, str):
            stock_list = [stock_list]
        if not stock_list:
            return None
        if not 1 < len(stock_list) < 6: # 2-5
            return None
        
        try:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            df_dict = {
                stock.casefold() : prepare_df(getpath(stock.casefold())).loc[(lambda df: (df["original_date"] >= start) & (df["original_date"] <= end))].to_json(date_format="iso", orient="split")
                for stock in stock_list
            }
            return df_dict
        except Exception as e:
            print(e)
            return None

    # performance features
    # cum-return-graph
    @app.callback(
        Output("cum-return-graph", "figure"),
        Input("compare-df-store", "data")
    )
    def cum_return_graph(df_dict):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False, 
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            df["new_cum_return"] = (1 + df["return"]).cumprod()
            fig.add_scatter(x=df["original_date"], y=df["new_cum_return"], mode="lines", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    # rolling-return-chart
    @app.callback(
        Output("rolling-return-chart", "figure"),
        Input("compare-df-store", "data"),
        Input("cum-return-rolling-window-selector", "value")
    )
    def rolling_return_chart(df_dict, window):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict or not window or window not in windows:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            fig.add_scatter(x=df["original_date"], y=df[f"rolling_return_{window}"], mode="lines", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    # return-kde-histogram
    @app.callback(
        Output("return-kde-histogram", "figure"),
        Input("compare-df-store", "data")
    )
    def return_kde_histogram(df_dict):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )

        if not df_dict:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")

            # kde 
            returns = df["return"].dropna()
            kde = gaussian_kde(returns) # returns a function that reflects probability density function given data points
            kde_range = np.linspace(returns.min(), returns.max(), 200) # 200 equally spaced points from min-max
            kde_vals = kde(kde_range) # returns density at each of the 200 points (more points - smoother graph, more noise, potential overkill)

            fig.add_trace(go.Histogram(
                x=returns,
                histnorm="probability density",
                name=f"Stock {stock.upper()}",
                opacity=0.4,
                nbinsx=50,
                marker=dict(line=dict(width=0)),
                hovertemplate="%{y:.4f}"
            ))

            fig.add_trace(go.Scatter(
                x=kde_range, # points from min-max
                y=kde_vals, # density at those points
                mode="lines",
                name=f"KDE {stock.upper()}",
                hovertemplate="%{y:.4f}"
            ))
        return fig

    # risk/volatility features
    # rolling-volatility-chart
    @app.callback(
        Output("rolling-volatility-chart", "figure"),
        Input("compare-df-store", "data"),
        Input("risk-rolling-window-selector", "value")
    )
    def rolling_volatility_chart(df_dict, window):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict or not window or window not in windows:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            fig.add_scatter(x=df["original_date"], y=df[f"return_std_{window}"], mode="lines", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    # rolling-var-chart
    @app.callback(
        Output("rolling-var-chart", "figure"),
        Input("compare-df-store", "data"),
        Input("risk-rolling-window-selector", "value"),
        Input("var-confidence-level-selector", "value")
    )
    def rolling_var_chart(df_dict, window, ci):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict or not window or window not in windows or ci not in ["95%", "99%"]:
            return fig
        
        ci = int(ci[0:2])/100
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            fig.add_scatter(x=df["original_date"], y=df[f"var_{ci}_{window}"], mode="lines", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    # sharpe-ratio-chart
    @app.callback(
        Output("sharpe-ratio-chart", "figure"),
        Input("compare-df-store", "data"),
        Input("risk-rolling-window-selector", "value")
    )
    def sharpe_ratio_chart(df_dict, window):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict or not window or window not in windows:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            fig.add_scatter(x=df["original_date"], y=df[f"sharpe_{window}"], mode="lines", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    
    # volume and liquidity features
    # daily-volume-chart
    @app.callback(
        Output("daily-volume-chart", "figure"),
        Input("compare-df-store", "data")
    )
    def daily_volume_chart(df_dict):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            fig.add_scatter(x=df["original_date"], y=df["volume"], mode="lines", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    # volatility-volume-chart
    @app.callback(
        Output("volatility-volume-chart", "figure"),
        Input("compare-df-store", "data"),
        Input("volatility-volume-rolling-window-selector", "value")
    )
    def volatility_volume_chart(df_dict, window):
        # figure layout
        fig = go.Figure()
        fig.update_layout(
            hoverlabel=dict(
                font_size=10,
                font_color="black", 
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="rgba(255,255,255,0.2)",
            ),
            paper_bgcolor=contentColor,
            plot_bgcolor=contentColor,
            font_color=color,
            colorway=colorway,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_xaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        fig.update_yaxes(
            linecolor=color,
            gridcolor=color,
            griddash="dash"
        )
        if not df_dict or not window or window not in windows:
            return fig
        
        for (stock, json_df) in df_dict.items():
            df = pd.read_json(io.StringIO(json_df), orient="split")
            fig.add_scatter(x=df["volume"], y=df[f"return_std_{window}"], mode="markers", name=stock.upper(), hovertemplate="%{y:.4f}")
        return fig
    
    app.run()