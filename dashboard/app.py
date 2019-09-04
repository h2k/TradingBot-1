import json
import base64
import datetime
import requests
import pandas as pd
import flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)
app.title = "PoolTrade | Dashboard Admin"

# loading historical tick data
EURUSD = pd.read_csv("pairs/EURUSD.csv", index_col=1, parse_dates=["Date"])
USDJPY = pd.read_csv("pairs/USDJPY.csv", index_col=1, parse_dates=["Date"])
GBPUSD = pd.read_csv("pairs/GBPUSD.csv", index_col=1, parse_dates=["Date"])
USDCHF = pd.read_csv("pairs/USDCHF.csv", index_col=1, parse_dates=["Date"])

currencies = ["EURUSD", "USDCHF", "USDJPY", "GBPUSD"]  # list of currencies


def get_logo():
    """
    Load PoolTrade logo
    :return:
    """
    image = "assets/images/logotype-pooltrade.png"
    encoded_image = base64.b64encode(open(image, "rb").read())
    logo = html.Div(
        html.Img(src="data:image/png;base64,{}".format(encoded_image.decode()), height="57"),
        style={"marginTop": "0"},
        className="sept columns",
    )
    return logo


def generate_user_info_table(dataframe, max_rows=10):
    return html.Div(
        [
            html.Div(
                html.Table(
                    [html.Tr([html.Th()])]
                    +
                    [
                        html.Tr(
                            [
                                html.Td(
                                    html.A(
                                        dataframe.iloc[i]["title"],
                                        href=dataframe.iloc[i]["url"],
                                        target="_blank",
                                    )
                                )
                            ]
                        )
                        for i in range(min(len(dataframe), max_rows))
                    ]
                ),
                style={"height": "150px", "overflowY": "scroll"},
            ),
            html.P(
                id="user_info_update",
                style={"fontSize": "11", "marginTop": "4", "color": "#45df7e"},
            ),
        ],
        style={"height": "100%"},
    )


def generate_stock_list_table(dataframe, max_rows=4):
    return html.Div(
        [
            html.Div(
                html.Table(
                    [html.Tr([html.Th()])]
                    +
                    [
                        html.Tr([html.Td(dataframe.iloc[i]["symbol"])])
                        for i in range(min(len(dataframe), max_rows))
                    ]
                ),
                style={"height": "150px", "overflowY": "scroll"},
            ),
            html.P(
                id="stock_list_update",
                style={"fontSize": "11", "marginTop": "4", "color": "#45df7e"},
            ),
        ],
        style={"height": "100%"},
    )


def update_stock_list():
    """
    Retrieve and displays list of stock
    :return:
    """
    try:
        json_data = r.json()["stock"]
        df = pd.DataFrame(json_data).app
    except Exception:
        data = {"title": ["La liste des utilisateurs n'est pas récupérable"], "url": [url]}
        df = pd.DataFrame(data=data, columns=["symbol", ""])
    return generate_stock_list_table(df)


def update_user_info():
    """
    Retrieve and displays user's information
    :return:
    """
    url = 'https://dev-pooltradeapi.azurewebsites.net/general_balance'
    try:
        r = requests.get(url)
        json_data = r.json()["user"]
        df = pd.DataFrame(json_data)
    except Exception:
        data = {"title": ["La liste des utilisateurs n'est pas récupérable"], "url": [url]}
        df = pd.DataFrame(data=data, columns=["title", "url"])
    return generate_user_info_table(df)


# returns  table header with live clock
def get_stock_list_header():
    return html.Div(
        [
            html.P(
                "Symbole",
                className="four columns",
                style={"color": "white", "textAlign": "center"},
            ),
            html.P(
                "Prix",
                className="four columns",
                style={"color": "white", "textAlign": "center"},
            )
        ],
        style={"paddingTop": "15"},
    )


# returns  table header with live clock
def get_ask_bid_header():
    return html.Div(
        [
            html.P(
                "Marché",
                className="four columns",
                style={"color": "white", "textAlign": "center"},
            ),
            html.P(
                "Achat",
                className="four columns",
                style={"color": "white", "textAlign": "center"},
            ),
            html.P(
                "Vente",
                className="four columns",
                style={"color": "white", "textAlign": "center"},
            )
        ],
        style={"paddingTop": "15"},
    )


def get_color(a, b):
    """
    Color of Bid & Ask rates
    White, red or green
    :param a: current price
    :param b: Previous price
    :return:
    """
    if a == b:
        return "white"
    elif a > b:
        return "#45df7e"
    else:
        return "#da5657"


def get_ask_bid(currency_pair, index, modal=False):
    if not modal:
        return globals()[currency_pair].ix[index]
    else:
        return globals()[currency_pair].ix[index - 10: index]


def first_ask_bid(currency_pair, t):
    """
    Get value with nearest datetime to current time
    :param currency_pair:
    :param t:
    :return:
    """
    t = t.replace(year=2016, month=1, day=5)
    items = globals()[currency_pair]
    dates = items.index.to_pydatetime()
    index = min(dates, key=lambda x: abs(x - t))
    df_row = items.loc[index]
    int_index = items.index.get_loc(index)
    return [df_row, int_index]


def get_row(data):
    """
    Creates a Bid & Ask row for a currency pair + buttons
    :param data:
    :return:
    """
    index = data[1]
    current_row = data[0]
    return html.Details(
        [
            html.Summary(
                [

                    html.Div(
                        [
                            html.P(
                                current_row[0],  # currency pair name
                                id=current_row[0],
                                className="four columns",
                                style={"textAlign": "center"},
                            ),
                            html.P(
                                current_row[1].round(5),  # Bid value
                                id=current_row[0] + "bid",
                                className="four columns",
                                style={"textAlign": "center", "color": "white"},
                            ),
                            html.P(
                                current_row[2].round(5),  # Ask value
                                className="four columns",
                                id=current_row[0] + "ask",
                                style={"textAlign": "center", "color": "white"},
                            ),
                            html.Div(
                                index,
                                id=current_row[0] + "index",
                                style={"display": "none"},
                            ),
                        ],
                        id=current_row[0] + "row",
                        className="row eleven columns",
                        style={"height": "25", "float": "right"},
                    ),
                ],
                className="row",
                style={"paddingLeft": "10"}
            ),
            html.Button(
                "Chart",
                id=current_row[0] + "Button_chart",
                n_clicks=1 if current_row[0] in ["EURUSD", "USDCHF"] else 0,
                style={"margin": "0px 7px 7px 10px", "textAlign": "center"},
            ),
        ],
        id=current_row[0] + "row_div",
        n_clicks=0,
        style={"textAlign": "center", "paddingTop": "4"},
    )


def replace_row(currency_pair, index, bid, ask):
    index = index + 1
    new_row = (
        get_ask_bid(currency_pair, index)
        if index != len(globals()[currency_pair])
        else first_ask_bid(currency_pair, datetime.datetime.now())
    )

    return [
        html.P(
            currency_pair,  # currency pair name
            id=currency_pair,
            className="four columns",
            style={"textAlign": "center"},
        ),
        html.P(
            new_row[1].round(5),  # Bid value
            id=new_row[0] + "bid",
            className="four columns",
            style={"textAlign": "center", "color": get_color(new_row[1], bid)},
        ),
        html.P(
            new_row[1].round(5),  # Ask value
            className="four columns",
            id=new_row[0] + "ask",
            style={"textAlign": "center", "color": get_color(new_row[2], ask)},
        ),
        html.Div(index, id=currency_pair + "index", style={"display": "none"}),
    ]


def get_first_pairs(t):
    """
    Displays left panel rows with chart
    :param t:
    :return:
    """
    return [get_row(first_ask_bid(pair, t)) for pair in currencies]


def get_top_bar_cell(up, down, color="white"):
    """
    Returns one cell of top bar
    :param up:
    :param down:
    :param color:
    :return:
    """
    return html.Div(
        [
            html.P(up, style={"marginBottom": "1px", "color": "#afdee4"}),
            html.P(down, id=up, style={"marginBottom": "3"}),
        ],
        className="one columns",
        style={"width": "15%", "fontSize": "12", "color": color},
    )


def get_top_bar(user="TradingBot", balance=50000.00, equity=50000, margin=0, fm=50000, m_level="%", open_pl=0):
    """
    Returns top_bar with updated values
    :param user:
    :param balance:
    :param equity:
    :param margin:
    :param fm:
    :param m_level:
    :param open_pl:
    :return:
    """
    color_open_pl = get_color(float(open_pl), 0)
    return [
        get_top_bar_cell("Utilisateur", user),
        get_top_bar_cell("Balance", balance),
        get_top_bar_cell("Fonds disponible", equity),
        get_top_bar_cell("Marge", margin),
        get_top_bar_cell("Taux de risque/marge", m_level),
        get_top_bar_cell("Positionnement P/L", open_pl, color=color_open_pl),
    ]


def get_OHLC_data(currency_pair, period="5Min"):
    """
    Get OHLC price (Open High Low Close)
    :param currency_pair:
    :param period:
    :return:
    """
    data_frame = globals()[currency_pair]
    t = datetime.datetime.now()
    data = data_frame.loc[:t.strftime("2016-01-05 %H:%M:%S")]
    data_bid = data["Bid"]
    data_bid = data_bid.resample(period).ohlc()
    return data_bid


def moving_average_trace(df, fig):
    """
    Calcul Moving average (Moyenne mobile)
    :param df:
    :param fig:
    :return:
    """
    df2 = df.rolling(window=5).mean()
    trace = go.Scatter(x=df2.index, y=df2["close"], mode="lines", showlegend=False, name="MA")
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


def e_moving_average_trace(df, fig):
    """
    Calcul Exponential moving average (eMACD)
    :param df:
    :param fig:
    :return:
    """
    df2 = df.rolling(window=20).mean()
    trace = go.Scatter(x=df2.index, y=df2["close"], mode="lines", showlegend=False, name="EMA")
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


def bollinger_trace(df, fig, window_size=10, num_of_std=5):
    """
    Calcul BBbands
    :param df:
    :param fig:
    :param window_size:
    :param num_of_std:
    :return:
    """
    price = df["close"]
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    trace = go.Scatter(x=df.index, y=upper_band, mode="lines", showlegend=False, name="Bande supérieure")
    trace2 = go.Scatter(x=df.index, y=rolling_mean, mode="lines", showlegend=False, name="Moyenne mobile")
    trace3 = go.Scatter(x=df.index, y=lower_band, mode="lines", showlegend=False, name="Bande inférieure")
    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    return fig


def accumulation_trace(df):
    """
    Calcul accumulation distribué
    :param df:
    :return:
    """
    df["volume"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    trace = go.Scatter(x=df.index, y=df["volume"], mode="lines", showlegend=False, name="Accumulation")
    return trace


def cci_trace(df, ndays=5):
    """
    Calcul CCI Commodity Channel Index
    :param df:
    :param ndays:
    :return:
    """
    TP = (df["high"] + df["low"] + df["close"]) / 3
    CCI = pd.Series(
        (TP - TP.rolling(window=10, center=False).mean())
        / (0.015 * TP.rolling(window=10, center=False).std()),
        name="cci",
    )
    trace = go.Scatter(x=df.index, y=CCI, mode="lines", showlegend=False, name="CCI")
    return trace


def roc_trace(df, ndays=5):
    """
    Calcul prix taux de change
    :param df:
    :param ndays:
    :return:
    """
    N = df["close"].diff(ndays)
    D = df["close"].shift(ndays)
    ROC = pd.Series(N / D, name="roc")
    trace = go.Scatter(x=df.index, y=ROC, mode="lines", showlegend=False, name="ROC")
    return trace


def stoc_trace(df):
    """
    Calcul stochastique
    :param df:
    :return:
    """
    sok = pd.Series((df["close"] - df["low"]) / (df["high"] - df["low"]), name="SO%k")
    trace = go.Scatter(x=df.index, y=sok, mode="lines", showlegend=False, name="SO%k")
    return trace


def mom_trace(df, n=5):
    """
    Calcul du momemtum
    Voir https://www.datacamp.com/community/tutorials/finance-python-trading
    :param df:
    :param n:
    :return:
    """
    M = pd.Series(df["close"].diff(n), name="Momentum_" + str(n))
    trace = go.Scatter(x=df.index, y=M, mode="lines", showlegend=False, name="MOM")
    return trace


def pp_trace(df, fig):
    """
    Calcul point de pivots
    Voir https://github.com/nhsb1/pivot-point-calculator
    :param df:
    :param fig:
    :return:
    """
    PP = pd.Series((df["high"] + df["low"] + df["close"]) / 3)
    R1 = pd.Series(2 * PP - df["low"])
    S1 = pd.Series(2 * PP - df["high"])
    R2 = pd.Series(PP + df["high"] - df["low"])
    S2 = pd.Series(PP - df["high"] + df["low"])
    R3 = pd.Series(df["high"] + 2 * (PP - df["low"]))
    S3 = pd.Series(df["low"] - 2 * (df["high"] - PP))
    trace = go.Scatter(x=df.index, y=PP, mode="lines", showlegend=False, name="PP")
    trace1 = go.Scatter(x=df.index, y=R1, mode="lines", showlegend=False, name="R1")
    trace2 = go.Scatter(x=df.index, y=S1, mode="lines", showlegend=False, name="S1")
    trace3 = go.Scatter(x=df.index, y=R2, mode="lines", showlegend=False, name="R2")
    trace4 = go.Scatter(x=df.index, y=S2, mode="lines", showlegend=False, name="S2")
    trace5 = go.Scatter(x=df.index, y=R3, mode="lines", showlegend=False, name="R3")
    trace6 = go.Scatter(x=df.index, y=S3, mode="lines", showlegend=False, name="S3")
    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 1)
    fig.append_trace(trace5, 1, 1)
    fig.append_trace(trace6, 1, 1)
    return fig


def line_trace(df):
    trace = go.Scatter(x=df.index, y=df["close"], mode="lines", showlegend=False, name="Ligne")
    return trace


def area_trace(df):
    trace = go.Scatter(x=df.index, y=df["close"], showlegend=False, fill="toself", name="Zone")
    return trace


def bar_trace(df):
    """
    Tracé barres colorées
    :param df:
    :return:
    """
    return go.Ohlc(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color="#888888")),
        decreasing=dict(line=dict(color="#888888")),
        showlegend=False,
        name="Barre",
    )


def colored_bar_trace(df):
    """
    Tracé chandelles low/open/high/close collorées
    :param df:
    :return:
    """
    return go.Ohlc(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        showlegend=False,
        name="Barre",
    )


def candlestick_trace(df):
    """
    Tracé chandelles low/open/high/close
    :param df:
    :return:
    """
    return go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color="#00ff00")),
        decreasing=dict(line=dict(color="white")),
        showlegend=False,
        name="Chandelle",
    )


def get_fig(currency_pair, type_trace, studies, period):
    """
    Creates graph figure
    :param currency_pair:
    :param type_trace:
    :param studies:
    :param period:
    :return:
    """
    df = get_OHLC_data(currency_pair, period)
    subplot_traces = [
        "accumulation_trace",
        "cci_trace",
        "roc_trace",
        "stoc_trace",
        "mom_trace",
    ]
    selected_subplots_studies = []
    selected_first_row_studies = []
    row = 1
    if studies:
        for study in studies:
            if study in subplot_traces:
                row += 1
                selected_subplots_studies.append(study)
            else:
                selected_first_row_studies.append(study)
    fig = tools.make_subplots(
        rows=row,
        shared_xaxes=True,
        shared_yaxes=True,
        cols=1,
        print_grid=False,
        vertical_spacing=0.12,
    )
    fig.append_trace(globals()[type_trace](df), 1, 1)
    for study in selected_first_row_studies:
        fig = globals()[study](df, fig)
    row = 1
    for study in selected_subplots_studies:
        row += 1
        fig.append_trace(globals()[study](df), row, 1)
    fig["layout"]["margin"] = {"b": 50, "r": 5, "l": 50, "t": 5}
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["xaxis"]["tickformat"] = "%H:%M"
    fig["layout"].update(paper_bgcolor="#18252E", plot_bgcolor="#18252E")
    return fig


def replace_fig(currency_pair, type_trace, period, old_fig, studies):
    """
    Updates figure
    :param currency_pair:
    :param type_trace:
    :param period:
    :param old_fig:
    :param studies:
    :return:
    """
    fig = get_fig(currency_pair, type_trace, studies, period)
    fig["layout"]["xaxis"]["range"] = old_fig["layout"]["xaxis"]["range"]
    return fig


def chart_div(pair):
    """
    Creates the chart div container
    :param pair:
    :return:
    """
    return html.Div(
        id=pair + "graph_div",
        children=[
            html.Span(
                "×",
                id=pair + "close",
                n_clicks=0,
                style={
                    "fontSize": "16",
                    "float": "right",
                    "paddingRight": "5",
                    "verticalAlign": "textTop",
                    "cursor": "pointer",
                },
                className="row",
            ),
            html.Div(
                children=[
                    html.Div(
                        "Indicateurs", id=pair + "menu_tab", style={"display": "none"}
                    ),
                    html.Span(
                        "×",
                        id=pair + "close_menu",
                        n_clicks=0,
                        style={
                            "fontSize": "16",
                            "float": "right",
                            "paddingRight": "5",
                            "verticalAlign": "textTop",
                            "cursor": "pointer",
                        },
                    ),
                    html.Span(
                        "Style",
                        id=pair + "style_header",
                        n_clicks_timestamp=2,
                        style={
                            "top": "0",
                            "float": "left",
                            "marginLeft": "5",
                            "marginRight": "5",
                            "textDecoration": "none",
                            "cursor": "pointer",
                        },
                    ),
                    html.Span(
                        "Indicateurs",
                        id=pair + "studies_header",
                        n_clicks_timestamp=1,
                        style={
                            "float": "left",
                            "textDecoration": "none",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(
                        html.Div(
                            dcc.Checklist(
                                id=pair + "studies",
                                options=[
                                    {"label": "Accumulation/D", "value": "accumulation_trace"},
                                    {"label": "BBbands", "value": "bollinger_trace"},
                                    {"label": "MA", "value": "moving_average_trace"},
                                    {"label": "EMA", "value": "e_moving_average_trace"},
                                    {"label": "CCI", "value": "cci_trace"},
                                    {"label": "ROC", "value": "roc_trace"},
                                    {"label": "Points pivot", "value": "pp_trace"},
                                    {"label": "Stochastique", "value": "stoc_trace"},
                                    {"label": "Momentum", "value": "mom_trace"},
                                ],
                                values=[],
                            ),
                            style={"marginTop": "30", "textAlign": "left"},
                        ),
                        id=pair + "studies_tab",
                        style={"display": "none"},
                    ),
                    html.Div(
                        dcc.RadioItems(
                            id=pair + "chart_type",
                            options=[
                                {"label": "Chandelles", "value": "candlestick_trace"},
                                {"label": "Ligne", "value": "line_trace"},
                                {"label": "Zone", "value": "area_trace"},
                                {"label": "Barres", "value": "bar_trace"},
                                {"label": "Barres colorées", "value": "colored_bar_trace"},
                            ],
                            value="colored_bar_trace",
                        ),
                        id=pair + "style_tab",
                        style={"marginTop": "30", "textAlign": "left"},
                    ),
                ],
                id=pair + "menu",
                className="not_visible ",
                style={
                    "overflow": "auto",
                    "borderRight": "1px solid" + "rgba(68,149,209,.9)",
                    "backgroundImage": "-webkit-linear-gradient(top,#18252e,#2a516e 63%)",
                    "zIndex": "20",
                    "width": "45%",
                    "height": "100%",
                    "position": "absolute",
                    "left": "0",
                    "top": "27px",
                },
            ),
            html.Div(
                [
                    html.Span(
                        pair,
                        style={"float": "left", "marginRight": "5", "color": "white"},
                    ),
                    html.Span(
                        "☰",
                        n_clicks=0,
                        id=pair + "menu_button",
                        style={"float": "left", "color": "white", "cursor": "pointer"},
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id=pair + "dropdown_period",
                            className="period",
                            options=[
                                {"label": "5m", "value": "5Min"},
                                {"label": "15m", "value": "15Min"},
                                {"label": "30m", "value": "30Min"},
                                {"label": "60m", "value": "60Min"},
                                {"label": "1J", "value": "1Day"},
                            ],
                            value="15Min",
                            clearable=False,
                        ),
                        style={
                            "float": "left",
                            "width": "45px",
                            "padding": "0",
                            "marginLeft": "3",
                            "fontSize": "5",
                        },
                    ),
                ],
                className="row",
                style={
                    "padding": "3",
                    "height": "20px",
                    "margin": "0 0 5 0",
                    "backgroundImage": "-webkit-linear-gradient(top,#18252e,#2a516e 63%)",
                },
            ),
            html.Div(
                dcc.Graph(
                    id=pair + "chart",
                    config={"displayModeBar": False, "scrollZoom": True},
                    style={"width": "100%", "height": "100%"},
                ),
                id=pair + "graph",
                style={"height": "100%"}
            ),

        ],
        className="",
        style={"display": "none"},
    )


def bottom_panel():
    """
    Bottom panel for orders
    :return:
    """
    return html.Div(
        children=[
            dcc.Location(id="bottom_tab", refresh=False),
            dcc.Link("Position(s) ouverte(s)", id="open_positions", href="/"),
            dcc.Link("Position(s) cloturée(s)", id="closed_positions", href="/closed"),
            html.Div(
                dcc.Dropdown(id="closable_orders", placeholder="Fermer l'ordre"),
                style={"width": "15%"},
                id="close_orders_div",
            ),
            html.Div(
                children=[html.Table(id="orders_table")],
                className="row",
                style={"padding": "3", "textAlign": "center"},
                id="bottom_content",
            ),
        ],
        id="bottom_panel",
        className="row",
        style={
            "overflowY": "auto",
            "margin": "9px 5px 0px 5px",
            "padding": "5",
            "height": "21%",
            "backgroundColor": "#1a2d46",
        },
    )


def orders_div():
    return [html.Div(id=pair + "orders", style={"display": "none"}) for pair in currencies]


def orders_rows(list_order, status):
    """
    Returns orders table tab content
    Status is open or closed
    :param list_order:
    :param status:
    :return:
    """
    headers = [
        "Id de l'ordre",
        "Heure",
        "Type",
        "Volume",
        "Symbole",
        "Take Profit (TP)",
        "Stop Loss (SL)",
        "Prix",
        "Profit",
        "Statut",
    ]
    if status == "closed":
        headers += ["Heure clotûtre", "Prix clotûre"]
    if list_order is not None:
        rows = []
        for order in list_order:
            tr_childs = []
            for attr in order:
                if order["status"] == status:
                    tr_childs.append(html.Td(order[attr]))
            order_style = {
                "background": "linear-gradient(to bottom, rgba(255,0,0,0), rgba(255,0,0,1))"
                if float(order["profit"]) < 0
                else "linear-gradient(to bottom, rgba(0,255,0,0), rgba(0,255,0,1))"
            }
            rows.append(html.Tr(tr_childs, style=order_style))
        table = [html.Tr([html.Th(title) for title in headers])] + rows
        return table
    return [html.Tr([html.Th(title) for title in headers], style={"width": "100%"})]


app.layout = html.Div(
    [
        # Interval component for ask bid updates
        dcc.Interval(id="i_bis", interval=1 * 2000, n_intervals=0),
        # Interval component for graph updates
        dcc.Interval(id="i_tris", interval=1 * 5000, n_intervals=0),
        # Interval component for graph updates
        dcc.Interval(id="i_user_info", interval=1 * 60000, n_intervals=0),
        html.Div(
            [
                get_logo(),
                html.Div(
                    children=[get_ask_bid_header()],
                    style={"backgroundColor": "#18252E", "paddingTop": "15"},
                    id="ask_bid_header",
                    className="row",
                ),
                html.Div(
                    get_first_pairs(datetime.datetime.now()),
                    style={
                        "maxHeight": "45%",
                        "backgroundColor": "#18252E",
                        "color": "white",
                        "fontSize": "12",
                        "paddingBottom": "15"
                    },
                    className="",
                    id="pairs",
                ),
                html.Div(
                    children=[get_stock_list_header()],
                    style={"backgroundColor": "#18252E", "paddingTop": "15"},
                    id="stock_list_header",
                    className="row",
                ),
                html.Div(
                    html.P("A venir"),
                    style={
                        "maxHeight": "45%",
                        "backgroundColor": "#18252E",
                        "color": "white",
                        "fontSize": "12",
                        "paddingBottom": "15"
                    },
                    className="",
                    id="stocks",
                ),
                html.Div([
                    html.P('Liste des utilisateurs', style={"fontSize": "13", "color": "#45df7e"}),
                    html.Div(id="user_info")
                ],
                    style={
                        "height": "33%",
                        "backgroundColor": "#18252E",
                        "color": "white",
                        "fontSize": "12",
                        "padding": "10px 10px 0px 10px",
                        "marginTop": "5",
                        "marginBottom": "0"
                    }),
            ],
            className="three columns",
            style={
                "backgroundColor": "#1a2d46",
                "padding": "10",
                "margin": "0",
                "height": "100%"
            },
        ),
        html.Div(
            [
                html.Div(
                    get_top_bar(),
                    id="top_bar",
                    className="row",
                    style={
                        "margin": "0px 5px 0px 5px",
                        "textAlign": "center",
                        "height": "6%",
                        "color": "white",
                        "backgroundColor": "#1a2d46",
                    },
                ),
                html.Div(
                    [chart_div(pair) for pair in currencies],
                    style={"height": "70%", "margin": "0px 5px"},
                    id="charts",
                    className="row",
                ),
                bottom_panel(),
            ],
            className="nine columns",
            id="rightpanel",
            style={
                "backgroundColor": "#18252E",
                "height": "100vh",
                "color": "white",
            },
        ),
        html.Div(id="charts_clicked", style={"display": "none"}),
        html.Div(orders_div()),
        html.Div(id="orders", style={"display": "none"}),
    ],
    style={"padding": "0", "height": "100vh", "backgroundColor": "#1a2d46"},
)


# replace pair row
def generate_ask_bid_row_callback(pair):
    def output_callback(n, i, bid, ask):
        return replace_row(pair, int(i), float(bid), float(ask))
    return output_callback


# returns string containing clicked charts
def generate_chart_button_callback():
    def chart_button_callback(*args):
        pairs = ""
        for i in range(len(currencies)):
            if args[i] > 0:
                pair = currencies[i]
                if pairs:
                    pairs = pairs + "," + pair
                else:
                    pairs = pair
        return pairs
    return chart_button_callback


# updates graph figure
def generate_figure_callback(pair):
    def chart_fig_callback(n_i, p, t, s, pairs, a, b, old_fig):
        if pairs is None:
            return {"layout": {}, "data": {}}
        pairs = pairs.split(",")  # list of displayed divs
        if pair not in pairs:
            return {
                "layout": {},
                "data": {},
            }
        if old_fig is None or old_fig == {"layout": {}, "data": {}}:
          return get_fig(pair, t, s, p)
        return replace_fig(pair, t, p, old_fig, s)
    return chart_fig_callback


def generate_close_graph_callback():
    def close_callback(n, n2):
        if n == 0:
            if n2 == 1:
                return 1
            return 0
        return 0
    return close_callback


def generate_open_close_menu_callback():
    def open_close_menu(n, n2, className):
        if n == 0:
            return "not_visible"
        if className == "visible":
            return "not_visible"
        else:
            return "visible"
    return open_close_menu


# updates hidden div that stores the last clicked menu tab
def generate_active_menu_tab_callback():
    def update_current_tab_name(n_style, n_studies):
        if n_style >= n_studies:
            return "Style"
        return "Studies"
    return update_current_tab_name


# show/hide 'studies' menu content
def generate_studies_content_tab_callback():
    def studies_tab(current_tab):
        if current_tab == "Studies":
            return {"display": "block", "textAlign": "left", "marginTop": "30"}
        return {"display": "none"}
    return studies_tab


# show/hide 'style' menu content
def generate_style_content_tab_callback():
    def style_tab(current_tab):
        if current_tab == "Style":
            return {"display": "block", "textAlign": "left", "marginTop": "30"}
        return {"display": "none"}
    return style_tab


# updates style of header 'studies' in menu
def generate_update_studies_header_callback():
    def studies_header(current_tab, old_style):
        if current_tab == "Studies":
            old_style["borderBottom"] = "2px solid" + " " + "#45df7e"
        else:
            old_style["borderBottom"] = "2px solid" + " " + "rgba(68,149,209,.9)"
        return old_style
    return studies_header


# updates style of header 'style' in menu
def generate_update_style_header_callback():
    def style_header(current_tab, old_style):
        if current_tab == "Style":
            old_style["borderBottom"] = "2px solid" + " " + "#45df7e"
        else:
            old_style["borderBottom"] = "2px solid" + " " + "rgba(68,149,209,.9)"
        return old_style
    return style_header


def generate_modal_open_callback():
    def open_modal(n):
        if n > 0:
            return {"display": "block"}
        else:
            return {"display": "none"}
    return open_modal


def generate_modal_close_callback():
    def close_modal(n, n2):
        return 0
    return close_modal


# set modal SL value to none
def generate_clean_sl_callback():
    def clean_sl(n):
        return 0
    return clean_sl


# set modal SL value to none
def generate_clean_tp_callback():
    def clean_tp(n):
        return 0
    return clean_tp


def generate_modal_figure_callback(pair):
    def figure_modal(index, n, old_fig):
        if (n == 0 and old_fig is None) or n == 1:
            return get_modal_fig(pair, index)
        return old_fig  # avoid to compute new figure when the modal is hidden
    return figure_modal


# updates the pair orders div
def generate_order_button_callback(pair):
    def order_callback(n, vol, type_order, sl, tp, pair_orders, ask, bid):
        if n > 0:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            l = [] if pair_orders is None else json.loads(pair_orders)
            price = bid if type_order == "sell" else ask
            if tp != 0:
                tp = (
                    price + tp * 0.001
                    if tp != 0 and pair[3:] == "JPY"
                    else price + tp * 0.00001
                )
            if sl != 0:
                sl = price - sl * 0.001 if pair[3:] == "JPY" else price + sl * 0.00001
            order = {
                "id": pair + str(len(l)),
                "time": t,
                "type": type_order,
                "volume": vol,
                "symbol": pair,
                "tp": tp,
                "sl": sl,
                "price": price,
                "profit": 0.00,
                "status": "open",
            }
            l.append(order)
            return json.dumps(l)
        return json.dumps([])
    return order_callback


def update_orders(orders, current_bids, current_asks, id_to_close):
    for order in orders:
        if order["status"] == "open":
            type_order = order["type"]
            current_bid = current_bids[currencies.index(order["symbol"])]
            current_ask = current_asks[currencies.index(order["symbol"])]
            profit = (
                order["volume"]
                * 100000
                * ((current_bid - order["price"]) / order["price"])
                if type_order == "buy"
                else (order["volume"] * 100000 * ((order["price"] - current_ask) / order["price"]))
            )
            order["profit"] = "%.2f" % profit
            price = current_bid if order["type"] == "buy" else current_ask
            if (order["id"] == id_to_close) or (order["tp"] != 0 and price >= order["tp"]) or\
                    (order["sl"] != 0 and order["sl"] >= price):
                order["status"] = "closed"
                order["close Time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                order["close Price"] = price
    return orders


def generate_update_orders_div_callback():
    def update_orders_callback(*args):
        orders = []
        current_orders = args[-1]
        if not current_orders:
            current_orders = '[{"id": "USDCHF0", "time": "2019-07-03 21:18:14", "type": "buy", "volume": 10, "symbol": "USDCHF", "tp": 0, "sl": 0, "price": 1.00879, "profit": "-158.61", "status": "open"}]'
        close_id = args[-2]
        args = args[:-2]  # contains list of orders for each pair + asks + bids
        len_args = len(args)
        current_bids = args[len_args // 3: 2 * len_args]
        current_asks = args[2 * len_args // 3: len_args]
        args = args[: len_args // 3]
        ids = []
        if current_orders is not None:
            orders = json.loads(current_orders)
            for order in orders:
                ids.append(order["id"])
        for list_order in args:  # each currency pair has its list of orders
            if list_order and list_order != "[]":
                list_order = json.loads(list_order)
                for order in list_order:
                    if order["id"] not in ids:  # only add new orders
                        orders.append(order)
        if len(orders) == 0:
            return None
        # we update status and profit of orders
        orders = update_orders(orders, current_bids, current_asks, close_id)
        return json.dumps(orders)
    return update_orders_callback


def generate_show_hide_graph_div_callback(pair):
    def show_hide_graph_callback(charts_clicked):
        if charts_clicked is None:
            return {"display": "none"}
        charts_clicked = charts_clicked.split(",")[:4]  # TODO : Enable 8 stocks at a time
        len_list = len(charts_clicked)
        for i in range(len_list):
            if charts_clicked[i] == pair:
                style = {
                    "position": "relative",
                    "float": "left",
                    "border": "1px solid",
                    "borderColor": "rgba(68,149,209,.9)",
                    "overflow": "hidden",
                    "marginBottom": "2px",
                }
                if i == 0 or (i == 2 and len_list == 4):
                    style["marginLeft"] = "0px"
                style["height"] = "50%" if len_list == 4 else "100%"
                return style
        return {"display": "none"}
    return show_hide_graph_callback


# Resize pair div according to the number of charts displayed
def generate_size_graph_div_callback(pair):
    def size_graph_div_callback(charts_clicked):
        if charts_clicked is None:
            return ""
        charts_clicked = charts_clicked.split(",")  # [:4] max of 4 graph
        len_list = len(charts_clicked)
        if pair not in charts_clicked:
            return ""
        width = ("six columns" if len_list % 2 == 0 else "twelve columns" if len_list == 1 else "four columns")
        return width
    return size_graph_div_callback


for pair in currencies:
    app.callback(
        Output(pair + "graph_div", "style"), [Input("charts_clicked", "children")]
    )(generate_show_hide_graph_div_callback(pair))

    app.callback(
        Output(pair + "graph_div", "className"), [Input("charts_clicked", "children")]
    )(generate_size_graph_div_callback(pair))

    # udaptes graph's figure
    app.callback(
        Output(pair + "chart", "figure"),
        [
            Input("i_tris", "n_intervals"),
            Input(pair + "dropdown_period", "value"),
            Input(pair + "chart_type", "value"),
            Input(pair + "studies", "values"),
            Input("charts_clicked", "children"),
        ],
        [
            State(pair + "ask", "children"),
            State(pair + "bid", "children"),
            State(pair + "chart", "figure"),
        ],
    )(generate_figure_callback(pair))

    # updates the ask and bid prices
    app.callback(
        Output(pair + "row", "children"),
        [Input("i_bis", "n_intervals")],
        [
            State(pair + "index", "children"),
            State(pair + "bid", "children"),
            State(pair + "ask", "children"),
        ],
    )(generate_ask_bid_row_callback(pair))

    # close graph by setting to 0 n_clicks property
    app.callback(
        Output(pair + "Button_chart", "n_clicks"),
        [Input(pair + "close", "n_clicks")],
        [State(pair + "Button_chart", "n_clicks")],
    )(generate_close_graph_callback())

    # show or hide graph menu
    app.callback(
        Output(pair + "menu", "className"),
        [
            Input(pair + "menu_button", "n_clicks"),
            Input(pair + "close_menu", "n_clicks"),
        ],
        [State(pair + "menu", "className")],
    )(generate_open_close_menu_callback())

    # stores in hidden div name of clicked tab name
    app.callback(
        Output(pair + "menu_tab", "children"),
        [
            Input(pair + "style_header", "n_clicks_timestamp"),
            Input(pair + "studies_header", "n_clicks_timestamp"),
        ],
    )(generate_active_menu_tab_callback())

    # hide/show menu tab content if clicked or not
    app.callback(
        Output(pair + "style_tab", "style"), [Input(pair + "menu_tab", "children")]
    )(generate_style_content_tab_callback())

    # hide/show menu tab content if clicked or not
    app.callback(
        Output(pair + "studies_tab", "style"), [Input(pair + "menu_tab", "children")]
    )(generate_studies_content_tab_callback())

    # styles menu tab depending on if clicked or not
    app.callback(
        Output(pair + "style_header", "style"),
        [Input(pair + "menu_tab", "children")],
        [State(pair + "style_header", "style")],
    )(generate_update_style_header_callback())

    # styles menu tab depending on if clicked or not
    app.callback(
        Output(pair + "studies_header", "style"),
        [Input(pair + "menu_tab", "children")],
        [State(pair + "studies_header", "style")],
    )(generate_update_studies_header_callback())

    # each pair saves its orders in hidden div
    app.callback(
        Output(pair + "orders", "children"),
        [Input(pair + "button_order", "n_clicks")],
        [
            State(pair + "volume", "value"),
            State(pair + "trade_type", "value"),
            State(pair + "SL", "value"),
            State(pair + "TP", "value"),
            State(pair + "orders", "children"),
            State(pair + "ask", "children"),
            State(pair + "bid", "children"),
        ],
    )(generate_order_button_callback(pair))

# Updates hidden div with all the clicked charts
app.callback(
    Output("charts_clicked", "children"),
    [Input(pair + "Button_chart", "n_clicks") for pair in currencies],
    [State("charts_clicked", "children")],
)(generate_chart_button_callback())

# Updates hidden orders div with all pairs orders
app.callback(
    Output("orders", "children"),
    [Input(pair + "orders", "children") for pair in currencies]
    + [Input(pair + "bid", "children") for pair in currencies]
    + [Input(pair + "ask", "children") for pair in currencies]
    + [Input("closable_orders", "value")],
    [State("orders", "children")],
)(generate_update_orders_div_callback())


@app.callback(Output("orders_table", "children"), [Input("orders", "children"), Input("bottom_tab", "pathname")])
def update_order_table(orders, url):
    url = "open" if url == "/" else "closed"
    if orders is None or orders is "[]":
        return orders_rows(None, url)
    return orders_rows(json.loads(orders), url)


@app.callback(Output("open_positions", "style"), [Input("bottom_tab", "pathname")])
def update_link_style_open(url):
    style = (
        {
            "borderBottom": "2px solid" + " " + "#45df7e",
            "textDecoration": "none",
            "color": "white",
        }
        if url == "/"
        else {
            "borderBottom": "2px solid" + " " + "rgba(68,149,209,.9)",
            "textDecoration": "none",
            "color": "white",
        }
    )
    return style


@app.callback(Output("open_positions", "children"), [Input("bottom_tab", "pathname"), Input("orders", "children")])
def update_link_label_open(url, orders):
    length = 0
    if orders is not None:
        orders = json.loads(orders)
        for order in orders:
            if order["status"] == "open":
                length += 1
    return "Position(s) ouverte(s) (" + str(length) + ")"


@app.callback(Output("closed_positions", "style"), [Input("bottom_tab", "pathname")])
def update_link_style_closed(url):
    style = (
        {
            "borderBottom": "2px solid" + " " + "#45df7e",
            "textDecoration": "none",
            "marginLeft": "10",
            "color": "white",
        }
        if url == "/closed"
        else {
            "borderBottom": "2px solid" + " " + "rgba(68,149,209,.9)",
            "textDecoration": "none",
            "marginLeft": "10",
            "color": "white",
        }
    )
    return style


@app.callback(Output("closed_positions", "children"), [Input("bottom_tab", "pathname"), Input("orders", "children")])
def update_link_label_closed(url, orders):
    length = 0
    if orders is not None:
        orders = json.loads(orders)
        for order in orders:
            if order["status"] == "closed":
                length += 1
    return "Position(s) cloturée(s) (" + str(length) + ")"


# hide/show div that contains 'close order' dropdown
@app.callback(Output("close_orders_div", "style"), [Input("bottom_tab", "pathname")])
def show_hide_close_orders(url):
    style = (
        {
            "float": "right",
            "right": "5",
            "top": "3",
            "width": "15%",
            "textAlign": "center",
        }
        if url == "/"
        else {"display": "none"}
    )
    return style


# updates close orders dropdown options
@app.callback(Output("closable_orders", "options"), [Input("orders", "children")])
def update_close_dropdown(orders):
    options = []
    if orders is not None:
        orders = json.loads(orders)
        for order in orders:
            if order["status"] == "open":
                options.append({"label": order["id"], "value": order["id"]})
    return options


@app.callback(Output("top_bar", "children"), [Input("orders", "children")])
def update_top_bar(orders):
    if orders is None or orders is "[]":
        return get_top_bar()
    orders = json.loads(orders)
    user = "John Doe"
    open_pl = 0
    balance = 50000
    margin = 0
    for order in orders:
        if order["status"] == "open":
            open_pl += float(order["profit"])
            conversion_price = (1 if order["symbol"][:3] == "USD" else float(order["price"]))
            margin += (float(order["volume"]) * 100000) / (200 * conversion_price)
        else:
            balance += float(order["profit"])
    equity = balance - open_pl
    free_margin = equity - margin
    margin_level = "%" if margin == 0 else "%2.F" % ((equity / margin) * 100) + "%"
    equity = "%.2F" % equity
    balance = "%.2F" % balance
    open_pl = "%.2F" % open_pl
    margin = "%2.F" % margin
    return get_top_bar(user, balance, equity, margin, free_margin, margin_level, open_pl)


@app.callback(Output("user_info_update", "children"), [Input("i_user_info", "n_intervals")])
def update_user_info_timestamp(n):
    return f'Dernière mise à jour le : {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}'


@app.callback(Output("stocks_info", "children"), [Input("i_stock_list", "n_intervals")])
def update_stock_list_div(n):
    return update_stock_list()


@app.callback(Output("user_info", "children"), [Input("i_user_info", "n_intervals")])
def update_user_info_div(n):
    return update_user_info()


if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)
