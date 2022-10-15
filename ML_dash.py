from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

df = pd.read_csv('insurance.csv')

def convert_sex(x): 
  if x == 'male':
    return "0"
  elif x == 'female':
    return "1"
df['sex'] = df['sex'].apply(convert_sex) 

def convert_smoker(x): 
  if x == 'no':
    return "0"
  elif x == 'yes':
    return "1"
df['smoker'] = df['smoker'].apply(convert_smoker)

def convert_region(x): 
  if x == 'northeast': 
    return "0"
  elif x == 'southeast':
    return "1"
  elif x == 'southwest': 
    return "2"
  elif x == 'northwest': 
    return "3"

df['region'] = df['region'].apply(convert_region) 
df['region'] = df['region'].astype(int)
df['smoker'] = df['smoker'].astype(int)
df['sex'] = df['sex'].astype(int)

X = df.drop(['charges'], axis = 1)
y = df.charges
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=34)

Lin_reg = lr()
Lin_reg.fit(X_train, y_train)

accuracy = Lin_reg.score(X_test, y_test)
y_pred = Lin_reg.predict(X_test)

def Header(name):
    title = html.H2(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=9)])

corr = df.corr()
fig = px.imshow(corr, color_continuous_scale='plasma')
fig_1 = px.scatter( df, x = 'bmi', y = 'charges', color = 'smoker', template = 'ggplot2')
fig_2 = px.scatter(x = y_test, y = y_pred, template = 'ggplot2', 
                   title="Actual vs Predicted")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


cards = [
    dbc.Card(
        [
            html.H2(f"{(accuracy)*100:.2f}%", className="card-title"),
            html.P("LinearRegression Accuracy", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2(f"{len(X_train)}/{len(X_test)}", className="card-title"),
            html.P("Train / Test split", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{(df.shape)}", className="card-title"),
            html.P("rows × columns", className="card-text"),
        ],
        body=True,
        color="primary",
        inverse=True,
    ),
]

cards_1 = [
    dbc.Card(
        [
            html.H2(f"{mae:.2f}", className="card-title"),
            html.P("Mean absolute error", className="card-text"),
        ],
        body=True,
        color="secondary",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{mse:.2f}", className="card-title"),
            html.P("Mean squared error", className="card-text"),
        ],
        body=True,
        color="secondary",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{rmse:.2f}", className="card-title"),
            html.P("Root mean squared error", className="card-text"),
        ],
        body=True,
        color="secondary",
        inverse=True,
    ),
]

app.layout = dbc.Container(
    [
        Header("Medical Cost Personal with Linear Regression in Python"),
        html.Hr(),
        dbc.Row([dbc.Col(card) for card in cards]),
        html.Hr(),
        dcc.Graph(figure=fig),
        dcc.Graph(figure=fig_1),
        html.Hr(),
        html.H3('MAE & MSE & RMSE'),
        dbc.Row([dbc.Col(card) for card in cards_1]),
        dcc.Graph(figure=fig_2),
        html.Hr(),
        html.H3('Medical Cost Personal Dataset'),
        html.H6('Define the sex column. Let male equal 0, female equal 1.'),
        html.H6('Define the smoker column. Let no equal 0, yes equal 1.'),
        html.H6('Define the region column. Let northeast equal 0, southeast equal 1, southwest equal 2, northwest equal 3.'),
        dash_table.DataTable(df.to_dict('records'),[{"name": i, "id": i} for i in df.columns], id='tbl'),
        html.Br(),
    ],
    fluid=False,
)



if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8000', debug=True)