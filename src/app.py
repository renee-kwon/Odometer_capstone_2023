import base64
import cv2
import dash_bootstrap_components as dbc
import io
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import torch
from dash import dash_table


from dash import dash, html, dcc, Input, Output, State, dash_table, callback_context
from PIL import Image
from ultralytics import YOLO

# modules from src/
import plot
import predict

UPLOAD_DIRECTORY = "tmp/app"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])


# load results df to make sankey plot
results_df = pd.read_json("outputs/Test_Data_Results.json")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
odo_model_pth = "outputs/odo.pt"
digit_model_pth = "outputs/digit.pt"

# load models
odo_model = YOLO(odo_model_pth)
digit_model = YOLO(digit_model_pth)


app.layout = html.Div(
    style={"padding": "1%"},
    children=[
        html.H1(
            "ICBC Odometer Detection Demo",
            style={"text-align": "center", "padding": "1% 0%"},
        ),
        dcc.Tabs(
            style={"margin": "1% 5%"},
            children=[
                dcc.Tab(
                    label="Single Image Test",
                    children=[
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            style={"height": "100%", "margin-top": "25%"},
                            children=[
                                html.Div(
                                    style={
                                        "width": "45%",
                                        "float": "left",
                                        "padding": "1%",
                                        "margin-left": "5%",
                                    },
                                    children=[
                                        # column 1
                                        html.Div(
                                            id="output-image-upload",
                                            style={
                                                "textAlign": "center",
                                                "margin": "2%",
                                            },
                                        ),
                                        html.Div(
                                            id="output-prediction-digits",
                                            style={
                                                "textAlign": "center",
                                                "margin": "2%",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "width": "45%",
                                        "float": "right",
                                        "padding": "1%",
                                        "margin-right": "5%",
                                    },
                                    children=[
                                        # column 2
                                        dcc.Upload(
                                            id="upload-image",
                                            children=html.Div(
                                                [
                                                    "Drag and Drop or ",
                                                    html.A("Select an Image"),
                                                ]
                                            ),
                                            style={
                                                "height": "100px",
                                                "lineHeight": "100px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "5px",
                                                "textAlign": "center",
                                                "margin": "5%",
                                            },
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="output-prediction-text",
                                            style={
                                                "textAlign": "center",
                                                "padding": "5% 20%",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Test Set Accuracy",
                    children=[
                        html.Div(
                            style={
                                "width": "70%",
                                "float": "left",
                                "padding": "1%",
                                "margin-left": "5%",
                            },
                            children=[
                                # column 1
                                dcc.Loading(
                                    id="loading2",
                                    type="circle",
                                    style={"height": "100%", "margin-top": "25%"},
                                    children=[
                                        html.H2("Sankey Diagram of Image Detection"),
                                        dcc.Graph(id="sankey"),
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            style={
                                "width": "20%",
                                "float": "right",
                                "padding": "1%",
                                "margin-right": "5%",
                            },
                            children=[
                                # column 2
                                html.H3("Confidence Threshold"),
                                html.P(
                                    "Adjust the model confidence threshold of each of the stages in the model.",
                                    style={"margin-top": "1%", "margin-bottom": "15%"},
                                ),
                                html.H5(
                                    "Odometer Threshold", style={"margin": "5% 0%"}
                                ),
                                dcc.Slider(
                                    0,
                                    1,
                                    0.01,
                                    value=0.75,
                                    id="odo_threshold",
                                    marks=None,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                html.H5("Digit Threshold", style={"margin": "5% 0%"}),
                                dcc.Slider(
                                    0.3,
                                    1,
                                    0.01,
                                    value=0.85,
                                    id="digit_threshold",
                                    marks=None,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.callback(
    [
        Output("output-image-upload", "children"),
        Output("output-prediction-digits", "children"),
        Output("output-prediction-text", "children"),
    ],
    [Input("upload-image", "contents")],
    [State("upload-image", "contents")],
    prevent_initial_call=True,
)
def run_model(list_of_contents, contents_state):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    img = parse_contents(contents_state)
    image_path = os.path.join(UPLOAD_DIRECTORY, "temp.jpg")
    cv2.imwrite(image_path, img)

    # Run Model
    image = cv2.imread(image_path)
    result = predict.predict(
        odo_model, digit_model, image, device, return_raw_preds=True
    )

    # output-image-upload
    output_img = result["odo_result"][0].plot()
    output_img = cv2.resize(
        output_img,
        (
            500,
            int(500 * output_img.shape[0] / output_img.shape[1]),
        ),
    )
    _, encoded_img = cv2.imencode(".png", output_img)
    encoded_img = base64.b64encode(encoded_img).decode("utf-8")

    # output-prediction-digits
    digit_img = result["digit_results"][0].plot()
    buffered = io.BytesIO()
    Image.fromarray(cv2.cvtColor(digit_img, cv2.COLOR_BGR2RGB)).save(
        buffered, format="JPEG"
    )
    encoded_digits_img = base64.b64encode(buffered.getvalue()).decode()

    # output-prediction-text
    text_results = [
        html.P(f"Prediction: {result['pred']}"),
        html.P(f"Odometer Confidence: {result['odo_conf']:.3f}"),
    ]

    digits_table = dash_table.DataTable(
        data=[
            {"Digit": "{:.0f}".format(val), "Confidence": "{:.3f}".format(conf)}
            for val, conf in zip(result["digits"], result["digits_conf"])
        ],
        columns=[
            {"name": "Digit", "id": "Digit"},
            {"name": "Confidence", "id": "Confidence"},
        ],
        style_cell={
            "textAlign": "center",
        },
    )

    return (
        html.Img(
            src="data:image/png;base64,{}".format(encoded_img),
            style={
                "width": "auto",
                "height": "auto",
                "display": "block",
                "margin-left": "auto",
                "margin-right": "auto",
            },
        ),
        html.Img(
            src="data:image/jpg;base64,{}".format(encoded_digits_img),
            style={
                "width": "500px",
                "height": "auto",
                "display": "block",
                "margin-left": "auto",
                "margin-right": "auto",
            },
        ),
        text_results + [digits_table],
    )


@app.callback(
    [Output("sankey", "figure")],
    [Input("odo_threshold", "value"), Input("digit_threshold", "value")],
)
def update_sankey(odo_threshold, digit_threshold):
    fig = plot.make_sankey_plot(results_df, odo_threshold, digit_threshold)
    return [fig]


if __name__ == "__main__":
    app.run_server(debug=True)
