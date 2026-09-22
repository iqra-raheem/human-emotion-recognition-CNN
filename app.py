import os
import csv
import time
import io
import base64

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask,
    Response,
    render_template_string,
    redirect,
    url_for,
    request,
    session
)

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = "emotion_recognition_secret_key_123"


# =========================================================
# LOGIN DETAILS
# =========================================================

USERNAME = "iqra"
PASSWORD = "iqra@123"


# =========================================================
# MODEL
# =========================================================

model = load_model("emotion_model.keras")

emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# =========================================================
# FACE DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)


# =========================================================
# FILES
# =========================================================

HISTORY_FILE = "emotion_history.csv"

CONFUSION_MATRIX_FILE = "confusion_matrix.png"

CLASSIFICATION_REPORT_FILE = "classification_report.csv"

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# SAVE EMOTION HISTORY
# =========================================================

def save_emotion(
    emotion,
    confidence
):

    file_exists = os.path.exists(
        HISTORY_FILE
    )

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Time",
                "Emotion",
                "Confidence"
            ])

        writer.writerow([
            time.strftime(
                "%Y-%m-%d %I:%M:%S %p"
            ),
            emotion,
            round(
                float(confidence),
                2
            )
        ])


# =========================================================
# CREATE EMOTION GRAPH
# =========================================================

def create_graph():

    if not os.path.exists(
        HISTORY_FILE
    ):
        return None

    try:

        data = pd.read_csv(
            HISTORY_FILE
        )

        if data.empty:
            return None

        required_columns = [
            "Time",
            "Emotion",
            "Confidence"
        ]

        if not all(
            column in data.columns
            for column in required_columns
        ):
            return None

        data["Emotion"] = (
            data["Emotion"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        data = data[
            data["Emotion"].isin(
                emotions
            )
        ]

        if data.empty:
            return None

        emotion_counts = (
            data["Emotion"]
            .value_counts()
            .reindex(
                emotions,
                fill_value=0
            )
        )

        plt.figure(
            figsize=(7, 3.2)
        )

        bars = plt.bar(
            emotion_counts.index,
            emotion_counts.values
        )

        plt.title(
            "Emotion Analysis",
            fontsize=14
        )

        plt.xlabel(
            "Emotion",
            fontsize=10
        )

        plt.ylabel(
            "Number of Predictions",
            fontsize=10
        )

        plt.xticks(
            rotation=0
        )

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                height,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()

        image = io.BytesIO()

        plt.savefig(
            image,
            format="png",
            bbox_inches="tight",
            dpi=100
        )

        plt.close()

        image.seek(0)

        graph = base64.b64encode(
            image.getvalue()
        ).decode("utf-8")

        return graph

    except Exception as e:

        print(
            "Graph Error:",
            e
        )

        return None


# =========================================================
# GET EMOTION HISTORY
# =========================================================

def get_history():

    if not os.path.exists(
        HISTORY_FILE
    ):
        return []

    try:

        data = pd.read_csv(
            HISTORY_FILE
        )

        if data.empty:
            return []

        required_columns = [
            "Time",
            "Emotion",
            "Confidence"
        ]

        if not all(
            column in data.columns
            for column in required_columns
        ):
            return []

        data["Emotion"] = (
            data["Emotion"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        data["Confidence"] = pd.to_numeric(
            data["Confidence"],
            errors="coerce"
        )

        data = data[
            data["Emotion"].isin(
                emotions
            )
            &
            data["Confidence"].notna()
        ]

        if data.empty:
            return []

        return (
            data
            .tail(5)
            .iloc[::-1]
            .to_dict(
                orient="records"
            )
        )

    except Exception as e:

        print(
            "History Error:",
            e
        )

        return []


# =========================================================
# GET CONFUSION MATRIX
# =========================================================

def get_confusion_matrix():

    if not os.path.exists(
        CONFUSION_MATRIX_FILE
    ):
        return None

    try:

        with open(
            CONFUSION_MATRIX_FILE,
            "rb"
        ) as image_file:

            image = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return image

    except Exception as e:

        print(
            "Confusion Matrix Error:",
            e
        )

        return None


# =========================================================
# GET CLASSIFICATION REPORT
# =========================================================

def get_classification_report():

    if not os.path.exists(
        CLASSIFICATION_REPORT_FILE
    ):
        return []

    try:

        data = pd.read_csv(
            CLASSIFICATION_REPORT_FILE,
            index_col=0
        )

        data = data.reset_index()

        data = data.rename(
            columns={
                "index": "Emotion"
            }
        )

        report = []

        for _, row in data.iterrows():

            emotion = str(
                row["Emotion"]
            ).strip()

            emotion_lower = emotion.lower()

            if emotion_lower == "accuracy":
                continue

            precision = row.get(
                "precision"
            )

            recall = row.get(
                "recall"
            )

            f1 = row.get(
                "f1-score"
            )

            support = row.get(
                "support"
            )

            if (
                pd.isna(precision)
                or pd.isna(recall)
                or pd.isna(f1)
                or pd.isna(support)
            ):
                continue

            report.append({

                "Emotion": emotion,

                "Precision":
                    f"{float(precision) * 100:.2f}%",

                "Recall":
                    f"{float(recall) * 100:.2f}%",

                "F1":
                    f"{float(f1) * 100:.2f}%",

                "Support":
                    int(support)

            })

        return report

    except Exception as e:

        print(
            "Classification Report Error:",
            e
        )

        return []


# =========================================================
# GET OVERALL TEST ACCURACY
# =========================================================

def get_test_accuracy():

    if not os.path.exists(
        CLASSIFICATION_REPORT_FILE
    ):
        return None

    try:

        data = pd.read_csv(
            CLASSIFICATION_REPORT_FILE,
            index_col=0
        )

        if "accuracy" in data.index:

            row = data.loc["accuracy"]

            if (
                "precision" in data.columns
                and pd.notna(row["precision"])
            ):

                return float(
                    row["precision"]
                ) * 100

            if (
                "accuracy" in data.columns
                and pd.notna(row["accuracy"])
            ):

                return float(
                    row["accuracy"]
                ) * 100

        return None

    except Exception as e:

        print(
            "Accuracy Error:",
            e
        )

        return None


# =========================================================
# TRAINING GRAPHS
# =========================================================

def create_training_graphs():

    history_file = "training_history.json"

    if not os.path.exists(
        history_file
    ):
        return None, None

    try:

        with open(
            history_file,
            "r",
            encoding="utf-8"
        ) as file:

            history = pd.read_json(file)

        # =================================================
        # TRAINING ACCURACY
        # =================================================

        plt.figure(
            figsize=(7, 3.2)
        )

        plt.plot(
            history["accuracy"],
            label="Training Accuracy"
        )

        if "val_accuracy" in history.columns:

            plt.plot(
                history["val_accuracy"],
                label="Validation Accuracy"
            )

        plt.title(
            "Training Accuracy"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.legend()

        plt.grid(
            alpha=0.2
        )

        plt.tight_layout()

        accuracy_image = io.BytesIO()

        plt.savefig(
            accuracy_image,
            format="png",
            dpi=100
        )

        plt.close()

        accuracy_image.seek(0)

        accuracy_graph = base64.b64encode(
            accuracy_image.getvalue()
        ).decode("utf-8")


        # =================================================
        # TRAINING LOSS
        # =================================================

        plt.figure(
            figsize=(7, 3.2)
        )

        plt.plot(
            history["loss"],
            label="Training Loss"
        )

        if "val_loss" in history.columns:

            plt.plot(
                history["val_loss"],
                label="Validation Loss"
            )

        plt.title(
            "Training Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.legend()

        plt.grid(
            alpha=0.2
        )

        plt.tight_layout()

        loss_image = io.BytesIO()

        plt.savefig(
            loss_image,
            format="png",
            dpi=100
        )

        plt.close()

        loss_image.seek(0)

        loss_graph = base64.b64encode(
            loss_image.getvalue()
        ).decode("utf-8")

        return (
            accuracy_graph,
            loss_graph
        )

    except Exception as e:

        print(
            "Training Graph Error:",
            e
        )

        return None, None


# =========================================================
# GENERATE WEBCAM FRAMES
# =========================================================

def generate_frames():

    face_histories = []

    last_save_times = []

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        try:

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

        except cv2.error:

            continue

        faces = sorted(
            faces,
            key=lambda face: face[0]
        )

        if len(faces) != len(
            face_histories
        ):

            face_histories = [
                []
                for _ in faces
            ]

            last_save_times = [
                0
                for _ in faces
            ]

        for face_number, (
            x,
            y,
            w,
            h
        ) in enumerate(faces):

            face = gray[
                y:y+h,
                x:x+w
            ]

            face = cv2.resize(
                face,
                (48, 48)
            )

            face = face.astype(
                "float32"
            ) / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            face = np.expand_dims(
                face,
                axis=-1
            )

            prediction = model.predict(
                face,
                verbose=0
            )[0]

            face_histories[
                face_number
            ].append(
                prediction
            )

            if len(
                face_histories[
                    face_number
                ]
            ) > 5:

                face_histories[
                    face_number
                ].pop(0)

            average_prediction = np.mean(
                face_histories[
                    face_number
                ],
                axis=0
            )

            emotion_index = np.argmax(
                average_prediction
            )

            emotion = emotions[
                emotion_index
            ]

            confidence = (
                average_prediction[
                    emotion_index
                ] * 100
            )

            current_time = time.time()

            if (
                current_time
                -
                last_save_times[
                    face_number
                ]
                >= 2
            ):

                save_emotion(
                    emotion,
                    confidence
                )

                last_save_times[
                    face_number
                ] = current_time

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            person_text = (
                f"Person {face_number + 1}"
            )

            emotion_text = (
                f"{emotion.upper()} "
                f"({confidence:.1f}%)"
            )

            cv2.putText(
                frame,
                person_text,
                (x, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                emotion_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================================================
# DASHBOARD HTML
# =========================================================

DASHBOARD_HTML = """

<!DOCTYPE html>

<html>

<head>

    <title>
        Human Emotion Recognition
    </title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <style>

        * {
            box-sizing: border-box;
        }


        body {

            margin: 0;

            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #0f172a,
                    #1e293b
                );

            color: white;

            min-height: 100vh;

            overflow: auto;
        }


        /* ================= HEADER ================= */

        header {

            height: 85px;

            display: flex;

            justify-content: space-between;

            align-items: center;

            padding: 0 25px;

            background:
                rgba(15, 23, 42, 0.95);

            border-bottom:
                1px solid
                rgba(255,255,255,0.1);
        }


        header h1 {

            margin: 0;

            font-size: 26px;
        }


        .logout {

            background: #ef4444;

            color: white;

            text-decoration: none;

            padding: 10px 18px;

            border-radius: 8px;

            font-weight: bold;
        }


        .logout:hover {

            background: #dc2626;
        }


        /* ================= MAIN ================= */

        .container {

            padding: 15px;

            min-height:
                calc(100vh - 85px);

            overflow: visible;
        }


        /* ================= TOP CARDS ================= */

        .cards {

            display: grid;

            grid-template-columns:
                1fr
                0.9fr;

            gap: 15px;

            min-height: 300px;
        }


        .card {

            background:
                rgba(30, 41, 59, 0.95);

            border-radius: 14px;

            padding: 15px;

            box-shadow:
                0 8px 25px
                rgba(0,0,0,0.25);

            overflow: hidden;
        }


        .card h2 {

            margin:
                0 0 10px 0;

            font-size: 19px;
        }


        /* ================= CAMERA ================= */

        .camera {

            width: 100%;

            height: 235px;

            object-fit: cover;

            border-radius: 10px;

            background: black;
        }


        /* ================= UPLOAD ================= */

        .upload-box {

            border:
                2px dashed
                #64748b;

            border-radius: 10px;

            padding: 18px 10px;

            text-align: center;

            margin-bottom: 10px;
        }


        .upload-box p {

            font-size: 13px;

            color: #cbd5e1;
        }


        input[type="file"] {

            width: 100%;

            margin: 8px 0;

            font-size: 12px;
        }


        .upload-btn {

            border: none;

            background: #2563eb;

            color: white;

            padding: 9px 16px;

            border-radius: 7px;

            cursor: pointer;

            font-weight: bold;
        }


        .upload-btn:hover {

            background: #1d4ed8;
        }


        .result-image {

            width: 130px;

            height: 130px;

            object-fit: cover;

            border-radius: 10px;

            display: block;

            margin: auto;
        }


        .result-text {

            text-align: center;

            margin-top: 7px;

            font-size: 15px;
        }


        .error {

            color: #f87171;

            text-align: center;

            font-size: 13px;
        }


        /* ================= GRAPH ================= */

        .graph {

            width: 100%;

            height: 235px;

            object-fit: contain;

            background: white;

            border-radius: 10px;
        }


        .no-graph {

            height: 235px;

            display: flex;

            justify-content: center;

            align-items: center;

            color: #94a3b8;
        }


        /* ================= HISTORY ================= */

        .history-card {

            margin-top: 15px;
        }


        .history-table {

            width: 100%;

            border-collapse:
                collapse;

            font-size: 13px;
        }


        .history-table th {

            background: #334155;

            padding: 8px;

            text-align: left;
        }


        .history-table td {

            padding: 7px 8px;

            border-bottom:
                1px solid
                rgba(255,255,255,0.08);
        }


        .emotion {

            font-weight: bold;

            text-transform:
                capitalize;
        }


        /* ================= ACCURACY ================= */

        .accuracy-card {

            text-align: center;

            height: 100%;
        }


        .accuracy-number {

            font-size: 48px;

            font-weight: bold;

            margin-top: 25px;

            margin-bottom: 5px;
        }


        .accuracy-title {

            color: #cbd5e1;

            font-size: 15px;
        }


        .accuracy-info {

            color: #94a3b8;

            font-size: 12px;

            margin-top: 8px;
        }


        /* ================= RESULTS ================= */

        .results-section {

            display: grid;

            grid-template-columns:
                1fr 1fr;

            gap: 15px;

            margin-top: 15px;
        }


        .result-card {

            background:
                rgba(30, 41, 59, 0.95);

            border-radius: 14px;

            padding: 15px;

            box-shadow:
                0 8px 25px
                rgba(0,0,0,0.25);

            overflow: hidden;
        }


        .result-card h2 {

            margin-top: 0;

            font-size: 19px;
        }


        /* ================= CONFUSION MATRIX ================= */

        .confusion-image {

            width: 100%;

            max-height: 500px;

            object-fit: contain;

            background: white;

            border-radius: 10px;
        }


        /* ================= CLASSIFICATION REPORT ================= */

        .report-table {

            width: 100%;

            border-collapse:
                collapse;

            font-size: 13px;
        }


        .report-table th {

            background: #334155;

            padding: 9px;

            text-align: center;
        }


        .report-table td {

            padding: 8px;

            text-align: center;

            border-bottom:
                1px solid
                rgba(255,255,255,0.08);
        }


        .report-table tr:hover {

            background:
                rgba(255,255,255,0.05);
        }


        /* ================= MOBILE ================= */

        @media(max-width: 900px) {

            body {

                overflow: auto;
            }


            .container {

                height: auto;

                overflow: visible;
            }


            .cards {

                grid-template-columns: 1fr;

                height: auto;
            }


            .card {

                height: auto;
            }


            .results-section {

                grid-template-columns: 1fr;
            }

        }

    </style>

</head>


<body>


<!-- ================= HEADER ================= -->

<header>

    <h1>
        Human Emotion Recognition
    </h1>

    <a
        href="/logout"
        class="logout"
    >
        Logout
    </a>

</header>


<div class="container">


    <!-- ================= TOP SECTION ================= -->

    <div class="cards">


        <!-- ================= CAMERA ================= -->

        <div class="card">

            <h2>📷 Live Webcam</h2>

            <img
                src="/video_feed"
                class="camera"
            >

        </div>


        <!-- ================= UPLOAD ================= -->

        <div class="card">

            <h2>📤 Upload Image</h2>


            {% if upload_result %}

                <img
                    src="data:image/jpeg;base64,{{ upload_result }}"
                    class="result-image"
                >

                <div class="result-text">

                    <b>Emotion:</b>
                    {{ upload_emotion|capitalize }}

                    <br>

                    <b>Confidence:</b>
                    {{ "%.1f"|format(upload_confidence) }}%

                </div>


            {% elif upload_error %}

                <div class="error">

                    {{ upload_error }}

                </div>


            {% else %}

                <div class="upload-box">

                    <p>

                        Upload a face image
                        to detect emotion.

                    </p>


                    <form
                        action="/predict_image"
                        method="POST"
                        enctype="multipart/form-data"
                    >

                        <input
                            type="file"
                            name="image"
                            accept="image/*"
                            required
                        >


                        <button
                            type="submit"
                            class="upload-btn"
                        >

                            Predict Emotion

                        </button>

                    </form>

                </div>


                <p
                    style="
                        text-align:center;
                        color:#94a3b8;
                        font-size:12px;
                    "
                >

                    Supported:
                    JPG, PNG, JPEG

                </p>

            {% endif %}


            {% if upload_result or upload_error %}

                <form
                    action="/predict_image"
                    method="POST"
                    enctype="multipart/form-data"
                    style="text-align:center;"
                >

                    <input
                        type="file"
                        name="image"
                        accept="image/*"
                        required
                    >


                    <button
                        type="submit"
                        class="upload-btn"
                    >

                        Upload Another

                    </button>

                </form>

            {% endif %}

        </div>


    </div>


    <!-- ================================================= -->
    <!-- TRAINING GRAPHS -->
    <!-- ================================================= -->

    <div class="results-section">


        <!-- ================= TRAINING ACCURACY ================= -->

        <div class="card">

            <h2>📈 Training Accuracy</h2>

            {% if training_accuracy_graph %}

                <img
                    src="data:image/png;base64,{{ training_accuracy_graph }}"
                    class="graph"
                >

            {% else %}

                <div class="no-graph">

                    Training accuracy graph
                    not available.

                </div>

            {% endif %}

        </div>


        <!-- ================= TRAINING LOSS ================= -->

        <div class="card">

            <h2>📉 Training Loss</h2>

            {% if training_loss_graph %}

                <img
                    src="data:image/png;base64,{{ training_loss_graph }}"
                    class="graph"
                >

            {% else %}

                <div class="no-graph">

                    Training loss graph
                    not available.

                </div>

            {% endif %}

        </div>


    </div>


    <!-- ================================================= -->
    <!-- HISTORY -->
    <!-- ================================================= -->

    <div class="card history-card">

        <h2>📋 Emotion History</h2>


        {% if history %}

            <table
                class="history-table"
            >

                <tr>

                    <th>Time</th>

                    <th>Emotion</th>

                    <th>Confidence</th>

                </tr>


                {% for row in history %}

                <tr>

                    <td>
                        {{ row["Time"] }}
                    </td>

                    <td class="emotion">
                        {{ row["Emotion"] }}
                    </td>

                    <td>
                        {{ row["Confidence"] }}%
                    </td>

                </tr>

                {% endfor %}

            </table>

        {% else %}

            <p
                style="color:#94a3b8;"
            >

                No emotion predictions
                recorded yet.

            </p>

        {% endif %}

    </div>


    <!-- ================================================= -->
    <!-- EMOTION ANALYSIS + TEST ACCURACY -->
    <!-- ================================================= -->

    <div class="results-section">


        <!-- ================= EMOTION ANALYSIS ================= -->

        <div class="card">

            <h2>📊 Emotion Analysis</h2>


            {% if graph %}

                <img
                    src="data:image/png;base64,{{ graph }}"
                    class="graph"
                >

            {% else %}

                <div class="no-graph">

                    No emotion history
                    available yet.

                </div>

            {% endif %}

        </div>


        <!-- ================= TEST ACCURACY ================= -->

        <div class="card accuracy-card">

            <h2>🎯 Model Evaluation</h2>


            {% if test_accuracy is not none %}

                <div class="accuracy-number">

                    {{ "%.2f"|format(test_accuracy) }}%

                </div>

                <div class="accuracy-title">

                    Overall Test Accuracy

                </div>

                <div class="accuracy-info">

                    Evaluated on 7,178 test images

                </div>

            {% else %}

                <p
                    style="color:#94a3b8;"
                >

                    Test accuracy is not available.

                </p>

            {% endif %}

        </div>


    </div>


    <!-- ================================================= -->
    <!-- EVALUATION RESULTS -->
    <!-- ================================================= -->

    <div class="results-section">


        <!-- ================= CONFUSION MATRIX ================= -->

        <div class="result-card">

            <h2>📊 Confusion Matrix</h2>


            {% if confusion_matrix %}

                <img
                    src="data:image/png;base64,{{ confusion_matrix }}"
                    class="confusion-image"
                >

            {% else %}

                <p
                    style="color:#94a3b8;"
                >

                    Confusion matrix
                    is not available.

                </p>

            {% endif %}

        </div>


        <!-- ================= CLASSIFICATION REPORT ================= -->

        <div class="result-card">

            <h2>📑 Classification Report</h2>


            {% if classification_report %}

                <table
                    class="report-table"
                >

                    <tr>

                        <th>Emotion</th>

                        <th>Precision</th>

                        <th>Recall</th>

                        <th>F1-Score</th>

                        <th>Support</th>

                    </tr>


                    {% for row in classification_report %}

                    <tr>

                        <td>
                            {{ row["Emotion"]|capitalize }}
                        </td>

                        <td>
                            {{ row["Precision"] }}
                        </td>

                        <td>
                            {{ row["Recall"] }}
                        </td>

                        <td>
                            {{ row["F1"] }}
                        </td>

                        <td>
                            {{ row["Support"] }}
                        </td>

                    </tr>

                    {% endfor %}

                </table>

            {% else %}

                <p
                    style="color:#94a3b8;"
                >

                    Classification report
                    is not available.

                </p>

            {% endif %}

        </div>


    </div>


</div>


</body>

</html>

"""


# =========================================================
# LOGIN PAGE
# =========================================================

LOGIN_HTML = """

<!DOCTYPE html>

<html>

<head>

    <title>
        Login - Emotion Recognition
    </title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <style>

        * {
            box-sizing: border-box;
        }


        body {

            margin: 0;

            height: 100vh;

            display: flex;

            justify-content: center;

            align-items: center;

            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #0f172a,
                    #1e3a8a
                );
        }


        .login-box {

            width: 360px;

            background: #1e293b;

            padding: 35px;

            border-radius: 15px;

            box-shadow:
                0 10px 30px
                rgba(0,0,0,0.4);

            color: white;
        }


        h1 {

            text-align: center;

            margin-bottom: 25px;

            font-size: 24px;
        }


        label {

            display: block;

            margin-bottom: 6px;

            font-size: 14px;
        }


        input {

            width: 100%;

            padding: 11px;

            margin-bottom: 15px;

            border: none;

            border-radius: 7px;

            outline: none;
        }


        button {

            width: 100%;

            padding: 11px;

            border: none;

            border-radius: 7px;

            background: #2563eb;

            color: white;

            font-weight: bold;

            cursor: pointer;
        }


        button:hover {

            background: #1d4ed8;
        }


        .error {

            color: #f87171;

            text-align: center;

            margin-bottom: 15px;

            font-size: 14px;
        }

    </style>

</head>


<body>


<div class="login-box">


    <h1>
        Emotion Recognition
    </h1>


    {% if error %}

        <div class="error">

            {{ error }}

        </div>

    {% endif %}


    <form method="POST">


        <label>
            Username
        </label>

        <input
            type="text"
            name="username"
            placeholder="Enter username"
            required
        >


        <label>
            Password
        </label>

        <input
            type="password"
            name="password"
            placeholder="Enter password"
            required
        >


        <button
            type="submit"
        >

            Login

        </button>


    </form>


</div>


</body>

</html>

"""


# =========================================================
# LOGIN ROUTE
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username"
        )

        password = request.form.get(
            "password"
        )

        if (
            username == USERNAME
            and
            password == PASSWORD
        ):

            session["logged_in"] = True

            return redirect(
                url_for("dashboard")
            )

        else:

            return render_template_string(

                LOGIN_HTML,

                error=
                "Invalid username or password"

            )

    return render_template_string(

        LOGIN_HTML,

        error=""

    )


# =========================================================
# DASHBOARD ROUTE
# =========================================================

@app.route("/")
def dashboard():

    if not session.get(
        "logged_in"
    ):

        return redirect(
            url_for("login")
        )

    accuracy_graph, loss_graph = (
        create_training_graphs()
    )

    return render_template_string(

        DASHBOARD_HTML,

        graph=create_graph(),

        training_accuracy_graph=
            accuracy_graph,

        training_loss_graph=
            loss_graph,

        history=get_history(),

        confusion_matrix=
            get_confusion_matrix(),

        classification_report=
            get_classification_report(),

        test_accuracy=
            get_test_accuracy(),

        upload_result=None,

        upload_emotion=None,

        upload_confidence=None,

        upload_error=None

    )


# =========================================================
# IMAGE UPLOAD + CNN PREDICTION
# =========================================================

@app.route(
    "/predict_image",
    methods=["POST"]
)
def predict_image():

    if not session.get(
        "logged_in"
    ):

        return redirect(
            url_for("login")
        )

    file = request.files.get(
        "image"
    )

    # Create training graphs
    accuracy_graph, loss_graph = (
        create_training_graphs()
    )

    # =====================================================
    # CHECK FILE
    # =====================================================

    if not file or file.filename == "":

        return render_template_string(

            DASHBOARD_HTML,

            graph=create_graph(),

            training_accuracy_graph=
                accuracy_graph,

            training_loss_graph=
                loss_graph,

            history=get_history(),

            confusion_matrix=
                get_confusion_matrix(),

            classification_report=
                get_classification_report(),

            test_accuracy=
                get_test_accuracy(),

            upload_result=None,

            upload_emotion=None,

            upload_confidence=None,

            upload_error=
                "Please select an image."

        )


    # =====================================================
    # SAVE UPLOADED IMAGE
    # =====================================================

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(
        filepath
    )


    # =====================================================
    # READ IMAGE
    # =====================================================

    image = cv2.imread(
        filepath
    )

    if image is None:

        return render_template_string(

            DASHBOARD_HTML,

            graph=create_graph(),

            training_accuracy_graph=
                accuracy_graph,

            training_loss_graph=
                loss_graph,

            history=get_history(),

            confusion_matrix=
                get_confusion_matrix(),

            classification_report=
                get_classification_report(),

            test_accuracy=
                get_test_accuracy(),

            upload_result=None,

            upload_emotion=None,

            upload_confidence=None,

            upload_error=
                "Invalid image file."

        )


    # =====================================================
    # CONVERT TO GRAYSCALE
    # =====================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # RESIZE FOR CNN
    # =====================================================

    face = cv2.resize(
        gray,
        (48, 48)
    )


    # =====================================================
    # NORMALIZE
    # =====================================================

    face = face.astype(
        "float32"
    ) / 255.0


    # =====================================================
    # PREPARE CNN INPUT
    # =====================================================

    face = np.expand_dims(
        face,
        axis=0
    )

    face = np.expand_dims(
        face,
        axis=-1
    )


    # =====================================================
    # CNN PREDICTION
    # =====================================================

    prediction = model.predict(
        face,
        verbose=0
    )[0]

    emotion_index = np.argmax(
        prediction
    )

    emotion = emotions[
        emotion_index
    ]

    confidence = (
        prediction[
            emotion_index
        ] * 100
    )


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    save_emotion(
        emotion,
        confidence
    )


    # =====================================================
    # SHOW RESULT ON IMAGE
    # =====================================================

    text = (
        f"{emotion.upper()} "
        f"({confidence:.1f}%)"
    )

    cv2.putText(
        image,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # =====================================================
    # ENCODE IMAGE
    # =====================================================

    success, encoded_image = cv2.imencode(
        ".jpg",
        image
    )

    if not success:

        return render_template_string(

            DASHBOARD_HTML,

            graph=create_graph(),

            training_accuracy_graph=
                accuracy_graph,

            training_loss_graph=
                loss_graph,

            history=get_history(),

            confusion_matrix=
                get_confusion_matrix(),

            classification_report=
                get_classification_report(),

            test_accuracy=
                get_test_accuracy(),

            upload_result=None,

            upload_emotion=None,

            upload_confidence=None,

            upload_error=
                "Could not process image."

        )


    upload_result = base64.b64encode(
        encoded_image.tobytes()
    ).decode(
        "utf-8"
    )


    # =====================================================
    # SHOW RESULT
    # =====================================================

    return render_template_string(

        DASHBOARD_HTML,

        graph=create_graph(),

        training_accuracy_graph=
            accuracy_graph,

        training_loss_graph=
            loss_graph,

        history=get_history(),

        confusion_matrix=
            get_confusion_matrix(),

        classification_report=
            get_classification_report(),

        test_accuracy=
            get_test_accuracy(),

        upload_result=
            upload_result,

        upload_emotion=
            emotion,

        upload_confidence=
            confidence,

        upload_error=None

    )


# =========================================================
# VIDEO FEED
# =========================================================

@app.route(
    "/video_feed"
)
def video_feed():

    if not session.get(
        "logged_in"
    ):

        return redirect(
            url_for("login")
        )

    return Response(

        generate_frames(),

        mimetype=
            "multipart/x-mixed-replace; boundary=frame"

    )


# =========================================================
# LOGOUT
# =========================================================

@app.route(
    "/logout"
)
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )