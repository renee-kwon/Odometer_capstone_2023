import numpy as np
import pandas as pd
import plotly.graph_objects as go


def evaluate(results_df, odo_threshold, digit_threshold):
    results_df["odo_detected"] = np.where(
        results_df["odo_conf"] > odo_threshold, True, False
    )
    results_df["digits_detected"] = np.where(
        results_df["value_conf"] > digit_threshold, True, False
    )
    results_df["correct"] = np.where(
        results_df["VERIFIED_ODOMETER_READING"] == results_df["pred"],
        "true",
        np.where(
            results_df["VERIFIED_ODOMETER_READING"] == -1,
            "non-odometer",
            np.where(
                results_df["VERIFIED_ODOMETER_READING"] > results_df["pred"],
                "under",
                "over",
            ),
        ),
    )

    return results_df


def count(df):
    summary = []

    # of the images of odometers
    results_odo = df[df["VERIFIED_ODOMETER_READING"] >= 0]
    summary.append(
        {
            "source": "Odometer Images",
            "target": "Odometer detected",
            "value": results_odo["odo_detected"].sum(),
        }
    )
    summary.append(
        {
            "source": "Odometer Images",
            "target": "No odometer detected",
            "value": len(results_odo) - results_odo["odo_detected"].sum(),
        }
    )

    # of the images of Non-odometers
    results_nodo = df[df["VERIFIED_ODOMETER_READING"] == -1]
    summary.append(
        {
            "source": "Non-Odometer Images",
            "target": "Odometer detected",
            "value": results_nodo["odo_detected"].sum(),
        }
    )
    summary.append(
        {
            "source": "Non-Odometer Images",
            "target": "No odometer detected",
            "value": len(results_nodo) - results_nodo["odo_detected"].sum(),
        }
    )

    # of the odometer detected images
    odo_detected = df[df["odo_detected"] == True]
    summary.append(
        {
            "source": "Odometer detected",
            "target": "Digits detected",
            "value": odo_detected["digits_detected"].sum(),
        }
    )
    summary.append(
        {
            "source": "Odometer detected",
            "target": "No Digits detected",
            "value": len(odo_detected) - odo_detected["digits_detected"].sum(),
        }
    )

    # of odometer and digit detected images
    digit_detected = odo_detected[odo_detected["digits_detected"] == True]
    summary.append(
        {
            "source": "Digits detected",
            "target": "Correct Predictions",
            "value": digit_detected["correct"].value_counts().get("true", 0),
        }
    )
    summary.append(
        {
            "source": "Digits detected",
            "target": "False Predictions",
            "value": digit_detected["correct"].value_counts().get("under", 0)
            + digit_detected["correct"].value_counts().get("over", 0),
        }
    )

    # of the falsly prediced odometer and digit detected images
    summary.append(
        {
            "source": "False Predictions",
            "target": "Over predicted",
            "value": digit_detected["correct"].value_counts().get("over", 0),
        }
    )
    summary.append(
        {
            "source": "False Predictions",
            "target": "Under predicted",
            "value": digit_detected["correct"].value_counts().get("under", 0),
        }
    )
    summary.append(
        {
            "source": "False Predictions",
            "target": "Not a odometer",
            "value": digit_detected["correct"].value_counts().get("non-odometer", 0),
        }
    )

    label = get_labels(summary)
    label = sorted(label)

    source = []
    target = []
    value = []

    for item in summary:
        source.append(label.index(item["source"]))
        target.append(label.index(item["target"]))
        value.append(item["value"])

    return label, {"source": source, "target": target, "value": value}


def get_labels(data):
    return list(
        set([item["source"] for item in data] + [item["target"] for item in data])
    )

def get_label_counts(links, labels):
    # Calculate node counts based on links
    label_counts = {}
    total_count = sum(links['value'])  # Calculate the total count

    for link in range(len(links['source'])):
        source_label = labels[links['source'][link]]
        target_label = labels[links['target'][link]]
        count = links['value'][link]

        if source_label in label_counts:
            label_counts[source_label] += count
        else:
            label_counts[source_label] = count

        if target_label in label_counts:
            label_counts[target_label] += count
        else:
            label_counts[target_label] = count

    # Update the count for "Odometer detected" and "Digits detected" nodes
    label_counts["Odometer detected"] = links['value'][0]
    label_counts["Digits detected"] = links['value'][4]

    # Create labels with counts and percentages
    label_with_counts = []
    for label in labels:
        count = label_counts.get(label, 0)
        percentage = (count / label_counts['Odometer Images']) * 100
        label_with_counts.append(f"{label} ({count})")
    
    return label_with_counts

def make_sankey_plot(results_df, odo_threshold, digit_threshold):
    df = evaluate(results_df, odo_threshold, digit_threshold)
    labels, links = count(df)

    label_with_counts = get_label_counts(links, labels)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label_with_counts,
                    color="blue",
                    x=[0.7, 0.5, 0.7, 0.5, 0.25, 0, 1, 0, 0.25, 1, 1],
                    y=[0.1, 0.1, 0.4, 0.5, 0.75, 0.75, 0.8, 0.25, 0.25, 0.60, 0.70],
                ),
                link=links,
            )
        ]
    )
    return fig