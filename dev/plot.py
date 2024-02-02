import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .constants import PERFORMANCE_METRICS, QUALITY_METRICS, ATTACK_NAMES

colors = (
    px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
)


def style_progress_dataframe(row_list):
    dataframe = pd.DataFrame(
        row_list,
        columns=[
            "Dataset",
            "Source",
            "Attack",
            "Strength",
            "Generated",
            "Reversed",
            "Decoded",
            "Measured",
        ],
    ).sort_values(
        by=["Dataset", "Source", "Attack", "Strength"],
        ascending=[True, True, True, True],
    )
    dataframe = dataframe.astype(
        {
            "Dataset": "string",
            "Source": "string",
            "Attack": "string",
            "Strength": "string",
            "Generated": "Int64",
            "Reversed": "Int64",
            "Decoded": "Int64",
            "Measured": "Int64",
        }
    )

    def style_rows_by_status(row):
        if pd.isna(row["Generated"]) or row["Generated"] < 5000:
            return (
                [""] * row.index.get_loc("Generated")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Generated") - 1)
            )
        else:
            return [""] * len(row)

    def style_rows_by_reverse(row):
        if (row["Source"] == "real" or row["Source"].endswith("tree_ring")) and (
            pd.isna(row["Reversed"]) or row["Reversed"] < 5000
        ):
            return (
                [""] * row.index.get_loc("Reversed")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Reversed") - 1)
            )
        else:
            return [""] * len(row)

    def style_rows_by_decode(row):
        if pd.isna(row["Decoded"]) or row["Decoded"] < 5000:
            return (
                [""] * row.index.get_loc("Decoded")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Decoded") - 1)
            )
        else:
            return [""] * len(row)

    def style_rows_by_metric(row):
        if not pd.isna(row["Attack"]) and (
            pd.isna(row["Measured"]) or row["Measured"] < 5000
        ):
            return (
                [""] * row.index.get_loc("Measured")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Measured") - 1)
            )
        else:
            return [""] * len(row)

    styler = (
        dataframe.style.apply(style_rows_by_status, axis=1)
        .apply(style_rows_by_reverse, axis=1)
        .apply(style_rows_by_decode, axis=1)
        .apply(style_rows_by_metric, axis=1)
    )

    return styler


def aggregate_comparison_dataframe(row_list):
    dataframe = pd.DataFrame(
        row_list,
        columns=[
            "Attack",
            "Strength",
            *(list(PERFORMANCE_METRICS.values())),
            *(list(QUALITY_METRICS.values())),
        ],
    ).sort_values(
        by=["Attack", "Strength"],
        ascending=[True, True],
    )
    dataframe = dataframe.astype(
        {
            "Attack": "string",
            "Strength": "string",
            **{v: "float64" for v in PERFORMANCE_METRICS.values()},
            **{v: "float64" for v in PERFORMANCE_METRICS.values()},
        }
    )
    return dataframe


def plot_parallel_coordinates(dataframe):
    fig_performance = go.Figure(
        data=go.Parcoords(
            dimensions=[
                {"label": k, "values": dataframe[k]}
                for k in PERFORMANCE_METRICS.values()
            ],
        ),
        layout=go.Layout(
            title="Correlation of Performances",
            margin={"l": 500, "r": 500, "t": 500, "b": 500},
        ),
    )

    fig_quality = go.Figure(
        data=go.Parcoords(
            dimensions=[
                {"label": k, "values": dataframe[k]} for k in QUALITY_METRICS.values()
            ],
        ),
        layout=go.Layout(
            title="Correlation of Qualities",
            margin={"l": 500, "r": 500, "t": 500, "b": 500},
        ),
    )

    return fig_performance, fig_quality


def plot_2d_comparison(
    dataframe,
    performance_metric,
    quality_metric,
    show_text=False,
    line_width=3,
    marker_size=9,
    tick_size=10,
    legend_fontsize=15,
    plot_height=800,
):
    fig = go.Figure()

    for i, attack_name in enumerate(ATTACK_NAMES.keys()):
        if attack_name.startswith("dist"):
            marker = "square"
        elif attack_name.startswith("adv"):
            marker = "star"
        else:
            marker = "x"

        df_cat = dataframe[dataframe["Attack"] == attack_name]
        df_cat = df_cat.assign(Strength_float=df_cat["Strength"].astype(float))
        df_cat = df_cat.sort_values("Strength_float").drop(columns=["Strength_float"])
        fig.add_trace(
            go.Scatter(
                x=df_cat[quality_metric],
                y=df_cat[performance_metric],
                text=df_cat["Strength"],
                mode="lines+markers+text" if show_text else "lines+markers",
                name=ATTACK_NAMES[attack_name]
                if attack_name in ATTACK_NAMES
                else "N/A",
                line=dict(color=colors[i % len(colors)], width=line_width),
                marker=dict(symbol=marker, size=marker_size),
                textposition="bottom right",
            )
        )

    # Adjust the layout for the line plot and add legend
    fig.update_layout(
        title_text="Comparison of Attacks",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            font=dict(size=legend_fontsize),
        ),
        xaxis=dict(title=quality_metric, tickfont=dict(size=tick_size)),
        yaxis=dict(title=performance_metric, tickfont=dict(size=tick_size)),
        height=plot_height,
    )

    return fig
