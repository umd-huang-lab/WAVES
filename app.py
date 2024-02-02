import os
import numpy as np
import gradio as gr
from git import Repo
from dev import *
from dotenv import load_dotenv

load_dotenv()
result_dir = os.environ.get("RESULT_DIR")
github_token = os.environ.get("GITHUB_TOKEN")
repo_url = os.environ.get("REPO_URL")
branch_name = os.environ.get("BRANCH_NAME")

SOURCE_NAME_DICT_REVERSED = {
    "Not Watermarked": "real",
    **{v: k for k, v in WATERMARK_METHODS.items()},
}


def reload_results():
    try:
        if github_token is not None:
            # If github_token is provided, then it is on the HF space
            repo = Repo(result_dir)
            repo.remotes["origin"].set_url(
                f"https://{github_token}:x-oauth-basic@{repo_url}"
            )
            repo.remotes["origin"].pull()
        else:
            # If github_token is not provided, then it is on the local machine
            repo = Repo(result_dir)
            repo.git.pull()
        gr.Info(f"Reload results successfully, {len(get_all_json_paths())} JSONs found")
        gr.Info(
            f"Results last updated at {repo.head.commit.committed_datetime.strftime('%b %d, %H:%M:%S')}"
        )
    except Exception as e:
        raise gr.Error(e)


def clear_cache():
    try:
        clear_aggregated_cache()
        gr.Info("Aggregated cache successfully cleared")
    except Exception as e:
        raise gr.Error(e)


####################################################################################################
# Tab: Experiment Progress


def show_experiment_progress(
    progress_dataset_name_dropdown,
    progress_source_name_dropdown,
    progress=gr.Progress(),
):
    try:
        dataset_name = (
            {v: k for k, v in DATASET_NAMES.items()}[progress_dataset_name_dropdown]
            if progress_dataset_name_dropdown != "All"
            else None
        )
        source_name = (
            SOURCE_NAME_DICT_REVERSED[progress_source_name_dropdown]
            if progress_source_name_dropdown != "All"
            else None
        )
        json_dict = get_all_json_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                (_dataset_name == dataset_name if dataset_name else True)
                and (_source_name.startswith(source_name) if source_name else True)
            )
        )
        progress_dict = {
            (key[0], key[3], key[1], key[2]): [None, None, None, None]
            for key in json_dict.keys()
        }
        gr.Info(f"Found {len(progress_dict)} records")
        for key, json_path in progress.tqdm(json_dict.items()):
            progress_dict[(key[0], key[3], key[1], key[2])][
                ["status", "reverse", "decode", "metric"].index(key[4])
            ] = get_progress_from_json(json_path)

        return style_progress_dataframe(
            [[*key, *progress_dict[key]] for key in progress_dict.keys()]
        )
    except Exception as e:
        raise gr.Error(e)


####################################################################################################
# Tab: Attack Comparison


def aggregate_result_dataframe(
    compare_dataset_name_dropdown,
    compare_source_name_dropdown,
    compare_eval_setup_dropdown,
    progress=gr.Progress(),
):
    try:
        dataset_name = {v: k for k, v in DATASET_NAMES.items()}[
            compare_dataset_name_dropdown
        ]
        source_name = {v: k for k, v in WATERMARK_METHODS.items()}[
            compare_source_name_dropdown
        ]
        mode = {v: k for k, v in EVALUATION_SETUPS.items()}[compare_eval_setup_dropdown]
        json_dict = get_all_json_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                (_dataset_name == dataset_name if dataset_name else True)
                and (_source_name.startswith(source_name) if source_name else True)
                and _attack_name is not None
                and _attack_strength is not None
            )
        )

        attack_names_and_strengths = sorted(
            list(set([(key[1], key[2]) for key in json_dict.keys()]))
        )
        comparison_row_list = []
        for attack_name, attack_strength in progress.tqdm(attack_names_and_strengths):
            performance_dict = get_performance(
                dataset_name, source_name, attack_name, attack_strength, mode
            )
            quality_dict = get_quality(
                dataset_name, source_name, attack_name, attack_strength, mode
            )
            comparison_row_list.append(
                [
                    attack_name,
                    attack_strength,
                    *[performance_dict[k] for k in PERFORMANCE_METRICS.keys()],
                    *[
                        quality_dict[k][0] if quality_dict[k] is not None else None
                        for k in QUALITY_METRICS.keys()
                    ],
                ]
            )

        compare_result_dataframe = aggregate_comparison_dataframe(comparison_row_list)
        fig_performance, fig_quality = plot_parallel_coordinates(
            compare_result_dataframe
        )

        return [compare_result_dataframe, fig_performance, fig_quality]

    except Exception as e:
        raise gr.Error(e)


def draw_2d_plot(
    compare_result_dataframe,
    compare_performance_mode_dropdown,
    compare_quality_mode_dropdown,
    compare_show_strength_checkbox,
    compare_line_width_slider,
    compare_marker_size_slider,
    compare_tick_size_slider,
    compare_legend_fontsize_slider,
    compare_plot_height_slider,
):
    try:
        return plot_2d_comparison(
            compare_result_dataframe,
            compare_performance_mode_dropdown,
            compare_quality_mode_dropdown,
            compare_show_strength_checkbox,
            compare_line_width_slider,
            compare_marker_size_slider,
            compare_tick_size_slider,
            compare_legend_fontsize_slider,
            compare_plot_height_slider,
        )
    except Exception as e:
        raise gr.Error(e)


####################################################################################################
# Tab: Image Folder Viewer


def find_folder_by_dataset_source(
    folder_dataset_name_dropdown,
    folder_source_name_dropdown,
):
    if not folder_dataset_name_dropdown or not folder_source_name_dropdown:
        gr.Info("Please select dataset and source first")
        return gr.update(choices=[])
    try:
        dataset_name = {v: k for k, v in DATASET_NAMES.items()}[
            folder_dataset_name_dropdown
        ]
        source_name = SOURCE_NAME_DICT_REVERSED[folder_source_name_dropdown]
        json_dict = get_all_json_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                _dataset_name == dataset_name
                and (_source_name.startswith(source_name) if source_name else False)
            )
        )
        if len(json_dict) == 0:
            gr.Warning("No image folder is found")
            return gr.update(choices=[])
        attack_names = set([key[1] for key in json_dict.keys()])
        attack_names.discard(None)
        return gr.update(
            choices=["Not Attacked"] + sorted(list(attack_names)), value="Not Attacked"
        )
    except Exception as e:
        raise gr.Error(e)


def find_folder_by_attack_name(
    folder_dataset_name_dropdown,
    folder_source_name_dropdown,
    folder_attack_name_dropdown,
):
    if not folder_attack_name_dropdown:
        return gr.update(choices=["None"], value="None")
    try:
        dataset_name = {v: k for k, v in DATASET_NAMES.items()}[
            folder_dataset_name_dropdown
        ]
        source_name = SOURCE_NAME_DICT_REVERSED[folder_source_name_dropdown]
        attack_name = (
            folder_attack_name_dropdown
            if not folder_attack_name_dropdown == "Not Attacked"
            else None
        )
        json_dict = get_all_json_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                _dataset_name == dataset_name
                and (_source_name.startswith(source_name) if source_name else False)
                and _attack_name == attack_name
            )
        )
        if len(json_dict) == 0:
            gr.Warning("No image folder is found")
            return gr.update(choices=[])
        attack_strengths = set([key[2] for key in json_dict.keys()])
        if None in attack_strengths:
            attack_strengths.discard(None)
            attack_strengths = ["N/A"] + sorted(list(attack_strengths))
        else:
            attack_strengths = sorted(list(attack_strengths))
        return gr.update(choices=attack_strengths, value=attack_strengths[0])
    except Exception as e:
        raise gr.Error(e)


def retrieve_folder_view(
    folder_dataset_name_dropdown,
    folder_source_name_dropdown,
    folder_attack_name_dropdown,
    folder_attack_strength_dropdown,
):
    try:
        dataset_name = {v: k for k, v in DATASET_NAMES.items()}[
            folder_dataset_name_dropdown
        ]
        source_name = SOURCE_NAME_DICT_REVERSED[folder_source_name_dropdown]
        attack_name = (
            folder_attack_name_dropdown
            if not folder_attack_name_dropdown == "Not Attacked"
            else None
        )
        attack_strength = (
            float(folder_attack_strength_dropdown)
            if not folder_attack_strength_dropdown == "N/A"
            else None
        )
        json_dict = get_all_json_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                _dataset_name == dataset_name
                and (_source_name.startswith(source_name) if _source_name else False)
                and _attack_name == attack_name
                and (
                    abs(_attack_strength - attack_strength) < 1e-5
                    if (_attack_strength is not None and attack_strength is not None)
                    else (_attack_strength is None and attack_strength is None)
                )
            )
        )
        json_paths = {
            result_type: [
                path for key, path in json_dict.items() if key[4] == result_type
            ]
            for result_type in ["status", "reverse", "decode", "metric"]
        }
        result_types_found = [
            result_type for result_type in json_paths.keys() if json_paths[result_type]
        ]

        # Experiment Progress
        if len(result_types_found) == 0:
            gr.Warning("No image folder is found")
            return (
                [0] * 4
                + [[]]
                + ["N/A"]
                * (
                    3
                    + len(EVALUATION_SETUPS) * len(PERFORMANCE_METRICS)
                    + len(QUALITY_METRICS)
                )
            )
        else:
            updates = [
                (int(get_progress_from_json(paths[0]) / 5) / 10) if paths else 0
                for paths in json_paths.values()
            ]

        # Image Examples
        if "status" not in result_types_found:
            gr.Warning("This image folder miss the status JSON")
            return (
                updates
                + [[]]
                + ["N/A"]
                * (
                    3
                    + len(EVALUATION_SETUPS) * len(PERFORMANCE_METRICS)
                    + len(QUALITY_METRICS)
                )
            )
        else:
            updates += [get_example_from_json(json_paths["status"][0])]

        # Evaluation Distances
        if "decode" not in result_types_found:
            gr.Warning("This image folder has not been decoded")
            return updates + ["N/A"] * (
                3
                + len(EVALUATION_SETUPS) * len(PERFORMANCE_METRICS)
                + len(QUALITY_METRICS)
            )
        else:
            distance_dict = {
                mode: get_distances_from_json(json_paths["decode"][0], mode)
                for mode in WATERMARK_METHODS.keys()
            }
            updates += [
                f"{np.mean(distance_dict[mode]):.4e}"
                if distance_dict[mode] is not None
                else "N/A"
                for mode in WATERMARK_METHODS.keys()
            ]

        # Evaluation Performance
        if source_name == "real" or attack_name is None or attack_strength is None:
            updates += ["N/A"] * (len(EVALUATION_SETUPS) * len(PERFORMANCE_METRICS))
        else:
            performances = [
                performance
                for mode in EVALUATION_SETUPS.keys()
                for performance in get_performance(
                    dataset_name, source_name, attack_name, attack_strength, mode
                ).values()
            ]
            updates += [
                f"{performance:.4f}" if performance is not None else "N/A"
                for performance in performances
            ]
        # Quality Metrics
        if attack_name is None or attack_strength is None:
            return updates + ["N/A"] * len(QUALITY_METRICS)
        else:
            quality_dict = get_quality(
                dataset_name, source_name, attack_name, attack_strength, None
            )
            updates += [
                f"{quality_dict[mode][0]:.4e} ± {quality_dict[mode][1]:.4e}"
                if quality_dict[mode] is not None
                else "N/A"
                for mode in QUALITY_METRICS.keys()
            ]
            return updates

    except Exception as e:
        raise gr.Error(e)


####################################################################################################
# Gradio UIs

with gr.Blocks() as app:
    with gr.Row():
        reload_button = gr.Button("Reload Results")
        clear_cache_button = gr.Button("Clear Cache")

        reload_button.click(reload_results, inputs=None, outputs=None)
        clear_cache_button.click(clear_cache, inputs=None, outputs=None)

    with gr.Tabs():
        with gr.Tab("Experiment Progress"):
            with gr.Row():
                with gr.Column(scale=30):
                    progress_dataset_name_dropdown = gr.Dropdown(
                        choices=["All", *list(DATASET_NAMES.values())],
                        value="All",
                        label="Dataset",
                    )
                with gr.Column(scale=30):
                    progress_source_name_dropdown = gr.Dropdown(
                        choices=[
                            "All",
                            "Not Watermarked",
                            *list(WATERMARK_METHODS.values()),
                        ],
                        value="All",
                        label="Source",
                    )
                progress_show_button = gr.Button("Show")
            progress_dataframe = gr.DataFrame(
                headers=[
                    "Dataset",
                    "Source",
                    "Attack",
                    "Strength",
                    "Generated",
                    "Reversed",
                    "Decoded",
                    "Measured",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                ],
                col_count=(8, "fixed"),
                type="pandas",
                interactive=False,
            )
            progress_show_button.click(
                show_experiment_progress,
                inputs=[progress_dataset_name_dropdown, progress_source_name_dropdown],
                outputs=progress_dataframe,
            )

        with gr.Tab("Attack Comparison"):
            with gr.Row():
                with gr.Column(scale=20):
                    compare_dataset_name_dropdown = gr.Dropdown(
                        choices=list(DATASET_NAMES.values()),
                        value=list(DATASET_NAMES.values())[0],
                        label="Dataset",
                    )
                with gr.Column(scale=20):
                    compare_source_name_dropdown = gr.Dropdown(
                        choices=list(WATERMARK_METHODS.values()),
                        value=list(WATERMARK_METHODS.values())[0],
                        label="Source",
                    )
                with gr.Column(scale=20):
                    compare_eval_setup_dropdown = gr.Dropdown(
                        choices=list(EVALUATION_SETUPS.values()),
                        value="Removal",
                        label="Evaluation Setup",
                    )
                compare_draw_button = gr.Button("Draw")

            with gr.Accordion("Result Table"):
                compare_result_dataframe = gr.DataFrame(
                    headers=[
                        "Attack",
                        "Strength",
                        *list(PERFORMANCE_METRICS.values()),
                        *list(QUALITY_METRICS.values()),
                    ],
                    datatype=["str", "number"]
                    + ["number"] * (len(PERFORMANCE_METRICS) + len(QUALITY_METRICS)),
                    col_count=(
                        2 + len(PERFORMANCE_METRICS) + len(QUALITY_METRICS),
                        "fixed",
                    ),
                    type="pandas",
                    interactive=False,
                    visible=True,
                )

            with gr.Accordion("Parallel Coordinates"):
                compare_performance_parallel_coordinates = gr.Plot(
                    label="Correlation of Performances", show_label=False
                )
                compare_quality_parallel_coordinates = gr.Plot(
                    label="Correlation of Qualities", show_label=False
                )

            with gr.Row(equal_height=True):
                with gr.Column(40):
                    compare_performance_mode_dropdown = gr.Dropdown(
                        choices=list(PERFORMANCE_METRICS.values()),
                        value=list(PERFORMANCE_METRICS.values())[3],
                        label="Performance Metric",
                    )
                with gr.Column(20):
                    compare_quality_mode_dropdown = gr.Dropdown(
                        choices=list(QUALITY_METRICS.values()),
                        value=list(QUALITY_METRICS.values())[0],
                        label="Quality Metric",
                    )

            with gr.Accordion("Plot Adjustments"):
                compare_show_strength_checkbox = gr.Checkbox(
                    value=False, label="Show Strength", scale=5
                )
                compare_line_width_slider = gr.Slider(
                    minimum=1, maximum=15, value=3, label="Line Width"
                )
                compare_marker_size_slider = gr.Slider(
                    minimum=1, maximum=30, value=9, label="Marker Size"
                )
                compare_tick_size_slider = gr.Slider(
                    minimum=1, maximum=30, value=10, label="Tick Size"
                )
                compare_legend_fontsize_slider = gr.Slider(
                    minimum=1, maximum=40, value=15, label="Legend Font Size"
                )
                compare_plot_height_slider = gr.Slider(
                    minimum=300, maximum=1500, value=800, label="Plot Height"
                )

            with gr.Accordion("2D Plots"):
                compare_2d_plot = gr.Plot(
                    label="Comparison of Attacks",
                    show_label=False,
                )

            compare_draw_button.click(
                aggregate_result_dataframe,
                inputs=[
                    compare_dataset_name_dropdown,
                    compare_source_name_dropdown,
                    compare_eval_setup_dropdown,
                ],
                outputs=[
                    compare_result_dataframe,
                    compare_performance_parallel_coordinates,
                    compare_quality_parallel_coordinates,
                ],
            )

            compare_result_dataframe.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )

            compare_performance_mode_dropdown.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )

            compare_quality_mode_dropdown.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )

            compare_show_strength_checkbox.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )
            compare_line_width_slider.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )
            compare_marker_size_slider.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )
            compare_tick_size_slider.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )
            compare_legend_fontsize_slider.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )
            compare_plot_height_slider.change(
                draw_2d_plot,
                inputs=[
                    compare_result_dataframe,
                    compare_performance_mode_dropdown,
                    compare_quality_mode_dropdown,
                    compare_show_strength_checkbox,
                    compare_line_width_slider,
                    compare_marker_size_slider,
                    compare_tick_size_slider,
                    compare_legend_fontsize_slider,
                    compare_plot_height_slider,
                ],
                outputs=[
                    compare_2d_plot,
                ],
            )

        with gr.Tab("Image Folder Viewer"):
            with gr.Row():
                with gr.Column(scale=30):
                    folder_dataset_name_dropdown = gr.Dropdown(
                        choices=list(DATASET_NAMES.values()),
                        value=list(DATASET_NAMES.values())[0],
                        label="Dataset",
                    )
                with gr.Column(scale=30):
                    folder_source_name_dropdown = gr.Dropdown(
                        choices=[
                            "Not Watermarked",
                            *list(WATERMARK_METHODS.values()),
                        ],
                        value="Not Watermarked",
                        label="Source",
                    )
                folder_find_button = gr.Button("Find")
            with gr.Row():
                with gr.Column(scale=40):
                    folder_attack_name_dropdown = gr.Dropdown(
                        choices=[], allow_custom_value=True, label="Attack"
                    )
                with gr.Column(scale=20):
                    folder_attack_strength_dropdown = gr.Dropdown(
                        choices=[], allow_custom_value=True, label="Stength"
                    )
            with gr.Accordion("Experiment Progress"):
                with gr.Row():
                    folder_generation_progress = gr.Slider(
                        label="Generation Progress (%)", interactive=False
                    )
                    folder_reverse_progress = gr.Slider(
                        label="Reverse Progress (%)", interactive=False
                    )
                with gr.Row():
                    folder_decode_progress = gr.Slider(
                        label="Decode Progress (%)", interactive=False
                    )
                    folder_metric_progress = gr.Slider(
                        label="Metric Progress (%)", interactive=False
                    )
            with gr.Accordion("Image Examples"):
                folder_example_gallery = gr.Gallery(
                    value=[],
                    show_label=False,
                    columns=4,
                    rows=1,
                    height=512,
                )
            with gr.Accordion("Evaluation Distances"):
                with gr.Row():
                    folder_eval_tree_ring_distance_number = gr.Textbox(
                        label="Tree-Ring Complex L1", interactive=False
                    )
                    folder_eval_stable_signature_distance_number = gr.Textbox(
                        label="Stable-Signature Bit Error Rate", interactive=False
                    )
                    folder_eval_stega_stamp_distance_number = gr.Textbox(
                        label="Stega-Stamp Bit Error Rate", interactive=False
                    )
            with gr.Accordion("Evaluation Performance"):
                folder_eval_performance_textbox_list = []
                for setup in EVALUATION_SETUPS.values():
                    with gr.Accordion(f"{setup} Setup"):
                        with gr.Row():
                            for metric in PERFORMANCE_METRICS.values():
                                folder_eval_performance_textbox_list.append(
                                    gr.Textbox(label=metric, interactive=False)
                                )
            with gr.Accordion("Quality Metrics"):
                folder_metric_textbox_list = []
                for i in range(0, len(QUALITY_METRICS), 4):
                    with gr.Row():
                        for metric in list(QUALITY_METRICS.values())[
                            i : min(i + 4, len(QUALITY_METRICS))
                        ]:
                            folder_metric_textbox_list.append(
                                gr.Textbox(label=metric, interactive=False)
                            )
            folder_find_button.click(
                find_folder_by_dataset_source,
                inputs=[
                    folder_dataset_name_dropdown,
                    folder_source_name_dropdown,
                ],
                outputs=folder_attack_name_dropdown,
            )
            folder_attack_name_dropdown.change(
                find_folder_by_attack_name,
                inputs=[
                    folder_dataset_name_dropdown,
                    folder_source_name_dropdown,
                    folder_attack_name_dropdown,
                ],
                outputs=folder_attack_strength_dropdown,
            )
            folder_attack_strength_dropdown.change(
                retrieve_folder_view,
                inputs=[
                    folder_dataset_name_dropdown,
                    folder_source_name_dropdown,
                    folder_attack_name_dropdown,
                    folder_attack_strength_dropdown,
                ],
                outputs=[
                    folder_generation_progress,
                    folder_reverse_progress,
                    folder_decode_progress,
                    folder_metric_progress,
                    folder_example_gallery,
                    folder_eval_tree_ring_distance_number,
                    folder_eval_stable_signature_distance_number,
                    folder_eval_stega_stamp_distance_number,
                    *folder_eval_performance_textbox_list,
                    *folder_metric_textbox_list,
                ],
            )


if __name__ == "__main__":
    if github_token is not None:
        # If github_token is provided, then it is on the HF space
        if os.path.isdir(result_dir):
            repo = Repo(result_dir)
            repo.remotes["origin"].set_url(
                f"https://{github_token}:x-oauth-basic@{repo_url}"
            )
            repo.remotes["origin"].pull()
        else:
            Repo.clone_from(
                f"https://{github_token}:x-oauth-basic@{repo_url}",
                result_dir,
                branch=branch_name,
            )
        app.launch(auth=("admin", os.getenv("LOGIN_PASSWORD")))
    else:
        # If github_token is not provided, then it is on the local machine
        assert os.path.isdir(result_dir)
        app.launch()
