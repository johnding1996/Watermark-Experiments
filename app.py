import os
import numpy as np
import gradio as gr
from git import Repo
from dev import *
from dotenv import load_dotenv

load_dotenv()
result_dir = os.environ.get("RESULT_DIR")
login_password = os.getenv("LOGIN_PASSWORD")
github_token = os.environ.get("GITHUB_TOKEN")
repo_url = os.environ.get("REPO_URL")
branch_name = os.environ.get("BRANCH_NAME")


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
        return f"Reload results successful, {len(get_all_json_paths())} JSONs found, last updated at {repo.head.commit.committed_datetime.strftime('%b %d, %H:%M:%S')}"
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
            (progress_dataset_name_dropdown.replace("-", "").replace(" ", "").lower())
            if progress_dataset_name_dropdown != "All"
            else None
        )
        source_name = (
            {
                "Real": "real",
                "Tree-Ring": "tree_ring",
                "Stable-Signature": "stable_sig",
                "Stega-Stamp": "stegastamp",
            }[progress_source_name_dropdown]
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
        for key, json_path in progress.tqdm(json_dict.items()):
            progress_dict[(key[0], key[3], key[1], key[2])][
                ["status", "reverse", "decode", "metric"].index(key[4])
            ] = get_progress_from_json(json_path)

        return gr.update(
            value=style_progress_dataframe(
                [[*key, *progress_dict[key]] for key in progress_dict.keys()]
            ),
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
        dataset_name = (
            folder_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[folder_source_name_dropdown]
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
        return gr.update(choices=["None"] + list(sorted(attack_names)), value="None")
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
        dataset_name = (
            folder_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[folder_source_name_dropdown]
        attack_name = (
            folder_attack_name_dropdown
            if not folder_attack_name_dropdown == "None"
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
        attack_strengths = list(sorted(set([key[2] for key in json_dict.keys()])))
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
        dataset_name = (
            folder_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[folder_source_name_dropdown]
        attack_name = (
            folder_attack_name_dropdown
            if not folder_attack_name_dropdown == "None"
            else None
        )
        attack_strength = (
            float(folder_attack_strength_dropdown)
            if not (
                folder_attack_strength_dropdown is None
                or folder_attack_strength_dropdown == "None"
            )
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
            return [0] * 4 + [[]] + ["N/A"] * 12
        else:
            updates = [
                gr.update(value=(int(get_progress_from_json(paths[0]) / 5) / 10))
                if paths
                else gr.update(value=0)
                for paths in json_paths.values()
            ]

        # Image Examples
        if "status" not in result_types_found:
            gr.Warning("This image folder miss the status JSON")
            return updates + [[]] + ["N/A"] * 12
        else:
            updates += [gr.update(value=get_example_from_json(json_paths["status"][0]))]

        # Evaluation Distances
        if "decode" not in result_types_found:
            gr.Warning("This image folder has not been decoded")
            return updates + ["N/A"] * 12
        else:
            distance_dict = {
                mode: get_distances_from_json(json_paths["decode"][0], mode)
                for mode in WATERMARK_METHODS
            }
            updates += [
                gr.update(
                    value=np.mean(distance_dict[mode])
                    if distance_dict[mode] is not None
                    else "N/A"
                )
                for mode in WATERMARK_METHODS
            ]

        # Evaluation Performance
        if source_name == "real" or attack_name is None or attack_strength is None:
            return updates + ["N/A"] * 9
        else:
            performances = [
                performance
                for mode in EVALUATION_SETUPS
                for performance in get_performance(
                    dataset_name, source_name, attack_name, attack_strength, mode
                )
            ]
            updates += [
                gr.update(value=performance if performance is not None else "N/A")
                for performance in performances
            ]
            return updates
        # Quality Metrics
        if attack_name is None or attack_strength is None:
            pass

    except Exception as e:
        raise gr.Error(e)


####################################################################################################
# Gradio UIs

with gr.Blocks() as app:
    with gr.Row():
        reload_button = gr.Button("Reload Results")
        reload_status_textbox = gr.Textbox(label="Result Status", placeholder="")
        reload_button.click(reload_results, inputs=None, outputs=reload_status_textbox)
    with gr.Tabs():
        with gr.Tab("Experiment Progress"):
            with gr.Row():
                with gr.Column(scale=30):
                    progress_dataset_name_dropdown = gr.Dropdown(
                        choices=["All", "DiffusionDB", "MS-COCO", "DALL-E 3"],
                        value="All",
                        label="Dataset",
                    )
                with gr.Column(scale=30):
                    progress_source_name_dropdown = gr.Dropdown(
                        choices=[
                            "All",
                            "Real",
                            "Tree-Ring",
                            "Stable-Signature",
                            "Stega-Stamp",
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
        with gr.Tab("Image Folder Viewer"):
            with gr.Row():
                with gr.Column(scale=30):
                    folder_dataset_name_dropdown = gr.Dropdown(
                        choices=["DiffusionDB", "MS-COCO", "DALL-E 3"],
                        label="Dataset",
                    )
                with gr.Column(scale=30):
                    folder_source_name_dropdown = gr.Dropdown(
                        choices=[
                            "Real",
                            "Tree-Ring",
                            "Stable-Signature",
                            "Stega-Stamp",
                        ],
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
                with gr.Accordion("Combined Setup"):
                    with gr.Row():
                        folder_eval_combined_acc_number = gr.Textbox(
                            label="Mean Accuracy", interactive=False
                        )
                        folder_eval_combined_auc_number = gr.Textbox(
                            label="AUC Score", interactive=False
                        )
                        folder_eval_combined_low_number = gr.Textbox(
                            label="TPR@0.1%FPR", interactive=False
                        )
                with gr.Accordion("Removal Setup"):
                    with gr.Row():
                        folder_eval_removal_acc_number = gr.Textbox(
                            label="Mean Accuracy", interactive=False
                        )
                        folder_eval_removal_auc_number = gr.Textbox(
                            label="AUC Score", interactive=False
                        )
                        folder_eval_removal_low_number = gr.Textbox(
                            label="TPR@0.1%FPR", interactive=False
                        )
                with gr.Accordion("Spoofing Setup"):
                    with gr.Row():
                        folder_eval_spoofing_acc_number = gr.Textbox(
                            label="Mean Accuracy", interactive=False
                        )
                        folder_eval_spoofing_auc_number = gr.Textbox(
                            label="AUC Score", interactive=False
                        )
                        folder_eval_spoofing_low_number = gr.Textbox(
                            label="TPR@0.1%FPR", interactive=False
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
                    folder_eval_combined_acc_number,
                    folder_eval_combined_auc_number,
                    folder_eval_combined_low_number,
                    folder_eval_removal_acc_number,
                    folder_eval_removal_auc_number,
                    folder_eval_removal_low_number,
                    folder_eval_spoofing_acc_number,
                    folder_eval_spoofing_auc_number,
                    folder_eval_spoofing_low_number,
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
        app.launch(auth=("admin", login_password))
    else:
        # If github_token is not provided, then it is on the local machine
        assert os.path.isdir(result_dir)
        app.launch()
