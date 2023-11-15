import os
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


def summarize_status(dataset_dropdown, status_checkbox_group):
    return []


####################################################################################################
# Tab: Image Folder Viewer


def find_folder_attack_names(
    folder_dataset_name_dropdown,
    folder_source_name_dropdown,
):
    if not folder_dataset_name_dropdown or not folder_source_name_dropdown:
        gr.Info("Please select dataset and source first")
        return gr.update(choices=[])
    try:
        target_dataset_name = (
            folder_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        target_source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[folder_source_name_dropdown]
        json_dict = get_all_json_paths(
            lambda dataset_name, attack_name, attack_strength, source_name, result_type: (
                dataset_name == target_dataset_name
                and (
                    source_name.startswith(target_source_name) if source_name else False
                )
            )
        )
        if len(json_dict) == 0:
            gr.Warning("No JSON file is found")
            return gr.update(choices=[])
        attack_names = set([key[1] for key in json_dict.keys()])
        attack_names.discard(None)
        return gr.update(choices=[None] + list(sorted(attack_names)), value=None)
    except Exception as e:
        raise gr.Error(e)


def find_folder_attack_strengths(
    folder_dataset_name_dropdown,
    folder_source_name_dropdown,
    folder_attack_name_dropdown,
):
    if not folder_attack_name_dropdown:
        return gr.update(choices=[None], value=None)
    try:
        target_dataset_name = (
            folder_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        target_source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[folder_source_name_dropdown]
        target_attack_name = folder_attack_name_dropdown
        json_dict = get_all_json_paths(
            lambda dataset_name, attack_name, attack_strength, source_name, result_type: (
                dataset_name == target_dataset_name
                and (
                    source_name.startswith(target_source_name) if source_name else False
                )
                and attack_name == target_attack_name
            )
        )
        if len(json_dict) == 0:
            gr.Warning("No JSON file is found")
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
        target_dataset_name = (
            folder_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        target_source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[folder_source_name_dropdown]
        target_attack_name = folder_attack_name_dropdown
        target_attack_strength = (
            float(folder_attack_strength_dropdown)
            if folder_attack_strength_dropdown
            else None
        )
        json_dict = get_all_json_paths(
            lambda dataset_name, attack_name, attack_strength, source_name, result_type: (
                dataset_name == target_dataset_name
                and (
                    source_name.startswith(target_source_name) if source_name else False
                )
                and attack_name == target_attack_name
                and match_attack_strengths(attack_strength, target_attack_strength)
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
        if len(result_types_found) == 0:
            gr.Warning("No JSON file is found")
        else:
            gr.Info(f"Found JSON types: {result_types_found}")
        return [
            gr.update(value=get_progress_from_json(paths[0]) / 5000 * 100)
            if paths
            else gr.update(value=0)
            for paths in json_paths.values()
        ]
    except Exception as e:
        raise gr.Error(e)


# Gradio UIs
with gr.Blocks() as app:
    with gr.Row():
        reload_button = gr.Button("Reload")
        reload_status_textbox = gr.Textbox(label="Result Status", placeholder="")
        reload_button.click(reload_results, inputs=None, outputs=reload_status_textbox)
    with gr.Tabs():
        with gr.Tab("Experiment Status"):
            with gr.Row():
                with gr.Column(scale=10):
                    status_dataset_name_dropdown = gr.Dropdown(
                        choices=["All", "DiffusionDB", "MS-COCO", "DALL-E 3"],
                        label="Dataset",
                    )
                with gr.Column(scale=20):
                    status_option_checkbox_group = gr.CheckboxGroup(
                        ["Show Individual Strength", ""],
                        label="Options",
                    )
                status_generate_button = gr.Button("Show")
            status_dataframe = gr.DataFrame(
                headers=[
                    "Dataset",
                    "Attack",
                    "# of Strengths",
                    "Source",
                    "Attack Progress",
                    "Reverse Progress",
                    "Decode Progress",
                    "Metric Progress",
                ],
                type="array",
                interactive=False,
            )
            status_generate_button.click(
                summarize_status,
                inputs=[status_dataset_name_dropdown, status_option_checkbox_group],
                outputs=status_dataframe,
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
            with gr.Group():
                folder_generation_progress = gr.Slider(
                    label="Generation Progress (%)", interactive=False
                )
                folder_reverse_progress = gr.Slider(
                    label="Reverse Progress (%)", interactive=False
                )
                folder_decode_progress = gr.Slider(
                    label="Decode Progress (%)", interactive=False
                )
                folder_metric_progress = gr.Slider(
                    label="Metric Progress (%)", interactive=False
                )
            folder_find_button.click(
                find_folder_attack_names,
                inputs=[
                    folder_dataset_name_dropdown,
                    folder_source_name_dropdown,
                ],
                outputs=folder_attack_name_dropdown,
            )
            folder_attack_name_dropdown.change(
                find_folder_attack_strengths,
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
