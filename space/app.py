import os
import gradio as gr
from git import Repo
from result_utils import get_all_json_paths
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
# Tab: JSON Viewer


def find_json_attack_names(
    json_dataset_name_dropdown,
    json_source_name_dropdown,
    json_result_type_dropdown,
):
    if (
        not json_dataset_name_dropdown
        or not json_source_name_dropdown
        or not json_result_type_dropdown
    ):
        gr.Info("Please select dataset, source, and result type first")
        return gr.update(choices=[])
    try:
        target_dataset_name = (
            json_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        target_source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[json_source_name_dropdown]
        target_result_type = json_result_type_dropdown.lower()
        json_dict = get_all_json_paths(
            lambda dataset_name, attack_name, attack_strength, source_name, result_type: (
                dataset_name == target_dataset_name
                and (
                    source_name.startswith(target_source_name) if source_name else False
                )
                and result_type == target_result_type
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


def find_json_attack_strengths(
    json_dataset_name_dropdown,
    json_source_name_dropdown,
    json_result_type_dropdown,
    json_attack_name_dropdown,
):
    if not json_attack_name_dropdown:
        return gr.update(choices=[None], value=None)
    try:
        target_dataset_name = (
            json_dataset_name_dropdown.replace("-", "").replace(" ", "").lower()
        )
        target_source_name = {
            "Real": "real",
            "Tree-Ring": "tree_ring",
            "Stable-Signature": "stable_sig",
            "Stega-Stamp": "stegastamp",
        }[json_source_name_dropdown]
        target_result_type = json_result_type_dropdown.lower()
        target_attack_name = json_attack_name_dropdown
        json_dict = get_all_json_paths(
            lambda dataset_name, attack_name, attack_strength, source_name, result_type: (
                dataset_name == target_dataset_name
                and (
                    source_name.startswith(target_source_name) if source_name else False
                )
                and result_type == target_result_type
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


def retrieve_json_file(
    json_dataset_name_dropdown,
    json_source_name_dropdown,
    json_result_type_dropdown,
    json_attack_name_dropdown,
    json_attack_strength_dropdown,
):
    pass


# Gradio UIs
with gr.Blocks() as app:
    with gr.Row():
        reload_button = gr.Button("Reload")
        reload_status_textbox = gr.Textbox(label="Result Status", placeholder="")
        reload_button.click(reload_results, inputs=None, outputs=reload_status_textbox)
    with gr.Tabs():
        with gr.Tab("Experiment Status"):
            with gr.Row():
                with gr.Column(scale=20):
                    status_dataset_name_dropdown = gr.Dropdown(
                        choices=["All", "DiffusionDB", "MS-COCO", "DALL-E 3"],
                        label="Dataset",
                    )
                with gr.Column(scale=50):
                    status_option_checkbox_group = gr.CheckboxGroup(
                        ["Show Individual Strength", ""],
                        label="Options",
                    )
                with gr.Column(scale=30):
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
        with gr.Tab("JSON Viewer"):
            with gr.Row():
                json_dataset_name_dropdown = gr.Dropdown(
                    choices=["DiffusionDB", "MS-COCO", "DALL-E 3"],
                    label="Dataset",
                )
                json_source_name_dropdown = gr.Dropdown(
                    choices=["Real", "Tree-Ring", "Stable-Signature", "Stega-Stamp"],
                    label="Source",
                )
                json_result_type_dropdown = gr.Dropdown(
                    choices=["Status", "Reverse", "Decode", "Metric"],
                    label="Result Type",
                )
                json_filter_button = gr.Button("Filter")
            with gr.Row():
                with gr.Column(scale=50):
                    json_attack_name_dropdown = gr.Dropdown(
                        choices=[], allow_custom_value=True, label="Attack"
                    )
                with gr.Column(scale=25):
                    json_attack_strength_dropdown = gr.Dropdown(
                        choices=[], allow_custom_value=True, label="Stength"
                    )
            json_json_display = gr.Json(show_label=False)
            json_filter_button.click(
                find_json_attack_names,
                inputs=[
                    json_dataset_name_dropdown,
                    json_source_name_dropdown,
                    json_result_type_dropdown,
                ],
                outputs=json_attack_name_dropdown,
            )
            json_attack_name_dropdown.change(
                find_json_attack_strengths,
                inputs=[
                    json_dataset_name_dropdown,
                    json_source_name_dropdown,
                    json_result_type_dropdown,
                    json_attack_name_dropdown,
                ],
                outputs=json_attack_strength_dropdown,
            )
            json_attack_strength_dropdown.change(
                retrieve_json_file,
                inputs=[
                    json_dataset_name_dropdown,
                    json_source_name_dropdown,
                    json_result_type_dropdown,
                    json_attack_name_dropdown,
                    json_attack_strength_dropdown,
                ],
                outputs=json_json_display,
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
        app.launch(debug=True)
