import os
import io
import gradio as gr
import subprocess
import paramiko
from git import Repo, Git
from git.exc import GitCommandError
from dotenv import load_dotenv

load_dotenv()
result_dir = os.environ.get("RESULT_DIR")
login_password = os.getenv("LOGIN_PASSWORD")
github_token = os.environ.get("GITHUB_TOKEN")
repo_url = os.environ.get("REPO_URL")
branch_name = os.environ.get("BRANCH_NAME")


def count_json_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                count += 1
    return count


def reload_results():
    try:
        if github_token is not None:
            # If github_token is provided, then it is on the HF space and we can clone the repository
            repo = Repo(result_dir)
            repo.remotes["origin"].set_url(
                f"https://{github_token}:x-oauth-basic@{repo_url}"
            )
            repo.remotes["origin"].pull()
        else:
            # If github_token is not provided, then it is on the local machine
            Repo(result_dir).git.pull()
        return f"Reloaded successfully, {count_json_files(result_dir)} files found."
    except GitCommandError as e:
        return f"Reloaded failed: {e}"


def test_func(name):
    return "Hello " + name + "!!"


# Gradio UIs
with gr.Blocks() as app:
    with gr.Row():
        reload_button = gr.Button("Reload")
        reload_status_textbox = gr.Textbox(label="Status", placeholder="")
        reload_button.click(reload_results, inputs=None, outputs=reload_status_textbox)
    with gr.Tabs():
        with gr.Tab("Test"):
            with gr.Row():
                input_textbox = gr.Textbox(label="Input", placeholder="Type here...")
                output_textbox = gr.Textbox(label="Output", placeholder="")
                generate_button = gr.Button("Generate")
                generate_button.click(
                    test_func, inputs=input_textbox, outputs=output_textbox
                )


if __name__ == "__main__":
    if github_token is not None:
        # If github_token is provided, then it is on the HF space and we can clone the repository
        assert not os.path.isdir(result_dir)
        Repo.clone_from(
            f"https://{github_token}:x-oauth-basic@{repo_url}",
            result_dir,
            branch=branch_name,
        )
    else:
        # If github_token is not provided, then it is on the local machine
        assert os.path.isdir(result_dir)
    app.launch(auth=("admin", login_password))
