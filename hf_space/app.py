import os
import io
import gradio as gr
import subprocess
import paramiko
from git import Repo, Git
from git.exc import GitCommandError
from dotenv import load_dotenv

load_dotenv()
login_password = os.getenv("LOGIN_PASSWORD")
ssh_private_key = os.environ.get("SSH_PRIVATE_KEY")
result_dir = os.environ.get("RESULT_DIR")
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
        with Git().custom_environment(GIT_SSH="./gitssh.py"):
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
    # Clone or pull the repository
    if ssh_private_key is not None:
        # If ssh_key is provided, then it is on the HF space and we can clone the repository
        assert not os.path.isdir(result_dir)
        # Dump the ssh_key to the file
        # with open("id_rsa", "w") as file:
        #     file.write(ssh_private_key)
        # os.chmod("id_rsa", 0o600)

        # Clone the repository if it does not exist
        with Git().custom_environment(GIT_SSH="./gitssh.py"):
            Repo.clone_from(repo_url, result_dir, branch=branch_name)
    else:
        # If ssh_key is not provided, then it is on the local machine and we can pull the repository
        assert os.path.isdir(result_dir)
    app.launch(auth=("admin", login_password))
