huggingface-cli login --token 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 --add-to-git-credential
huggingface-cli upload mcding/WMBench README.md README.md --repo-type=space
huggingface-cli upload mcding/WMBench requirements_space.txt requirements.txt --repo-type=space
huggingface-cli upload mcding/WMBench app.py app.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev ./dev --repo-type=space
