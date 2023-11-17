huggingface-cli login --token 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 --add-to-git-credential
huggingface-cli upload mcding/WMBench ./README.md ./README.md --repo-type=space
huggingface-cli upload mcding/WMBench ./requirements_space.txt ./requirements.txt --repo-type=space
huggingface-cli upload mcding/WMBench ./app.py ./app.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/__init__.py ./dev/__init__.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/aggregate.py ./dev/aggregate.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/constants.py ./dev/constants.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/eval.py ./dev/eval.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/find.py ./dev/find.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/io.py ./dev/io.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/parse.py ./dev/parse.py --repo-type=space
huggingface-cli upload mcding/WMBench ./dev/plot.py ./dev/plot.py --repo-type=space
