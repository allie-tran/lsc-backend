set -e

conda env create -f environment.yml

pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ --no-cache-dir --use-deprecated=legacy-resolver




