#!/bin/bash
set -eE
source .env

trap 'echo "Error at line $LINENO: $BASH_COMMAND"' ERR

# Install dependencies via pip
pip install -r requirements.txt

# Check for Linux or macOS
if [[ "$(uname)" == "Linux" ]]; then
  sudo apt update
  sudo apt install -y build-essential libssl-dev curl
  sudo apt-get install -y npm
  if ! command -v nvm &> /dev/null; then
    curl --insecure -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
    source ~/.bashrc
  fi
elif [[ "$(uname)" == "Darwin" ]]; then
  # macOS instructions for nvm installation
  if ! command -v nvm &> /dev/null; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
    source ~/.zshrc
  fi
fi

# Install nvm and typescript
nvm install 16.0.0
npm install -g typescript

# Go into app folder and run npm install
cd app
npm install

# Go back to the main directory and run download.py
cd ..
python3 ./datasets/download.py --max_train 102400 --max_test 1024