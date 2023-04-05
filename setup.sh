set -e 
pip install -r requirements.txt
source .env

# sudo apt update
# sudo apt install build-essential libssl-dev curl
# apt-get install -y npm
# curl --insecure -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
# source ~/.bashrc
# nvm install 16.0.0
# npm install -g typescript

# cd app && npm install && cd ..
python3 ./datasets/download.py