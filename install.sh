apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output anaconda.sh
bash anaconda.sh
source ~/.bashrc
conda env create --name first --file=test.yml
pip install --upgrade pip \
            pip install matplotlib \
            seaborn\
            scikit-learn \
            tensorflow \
            opencv-python

