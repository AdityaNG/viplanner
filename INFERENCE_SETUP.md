

```bash
# conda create -n viplanner python=3.9 -y
conda create -n viplanner python=3.8 -y
conda activate viplanner
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

pip install numpy==1.26.4 pillow opencv-python-headless PyYAML transformers accelerate gdown six yt-dlp scipy
pip install instructor anthropic openai
pip install urllib3==1.26.20
mkdir ckpts/
gdown https://drive.google.com/uc?id=1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc -O ckpts/model.pt
gdown https://drive.google.com/uc?id=1r1yhNQAJnjpn9-xpAQWGaQedwma5zokr -O ckpts/model.yaml

# start: 00:05, end: 14:05
# quality: 480p
yt-dlp -o assets/kora_walk.mp4 --download-sections "*00:05-14:05" --format "bv*[height=480]+ba" https://www.youtube.com/watch?v=mgVDoDcrjs8

yt-dlp -o assets/kora_walk.mini.mp4 --download-sections "*00:05-1:05" --format "bv*[height=480]+ba" https://www.youtube.com/watch?v=mgVDoDcrjs8

python3 main.py --video assets/kora_walk.mp4
```