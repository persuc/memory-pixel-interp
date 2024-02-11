cd /root
git clone https://github.com/persuck/memory-pixel-interp.git
cd memory-pixel-interp
mkdir /logs
conda install -y -c conda-forge x264=='1!164.3095' ffmpeg=6.1.1 | tee /logs/setup.log
echo "\nDone installing H264. Installing python packages...\n" | tee -a /logs/setup.log
pip install -r requirements.txt | tee -a /logs/setup.log
touch ~/.no_auto_tmux
conda init bash
exit 0