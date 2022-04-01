call  C:\Users\user\anaconda3\Scripts\activate.bat
cd C:\steel_detect_n
call conda env create  -f yolo.yml 
call conda activate yolo_env
call pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
call pip3 install flask
pause

