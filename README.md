
# 1 . Config파일 만드셈
 ==> python module/ConfigModule.py하면 
 
 config파일에 config_manual.yaml이랑 config_trainer.yaml이 만들어짐
 
 config.trainer에서 데이터 경로 수정해놓으면 좋음 (ConfigModule.py에서 파일 경로 수정해놓고 실행시키면, default에 있는대로 yaml파일 만들어짐)
 
# 2 . run.sh파일에 있는거 돌리셈
 model_type 수정하고 save_filename 

# @. data_processing.py 돌리는 법
 1. 데이터 작업할 경로를 하나 만들음 -> /data
 2. /data폴더에 data_processing.py파일을 위치시킴
 3. /data/original폴더에 준메이가 작업한 .json파일을 모두 가져오셈
 4. 그리고 python /data/data_processing.py
 5. 그러면 data/processed에 파일 생성됨
