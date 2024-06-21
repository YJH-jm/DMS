import os
import shutil

light = ['무광원', '정면광원', '좌측광원', '우측광원', '후면광원']  # 3
location = ['계기판', '네비게이션', '룸미러', '썬바이저', '정면', '좌사이드미러', '우사이드미러'] # 4
status = ['정상주시', '졸음재현', '하품재현', '통화재현'] #5

label_dir = "./dataset/졸음운전 예방을 위한 운전자 상태 정보 영상/Validation/[라벨]bbox(통제환경)/045_G1"
img_dir = "./dataset/졸음운전 예방을 위한 운전자 상태 정보 영상/Validation/[원천]bbox(통제환경)/045_G1"
save_path = "./dataset/sample_data"

if not os.path.isdir(save_path):
    os.makedirs(os.path.join(save_path, 'img'))
    os.makedirs(os.path.join(save_path, 'label'))


label_list = os.listdir(label_dir)
img_list = os.listdir(img_dir)

for label_file, img_file in zip(label_list, img_list):
    name, ext = img_file.split(".")
    
    p = name.split('_')
    p[3] = str(light.index(p[3]))
    p[4] = str(location.index(p[4]))
    p[5] = str(status.index(p[5]))

    new_file = "_".join(p) + "." +  ext

    # shutil.copyfile(os.path.join(img_dir, img_file),os.path.join(save_path,'img',new_file))