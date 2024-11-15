import os

os.environ["DEEPFACE_CACHE_DIR"] = r"E:\source\deepface\weights"
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import numpy as np

# from tests.test_verify import metrics

models_name = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",
               'Ensemble']


# 初始化
def init():
    print("==============================================================================>")
    print("pip下载命令：pip install deepface -i https://pypi.tuna.tsinghua.edu.cn/simple")
    print("修改了 folder_utils.py 的 get_deepface_home 函数")
    print("<==============================================================================")
    print()


# 展示图片
def show_img(img):
    if isinstance(img, Image.Image):
        img.show()
    else:
        img = Image.open(img)
        img.show()


# 给人脸添加画框
def picture_frame(src: str, res):
    _img = Image.open(src)  # 这里 src 是图像路径，正常加载
    # 访问 res 列表中的第一个字典
    x, y, w, h = res[0]['region']['x'], res[0]['region']['y'], res[0]['region']['w'], res[0]['region']['h']
    draw = ImageDraw.Draw(_img)
    draw.rectangle((x, y, x + w, y + h), outline="blue", width=3)
    print("emo:{}".format(res[0]["emotion"]))
    show_img(_img)


# 人脸识别模型 - 人脸验证 - Face Verification
def deepface_model_verify(img1: str, img2: str, model_name: str = models_name[1]):
    """
    给出两张图片比较两张图片是否为同一个人
    :param img1:
    :param img2:
    :param model_name:
    :return:
    """
    result = DeepFace.verify(img1_path=img1,
                             img2_path=img2,
                             model_name=model_name)
    return result


# 人脸识别模型 - 人脸识别 - Face recognition
def deepface_model_find(img: str, db: str, model_name=models_name[1]):
    """
    给一张图片以及一个文件夹，会在这个文件夹中找到和图片一样的人
    :param img:
    :param db:
    :param model_name:
    :return:
    """
    dfs = DeepFace.find(
        img_path=img,
        db_path=db
    )
    return dfs


# 人脸识别模型 - 人脸属性分析 - Face analyze
def deepface_model_analyze(img: str):
    """
    给定一张图片分析，年龄、性别、面部表情、种族
    :param img:
    :return:
    """
    result = DeepFace.analyze(img_path=img)
    return result


# 人脸识别模型 - 人脸检测 - Face extract
def deepface_model_extract(img: str):
    """
    给定一张图检测图中任务
    :param img:
    :return:
    """
    face_objs = DeepFace.extract_faces(
        img_path=img,
        detector_backend='opencv',
        align=True,
    )
    return face_objs


# 人脸识别模型 - 人脸特征提取 - Face represent
def deepface_model_represent(img: str):
    embedding_objs = DeepFace.represent(
        img_path=img
    )
    return embedding_objs


# 人脸识别模型 - 人脸实时分析 - Face stream
def deepface_model_stream(db: str, video: str):
    DeepFace.stream(db_path=db,
                    source=video)


# 打印结果
def printf(res, desc):
    print('-------------------------------->')
    print(desc)
    print("result type : ", type(res))
    print(res)
    print('<--------------------------------')
    print()


if __name__ == '__main__':
    init()

    # res1 = deepface_model_verify(img1='img/1-1.jpg', img2='img/1-2.jpg')
    # printf(res1, "deepface_model_verify")

    # dfs = deepface_model_find(img='img/1-1.jpg', db='img')
    # printf(dfs, 'deepface_model_find')

    # res2 = deepface_model_analyze(img='img/3-3.jpg')
    # print(type(res2))
    # printf(res2, 'deepface_model_analyze')
    # picture_frame('img/3-3.jpg', res2)

    # face_objs = deepface_model_extract('img/2-2.jpg')
    # printf(face_objs, 'deepface_model_extract')

    # res3 = deepface_model_represent('img/2-3.jpg')
    # printf(res3, 'deepface_model_represent')

    deepface_model_stream('img', 'video/3.mp4')
