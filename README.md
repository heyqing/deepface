# Deepface

deepface是【Github】上的一个开源人脸识别（Face Recognition）项目，项目包含了人脸识别、人脸属性分析等多个功能

代码地址为：[DeepFace](https://github.com/serengil/deepface)

```
https://github.com/serengil/deepface
```

模型下载地址：[Releases](https://github.com/serengil/deepface/releases)

```
https://github.com/serengil/deepface/releases
```

python安装方式(对python版本有限制，最好使用3.8 < version < 3.13的，使用3.11就可以) [python 3.11 下载](https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe)

```bash
pip install deepface -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 基础功能接口

- `DeepFace.verify`:人脸验证
- `DeepFace.find`:人脸识别
- `DeepFace.analyze`:人脸属性分析
- `DeepFace.extract_faces`:人脸检测和对齐
- `DeepFace.represent`:人脸特征提取
- `DeepFace.stream`:人脸实时分析

当然这个项目的功能很多，很抱歉我不能全都提到，下面会对这几个接口进行一一讲解，在此之前你需要准备一些基础数据 => 人物照片，视频等【我的放文章最后】

当然还需这几个python库作为辅助，如果你还未安装可使用下面命令进行安装

**matplotlib**：数据可视化库，用于创建静态、动态、交互式的图形和可视化。

```bash
pip install matplotlib
```

**PIL **:Python Imaging Library，图像处理的库，如图像剪裁、缩放、旋转、转换颜色空间、过滤等。

```bash
pip install Pillow
```

**cv2**:Open Source Computer Vision Library,开源计算机视觉库,主要用于图像处理、视频分析、机器视觉以及图像识别等.。

```bash
pip install opencv-python
```

如果你还需要一些额外的功能，如视频 I/O 或 OpenCV 的扩展功能，可以安装 `opencv-python-headless`：

```bash
pip install opencv-python-headless
```

**numpy **:Numerical Python,用于科学计算的 Python 库,提供了高效的数组对象,以及用于处理数组的各种功能和数学运算等。

```bash
pip install numpy
```

## 人脸验证

> 原文
>
> **Face Verification** - [`Demo`](https://youtu.be/KRCvkNCOphE)
>
> This function verifies face pairs as same person or different persons. It expects exact image paths as inputs. Passing numpy or base64 encoded images is also welcome. Then, it is going to return a dictionary and you should check just its verified key.
>
> 翻译过来就是
>
> 此功能可验证人脸对是同一个人还是不同人。它期望精确的图像路径作为输入。也欢迎传递 numpy 或 base64 编码图像。然后，它将返回一个字典，您应该仅检查其经过验证的密钥。

意思就是给两张图，判断这两张图是不是同一个人

这里是简单的代码示例

```python
# 人脸识别模型 - 人脸验证 - Face Verification
# img1 = 1-1.jpg , img2 = 1-2.jpg
def deepface_model_verify(img1: str, img2: str, model_name: str = 'VGG-Face'):
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
```

将运行结果result，打印得到：

```json
{
    'verified': True, 
    'distance': 0.23718864776725845, 
    'threshold': 0.4,
    'model': 'Facenet', 
    'detector_backend': 'opencv',
    'similarity_metric': 'cosine', 
    'facial_areas': 
    {
        'img1': 
            {
                'x': 339, 
                'y': 218,
                'w': 768,
                'h': 768,
                'left_eye': (850, 524),
                'right_eye': (571, 517)
            },
    	'img2': 
            {
                'x': 524,
                'y': 201,
                'w': 491,
                'h': 491,
                'left_eye': (858, 388),
                'right_eye': (663, 390)
            }
    },
    'time': 5.01
}
```

verified为True说明两张图片为同一个人，事实也是如此

> **针对模型之间的比较在下面的 - 人脸识别模型比较 - 提及**

**请求参数说明**

- `img1_path (str 或 np.ndarray 或 List[float])`:第一幅图像的路径。
- `img2_path (str 或 np.ndarray 或 List[float])`:第二幅图像的路径。(图像接受字符串形式的精确图像路径、numpy 数组 (BGR)、base64 编码图像或预先计算的嵌入)
- `model_name (str)`：人脸识别模型。
  - 选项：VGG-Face、Facenet、Facenet512、OpenFace、DeepFace、DeepID、Dlib、ArcFace、SFace 和 GhostFaceNet（默认为 VGG-Face）。
  - 模型加载需要h5文件，默认GitHub下载，若下载不下可点击 [.h5文件下载](https://pan.quark.cn/s/28e3ef150cde)
  - 下载后放在`C:\Users\你的用户文件夹\.deepface\weights`中
  - 也可自定义文件路径：修改 `folder_utils.py` 的 `get_deepface_home` 函数
- `detector_backend (str)`:人脸检测器。
  - 选项：'opencv'、'retinaface'、'mtcnn'、'ssd'、'dlib'、'mediapipe'、'yolov8'、'centerface' 或 'skip'（默认为 opencv）。
- `distance_metric (str)`:用于测量相似度的度量。
  - 选项：'cosine'、'euclidean'、'euclidean_l2'（默认为‘cosine’）。
- `force_detection (bool)`:如果在图像中未检测到面部，则引发异常。
  - 设置为 False 以避免低分辨率图像的异常（默认为 True）。
- `align(bool)`:用于启用面部对齐的标志（默认为 True）。
- `expand_percentage（int）`:以百分比扩大检测到的面部区域（默认为 0）。
- `normalization (str)`:在将输入图像输入模型之前对其进行规范化。
  - 选项：base、raw、Facenet、Facenet2018、VGGFace、VGGFace2、ArcFace（默认为 base）
- `silent (bool)`:抑制或允许某些日志消息，以实现更安静的分析过程（默认为 False）。
- `Threshold (float)`:指定阈值以确定一对代表同一个人还是不同的人。
  - 此阈值用于比较距离。如果未设置，将根据指定的模型名称和距离度量应用默认的预调整阈值（默认为 None）。
- `anti_spoofing (bool)`:启用反欺骗的标志（默认为 False）。

**返回参数说明**

字典类型，包含

- `verified:bool`:表示图像代表同一个人 (True)还是不同的人 (False)。
- `distance:float`:面部向量之间的距离测量值。距离越小，相似度越高。
- `threshold:float`:用于验证的最大阈值。如果距离低于此阈值，则认为图像匹配。
- `model:str`:所选的面部识别模型。
- `distance_metric:str`:所选的用于测量距离的相似度度量。
- `facial_areas:dict`:两幅图像中面部的矩形感兴趣区域。
  - `img1:{x：int，y：int，w：int，h：int}`:第一幅图像的感兴趣区域。
  - `img2:{x：int，y：int，w：int，h：int}`:第二幅图像的感兴趣区域。
- `time:float`:验证过程所用时间（以秒为单位）。

## 人脸识别

> [Face recognition](https://sefiks.com/2020/05/25/large-scale-face-recognition-for-deep-learning/) requires applying face verification many times. Herein, deepface has an out-of-the-box find function to handle this action. It's going to look for the identity of input image in the database path and it will return list of pandas data frame as output. Meanwhile, facial embeddings of the facial database are stored in a pickle file to be searched faster in next time. Result is going to be the size of faces appearing in the source image. Besides, target images in the database can have many faces as well.
>
> 人脸识别需要多次进行人脸验证。在这里，deepface 有一个开箱即用的查找功能来处理此操作。它将在数据库路径中查找输入图像的标识，并将返回 pandas 数据帧列表作为输出。同时，面部数据库的面部嵌入存储在pickle文件中，以便下次更快地搜索。结果将是源图像中出现的面部尺寸。此外，数据库中的目标图像也可以有多个面孔。

即给定一个人脸和一个数据集，接口将会在这个数据集中找到与目标人脸相似的数据

简单的代码示例

```python
# 人脸识别模型 - 人脸识别 - Face recognition
# img = 1-1.jpg , db = ./img/ 
def deepface_model_find(img: str, db: str, model_name='VGG-Face'):
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
```

将返回的dfs打印得到：

```
[      identity                                      hash  ...  threshold  distance
0  img\1-1.jpg  7281f3c678041b24471ba0baea4f57e0bac7e5cf  ...       0.68  0.000000
1  img\1-2.jpg  4d8cdc50d15e6c56060ecb57ebbe01bc23dc7d90  ...       0.68  0.412802
2  img\1-3.jpg  d6e080b2bda60541cf19e1d5abdf5be464421031  ...       0.68  0.434774

[3 rows x 12 columns]]
```

结果显示1-1.jpg,1-2.jpg,1-3.jpg三张与目标文件（1-1.jpg）中的人物相似，事实却是如此

**请求参数说明**

- `img_path (str 或 np.ndarray)`:图像的准确路径、BGR 格式的 numpy 数组或 base64 编码图像。如果源图像包含多个人脸，结果将包含每个检测到的人脸的信息。
- `db_path (str)`:包含图像文件的文件夹的路径。
- `model_name (str)`:人脸识别模型。,选项不变
- `distance_metric (str)`:用于测量相似度的指标。不变
- `force_detection (bool)`:如果图像中未检测到人脸，则引发异常。设置为 False 以避免低分辨率图像的异常（默认为 True）。
- `detector_backend (str)`:人脸检测器后端。不变
- `align (bool)`:根据眼睛位置执行对齐（默认为 True）。
- `expand_percentage (int)`:以百分比扩大检测到的面部区域（默认为 0）。
- `threshold (float)`:指定阈值以确定一对代表同一个人还是不同的人。不变
- `normalization (str)`:在将输入图像输入模型之前对其进行规范化。
- `silent (bool)`:抑制或允许某些日志消息，以实现更安静的分析过程（默认为 False）。
- ` refresh_database(bool)`:将图像表示（pkl）文件与目录/db 文件同步，如果设置为 false，它将忽略 db_path 内的任何文件更改（默认为 True）。
- `anti_spoofing (bool)`:启用反欺骗的标志（默认为 False）。

**返回结果说明**

返回类型为 list (List[pd.DataFrame])，DataFrame 列包括

- `identity`:检测到的个人的身份标签。
- `target_x ,target_y ,target_w ,target_h`:数据库中目标人脸的边界框坐标。
- `source_x ,source_y ,source_w ,source_h`:源图像中检测到的人脸的边界框坐标。
- `threshold`:确定一对人脸是同一个人还是不同人的阈值。
- `distance`:基于指定模型和距离度量的人脸之间的相似度得分

## 人脸属性分析

>**Facial Attribute Analysis** - [`Demo`](https://youtu.be/GT2UeN85BdA)
>
>DeepFace also comes with a strong facial attribute analysis module including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`facial expression`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions. Result is going to be the size of faces appearing in the source image.
>
>面部属性分析 - 演示 DeepFace还配备了强大的面部属性分析模块，包括年龄、性别、面部表情（包括愤怒、恐惧、中性、悲伤、厌恶、快乐和惊讶）和种族（包括亚洲人、白人、中东人、印度人、拉丁裔和黑人）预测。结果将是源图像中出现的面部尺寸。

代码演示

```python
# 人脸识别模型 - 人脸属性分析 - Face analyze
# img = 3-3.jpg
def deepface_model_analyze(img: str):
    """
    给定一张图片分析，年龄、性别、面部表情、种族
    :param img:
    :return:
    """
    result = DeepFace.analyze(img_path=img)
    return result
```

打印运行结果得到

```
[
{
    'emotion': 
    	{
            'angry': np.float32(0.3104471),
            'disgust': np.float32(1.3859168e-09), 
            'fear': np.float32(0.037615385),
            'happy': np.float32(0.0009396329), 
            'sad': np.float32(0.49794665),
            'surprise': np.float32(2.9795643e-05),
            'neutral': np.float32(99.153015)
        },
    'dominant_emotion': 'neutral', 
    'region': 
    	{
            'x': 79,
            'y': 280,
            'w': 879, 
            'h': 879, 
            'left_eye': (661, 642), 
            'right_eye': (341, 612)
        },
    'face_confidence': np.float64(0.9),
    'age': 32, 
    'gender':
    	{
            'Woman': np.float32(0.0053363745),
            'Man': np.float32(99.99466)
        },
    'dominant_gender': 'Man', 
    'race': 
    	{
            'asian': np.float32(0.0038025668),
            'indian': np.float32(0.008881129), 
            'black': np.float32(0.00014986782), 
            'white': np.float32(97.29132), 
            'middle eastern': np.float32(0.94727564), 
            'latino hispanic': np.float32(1.748561)
        },
    'dominant_race': 'white'
}
]
```

结果显示图片上为白种人、男性、32岁、面部无表情，结果与图片一致

**请求参数说明**

- `img_path (str or np.ndarray)`：图像路径
- `actions (tuple)`:要分析的属性。默认值为（'age'、'gender'、'emotion'、'race'）。如果需要，您可以从分析中排除其中一些属性。
- ` enforce_detection (bool)`:如果在图像中未检测到人脸，则引发异常。
- `detector_backend`,`distance_metric`,`align`,`expand_percentage`,`silent`,`anti_spoofing`：不变

**返回结果说明**

返回类型为字典列表 (List[Dict[str, Any]])，每个字典包含以下键：

- `region (dict)`:表示图像中检测到的人脸的矩形区域。
  - `x`:人脸左上角的 x 坐标。
  - `y`:人脸左上角的 y 坐标。
  - `w`:检测到的人脸区域的宽度。
  - `h`:检测到的人脸区域的高度。
- `age (float)`:检测到的人脸的估计年龄。
- `face_confidence (float)`:检测到的人脸的置信度得分。表示人脸检测的可靠性。
- `dominant_gender (str)`:检测到的人脸中的主导性别。“男人”或“女人”。
- `gender (dict)`:每个性别类别的置信度得分。
  - `Man`:男性性别的置信度得分。
  - `Woman`:女性性别的置信度得分。
- `dominant_emotion (str)`:检测到的面部中的主要情绪。可能的值包括“悲伤”、“愤怒”、“惊讶”、“恐惧”、“高兴”、“厌恶”和“中性”
- `emotion (dict)`:每个情绪类别的置信度得分。
  - `sad`:悲伤的置信度得分。
  - `angry`:愤怒的置信度得分。
  - `surprise`:惊讶的置信度得分。
  - `fear`:恐惧的置信度得分。
  - `happy`:高兴的置信度得分。
  - `disgust`:厌恶的置信度得分。
  - `neutral`:中立性的置信度得分。
- `dominant_race (str)`:检测到的脸部中的主要种族。可能的值包括“印度人”、“亚洲人”、“拉丁裔西班牙人”、“黑人”、“中东人”和“白人”。
- `race (dict)`:每个种族类别的置信度得分。
  - `indian`:印度族裔的置信度得分。
  - `asian`:亚洲族裔的置信度得分。
  - `latino hispanic`:拉丁裔/西班牙裔族裔的置信度得分。
  - `black`:黑人族裔的置信度得分。
  - `middle eastern`:中东族裔的置信度得分。
  - `white`:白人族裔的置信度得分。

## 人脸检测和对齐

> **Face Detection and Alignment** - [`Demo`](https://youtu.be/GZ2p2hj2H5k)
>
> Face detection and alignment are important early stages of a modern face recognition pipeline. [Experiments](https://github.com/serengil/deepface/tree/master/benchmarks) show that detection increases the face recognition accuracy up to 42%, while alignment increases it up to 6%. [`OpenCV`](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [`Ssd`](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/), [`MtCnn`](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/), `Faster MtCnn`, [`RetinaFace`](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/), [`MediaPipe`](https://sefiks.com/2022/01/14/deep-face-detection-with-mediapipe/), `Yolo`, `YuNet` and `CenterFace` detectors are wrapped in deepface.
>
> 人脸检测和对齐 - 演示 人脸检测和对齐是现代人脸识别流程的重要早期阶段。实验表明，检测将人脸识别准确率提高了 42%，而对齐则将其提高了 6%。 OpenCV、Ssd、Dlib、MtCnn、Faster MtCnn、RetinaFace、MediaPipe、Yolo、YuNet 和 CenterFace 检测器都封装在 Deepface 中。

代码示例

```python
# 人脸识别模型 - 人脸检测 - Face extract
# img = 2-2.jpg
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
```

返回结果为：

```
[
{
    'face': array([[[0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        ...,
        [0.99215686, 0.99215686, 1.        ],
        [0.99215686, 0.99215686, 0.99215686],
        [0.99607843, 0.99607843, 0.99607843]],

       [[0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        ...,
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        [1.        , 1.        , 1.        ]],

       [[0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        ...,
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843]],

       ...,

       [[1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        ...,
        [1.        , 1.        , 1.        ],
        [0.99607843, 0.99607843, 0.99607843],
        [1.        , 1.        , 1.        ]],

       [[1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        ...,
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 0.99607843, 0.99607843],
        [0.99607843, 1.        , 1.        ]],

       [[1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        ...,
        [0.99607843, 1.        , 1.        ],
        [1.        , 1.        , 1.        ],
        [0.99607843, 0.99607843, 0.99607843]]]),
	'facial_area': 
		{
            'x': 321,
            'y': 77,
            'w': 197,
            'h': 197,
            'left_eye': (452, 153),
            'right_eye': (384, 153)
        },
	'confidence': np.float64(0.91)
}
]
```



**请求参数说明**

- `img_path (str or np.ndarray)`：第一幅图像的路径
- ` force_detection(bool)`:如果在图像中未检测到人脸，则引发异常。
- `color_face (str)`:返回人脸图像输出的颜色。选项：'rgb'、'bgr' 或 'gray'（默认为 'rgb'）。
- `normalize_face (bool)`:用于启用输出人脸图像标准化（除以 255）的标志人脸图像输出人脸图像标准化（默认为 True）。
- `detector_backend`,`distance_metric`,`align`,`expand_percentage`,`silent`,`anti_spoofing`：不变

**返回结果说明**

返回结果类型为字典列表 (List[Dict[str, Any]])，每个字典包含：

- `face (np.ndarray)`:检测到的脸部作为 NumPy 数组。
- `facial_area (Dict[str, Any]) `:检测到的脸部区域作为字典，
  - `x,y,w,h (int)`
  - `left_eye,right_eye (int,int)`:左眼和右眼,分别是相对于人本身而不是观察者的左侧和右侧的眼睛。
- `confidence (float)`:与检测到的脸部相关的置信度分数。
- `is_real (bool)`:反欺骗分析结果。仅当输入参数中的 anti_spoofing 设置为 True 时，此键才在结果中可用。
- `antispoof_score (float)`:反欺骗分析结果的分数。仅当输入参数中的 anti_spoofing 设置为 True 时，此键才在结果中可用。

## 人脸特征提取

> **Embeddings** - [`Demo`](https://youtu.be/OYialFo7Qo4)
>
> Face recognition models basically represent facial images as multi-dimensional vectors. Sometimes, you need those embedding vectors directly. DeepFace comes with a dedicated representation function. Represent function returns a list of embeddings. Result is going to be the size of faces appearing in the image path.
>
> …
>
> 嵌入 - 演示 人脸识别模型基本上将面部图像表示为多维向量。有时，您直接需要这些嵌入向量。 DeepFace 带有专用的表示功能。表示函数返回嵌入列表。结果将是图像路径中出现的面部尺寸。
>
> 。。。

代码演示：

```python
# 人脸识别模型 - 人脸特征提取 - Face represent
def deepface_model_represent(img: str):
    embedding_objs = DeepFace.represent(
        img_path=img
    )
    return embedding_objs
```

返回结果

```
[
{
    'embedding': [0.0, 0.0, 0.013042318041708077, ... , 0.05823250615542309, 0.0],
    'facial_area': 
    	{
            'x': 576, 
            'y': 254,
            'w': 347,
            'h': 347,
            'left_eye': (810, 388), 
            'right_eye': (681, 389)
        },
    'face_confidence': 0.9
}
]
```

**请求参数**没有大的变化

**返回结果说明**

返回类型为 字典列表 (List[Dict[str, Any]])，每个字典包含以下字段：

- `embedding (List[float])`:表示面部特征的多维向量。维度数根据参考模型而不同（例如，FaceNet 返回 128 个维度，VGG-Face 返回 4096 个维度）。
- `facial_area (dict)`:字典格式的面部检测检测到的面部区域。
  - `x,y,w,h,left_eye,right_eye`
- `face_confidence (float)`:面部检测的置信度得分。如果 `detector_backend` 设置为“skip”，则置信度将为 0 并且没有意义。

## 人脸实时分析

> **Real Time Analysis** - [`Demo`](https://youtu.be/-c9sSJcx6wI)
>
> You can run deepface for real time videos as well. Stream function will access your webcam and apply both face recognition and facial attribute analysis. The function starts to analyze a frame if it can focus a face sequentially 5 frames. Then, it shows results 5 seconds.
>
> 实时分析 - 演示 您也可以运行 deepface 来播放实时视频。流功能将访问您的网络摄像头并应用面部识别和面部属性分析。如果该函数可以连续对焦一张脸 5 帧，则该函数开始分析一帧。然后，它会显示结果 5 秒。

代码说明

```python
# 人脸识别模型 - 人脸实时分析 - Face stream
# db = img , video = 3.mp4 
def deepface_model_stream(db: str, video: str):
    DeepFace.stream(db_path=db,
                    source=video)
```

运行结果

视频中有数据集中的人，视频中的32岁左右（32）、男性（M）、与img中的3-1.jpg相似，心情为happy

![](./res/stream_res_3.jpg)

**请求参数说明**

- `db_path (str)`:包含图像文件的文件夹路径。数据库中所有检测到的人脸都将在决策过程中考虑。
- `enable_face_analysis (bool)` :启用人脸分析的标志（默认为 True）。
- `source (Any)`:视频流的来源（默认为 0，代表默认摄像头）。
- `time_threshold (int)`:人脸识别的时间阈值（以秒为单位）（默认为 5）。
- `frame_threshold (int)`:人脸识别的帧阈值（默认为 5）。
- 其余不变

**返回结果**：无

## 人脸识别模型比较

> 原文
>
> FaceNet, VGG-Face, ArcFace and Dlib are overperforming ones based on experiments - see [`BENCHMARKS`](https://github.com/serengil/deepface/tree/master/benchmarks) for more details. You can find the measured scores of various models in DeepFace and the reported scores from their original studies in the following table.
>
> 翻译过来就是
> 根据实验，FaceNet、VGG-Face、ArcFace 和 Dlib 均表现出色 - 有关更多详细信息，请参阅基准测试。您可以在下表中找到 DeepFace 中各种模型的测量分数以及原始研究报告的分数。

| Model(模型)  | Measured Score(实测分数) | Declared Score(公布分数) |
| :----------: | :----------------------: | :----------------------: |
|  Facenet512  |          98.4%           |          99.6%           |
| Human-beings |          97.5%           |          97.5%           |
|   Facenet    |          97.4%           |          99.2%           |
|     Dlib     |          96.8%           |          99.3 %          |
|   VGG-Face   |          96.7%           |          98.9%           |
|   ArcFace    |          96.7%           |          99.5%           |
| GhostFaceNet |          93.3%           |          99.7%           |
|    SFace     |          93.0%           |          99.5%           |
|   OpenFace   |          78.7%           |          92.9%           |
|   DeepFace   |          69.0%           |          97.3%           |
|    DeepID    |          66.5%           |          97.4%           |

## docker部署

> **Dockerized Service** - [`Demo`](https://youtu.be/9Tk9lRQareA)
>
> [![Docker Pulls](https://camo.githubusercontent.com/9825c1e9faee50fba7085e39510d9427287f1c6d46f44238ba5eeb478e34075d/68747470733a2f2f696d672e736869656c64732e696f2f646f636b65722f70756c6c732f736572656e67696c2f64656570666163653f6c6f676f3d646f636b6572)](https://hub.docker.com/r/serengil/deepface)
>
> The following command set will serve deepface on `localhost:5005` via docker. Then, you will be able to consume deepface services such as verify, analyze and represent. Also, if you want to build the image by your own instead of pre-built image from docker hub, [Dockerfile](https://github.com/serengil/deepface/blob/master/Dockerfile) is available in the root folder of the project.

```bash
# docker build -t serengil/deepface . # build docker image from Dockerfile
docker pull serengil/deepface # use pre-built docker image from docker hub
docker run -p 5005:5000 serengil/deepface
```

命令行演示

```bash
#face verification
$ deepface verify -img1_path tests/dataset/img1.jpg -img2_path tests/dataset/img2.jpg

#facial analysis
$ deepface analyze -img_path tests/dataset/img1.jpg
```

docker运行相关的shell命令 - [地址](https://github.com/serengil/deepface/blob/master/scripts/dockerize.sh#L17)

```
https://github.com/serengil/deepface/blob/master/scripts/dockerize.sh#L17
```

## 辅助函数

- 显示图片

```python
# 展示图片
def show_img(img):
    if isinstance(img, Image.Image):
        img.show()
    else:
        img = Image.open(img)
        img.show()
```

- 给人脸加框

```python
# 给人脸添加画框 - 分析面部属性使用
def picture_frame(src: str, res):
    _img = Image.open(src)  # 这里 src 是图像路径，正常加载
    # 访问 res 列表中的第一个字典
    x, y, w, h = res[0]['region']['x'], res[0]['region']['y'], res[0]['region']['w'], res[0]['region']['h']
    draw = ImageDraw.Draw(_img)
    draw.rectangle((x, y, x + w, y + h), outline="blue", width=3)
    print("emo:{}".format(res[0]["emotion"]))
    show_img(_img)
```



## 测试资料

1-1.jpg

![1-1.jpg](./img/1-1.jpg)

1-2.jpg

![1-2.jpg](./img/1-2.jpg)

1-3.jpg

![1-3.jpg](./img/1-3.jpg)

2-1.jpg

![2-1.jpg](./img/2-1.jpg)

2-2.jpg

![2-2.jpg](./img/2-2.jpg)

2-3.jpg

![2-3.jpg](./img/2-3.jpg)

3-1.jpg

![3-1.jpg](./img/3-1.jpg)

3-2.jpg

![3-2.jpg](./img/3-2.jpg)

3-3.jpg

![3-3.jpg](./img/3-3.jpg)

视频3.mp4

[链接](https://movie.douban.com/trailer/208886/#content)