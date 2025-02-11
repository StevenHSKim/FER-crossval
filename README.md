# FER-crossval
구성 요소
- 모델의 데이터셋 별 반복실험 코드
- 얼굴 데이터셋 정렬 전처리 코드

## 폴더 구조
```bash
```

<br>

## 논문 정보 및 링크

| **모델 이름** | **논문 제목** | **Venue** | **논문 링크** | **Github 링크** |
|:---------------:|---------------|----------|:---------------:|:-----------------:|
| POSTER | "A pyramid cross-fusion transformer network for facial expression recognition" | ICCV Workshop (AMFG) 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Zheng%2C+Ce%2C+Matias+Mendieta%2C+and+Chen+Chen.+%22Poster%3A+A+pyramid+cross-fusion+transformer+network+for+facial+expression+recognition.%22+Proceedings+of+the+IEEE%2FCVF+International+Conference+on+Computer+Vision.+2023.&btnG=) | [Github](https://github.com/zczcwh/POSTER) |
| DAN | "Distract your attention: Multi-head cross attention network for facial expression recognition" | Biomimetics 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Wen%2C+Zhengyao%2C+et+al.+%22Distract+your+attention%3A+Multi-head+cross+attention+network+for+facial+expression+recognition.%22+Biomimetics+8.2+%282023%29%3A+199.&btnG=) | [Github](https://github.com/yaoing/DAN) |
| DDAMFN | "A dual-direction attention mixed feature network for facial expression recognition" | Electronics 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Zhang%2C+Saining%2C+et+al.+%22A+dual-direction+attention+mixed+feature+network+for+facial+expression+recognition.%22+Electronics+12.17+%282023%29%3A+3595.&btnG=) | [Github](https://github.com/SainingZhang/DDAMFN) |
| LNSU-Net | "Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition" | NeurIPS 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=Leave+No+Stone+Unturned%3A+Mine+Extra+Knowledge+for+Imbalanced+Facial+Expression+Recognition&btnG=) | [Github](https://github.com/zyh-uaiaaaa/Mine-Extra-Knowledge?tab=readme-ov-file) |
| Ada-DF | "A Dual-Branch Adaptive Distribution Fusion Framework for Real-World Facial Expression Recognition" | ICASSP 2023 | [Paper](https://scholar.google.com/scholar?hl=ko&as_sdt=0%2C5&q=A+Dual-Branch+Adaptive+Distribution+Fusion+Framework+for+Real-World+Facial+Expression+Recognition.&btnG=) | [Github](https://github.com/taylor-xy0827/Ada-DF) |
| POSTER++ | "POSTER++: A simpler and stronger facial expression recognition network" | Pattern Recognit. 2024 | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320324007027) | [Github](https://github.com/talented-q/poster_v2) |
| MFER | "Multiscale Facial Expression Recognition Based on Dynamic Global and Static Local Attention" | IEEE Trans. Affect. Comput. 2024 | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10678884) | [Github](https://github.com/XuJ1E/MFER/?tab=readme-ov-file) |
| GSDNet | "A gradual self distillation network with adaptive channel attention for facial expression recognition" | Appl. Soft. Comput. 2024 | [Paper](https://www.sciencedirect.com/science/article/pii/S1568494624005362) | [Github](https://github.com/Emy-cv/GSDNet) |


<br>

## 데이터셋
| **데이터셋 이름** | **이미지 개수** | **공식 홈페이지** | **취득 방법** | **데이터셋 다운로드 링크** |
|:---------------:|:----------:|:---------------:|---------------|:---------------:|
| RAFDB | 15339 | [Homepage](http://www.whdeng.cn/RAF/model1.html#dataset) | [MTCNN](https://github.com/foamliu/Face-Alignment)을 이용하여 얼굴을 정렬 완료한 데이터셋을 다운로드 받아서 사용 하였음 | [Google Drive](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view) |
| FER2013 | 35887 | [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | 공식 Kaggle 링크에서 `icml_face_data.csv`를 다운로드 받은 뒤 "emotion"과 "pixel" 열만 남긴 `fer2013_modified`를 생성하여 사용하였음 | [Google Drive](https://drive.google.com/drive/folders/1-mGIAchWBUEhgmIKT36PrvQ1-LXl3Y5n?usp=sharing) |
| FERPlus | 35711 | [Github](https://github.com/Microsoft/FERPlus) | 공식 Github 링크에서 `fer2013new.csv`를 다운로드 받고, 직접 정의한 [label 생성 코드](https://github.com/StevenHSKim/FERPlus_Vote_To_Label)를 통해 `FERPlus_Label_modified.csv`를 생성하였음. [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)의 pixel값을 공식 Github의 `generate_training_data.py`로 png로 바꾸어 `FERPlus_Image`를 생성하였음 | [Google Drive](https://drive.google.com/drive/folders/1n73_68Zq4aa0KBImIANHhiSJMg6j2zVV?usp=sharing) |
| ExpW | 90560 | [Homepage](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) | 홈페이지에서 다운로드 받은 뒤, [MTCNN](https://github.com/foamliu/Face-Alignment)을 이용하여 얼굴을 직접 정렬 하였음 | [Google Drive](https://drive.google.com/drive/folders/1jNmC5RWqyBFvFsTHnWpg-cti0kpWxEi0?usp=sharing) |
| SFEW2.0 | 1634 | [Homepage](https://users.cecs.anu.edu.au/~few_group/AFEW.html) | 홈페이지를 통해 저자에게 데이터셋을 요청한 뒤, `_Aligned_Face` 데이터셋을 다운로드 받아 얼굴이 아닌 이미지는 직접 삭제한 뒤에 사용하였음 | [Google Drive](https://drive.google.com/drive/folders/1FuhcMW5LXaaWe8sKoGizQW78s04VcOHW?usp=sharing) |
| CK+ | 981 | [Homepage](https://www.jeffcohn.net/Resources/), [Kaggle](https://www.kaggle.com/datasets/shuvoalok/ck-dataset) | 각 동영상의 마지막 3개의 프레임을 캡처한 데이터셋을 다운로드 받아서 사용하였음 | [Google Drive](https://drive.google.com/drive/folders/1kuT6zQhZtyBPgTB4UqNJWq0ZBLAooTfA?usp=sharing) |


<br>

## 실험 결과
10회의 반복실험 결과의 Mean Std

- Evaluation Measure: Accuracy
  
| Dataset Name | POSTER++ | GSDNet | LNSUNet | POSTER | DDAMFN | Ada-DF | DAN |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| **CK+**     | 0.960 ± 0.013 | 0.993 ± 0.006 | 0.987 ± 0.008 | 0.984 ± 0.011 | **0.996 ± 0.005** | 0.992 ± 0.005 | 0.993 ± 0.008 |
| **SFEW2.0** | 0.291 ± 0.038 | 0.335 ± 0.089 | 0.496 ± 0.033 | 0.448 ± 0.031 | 0.531 ± 0.042 | **0.574 ± 0.003** | 0.566 ± 0.033 |
| **RAFDB**   | 0.756 ± 0.009 | 0.829 ± 0.019 | **0.889 ± 0.006** | 0.797 ± 0.011 | 0.874 ± 0.006 | 0.797 ± 0.008 | 0.867 ± 0.007 |
| **FERPlus** | 0.787 ± 0.004 | 0.816 ± 0.009 | 0.835 ± 0.004 | 0.796 ± 0.008 | 0.842 ± 0.005 | 0.836 ± 0.005 | **0.843 ± 0.004** |
| **FER2013** | 0.604 ± 0.010 | 0.635 ± 0.008 | 0.686 ± 0.004 | 0.628 ± 0.010 | 0.684 ± 0.004 | **0.702 ± 0.007** | 0.691 ± 0.003 |
| **ExpW**    | 0.632 ± 0.004 | 0.642 ± 0.003 | **0.662 ± 0.003** | 0.635 ± 0.004 | 0.647 ± 0.002 | 0.654 ± 0.003 | 0.649 ± 0.002 |


- Evaluation Measure: Balanced Accuracy
  
| Dataset Name | POSTER++ | GSDNet | LNSUNet | POSTER | DDAMFN | Ada-DF | DAN |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| **CK+**     | 0.944 ± 0.018 | 0.990 ± 0.011 | 0.984 ± 0.013 | 0.975 ± 0.017 | **0.993 ± 0.011** | 0.987 ± 0.012 | 0.990 ± 0.013 |
| **SFEW2.0** | 0.247 ± 0.035 | 0.280 ± 0.081 | 0.448 ± 0.035 | 0.385 ± 0.037 | 0.476 ± 0.043 | **0.514 ± 0.004** | 0.513 ± 0.033 |
| **RAFDB**   | 0.624 ± 0.015 | 0.726 ± 0.022 | **0.815 ± 0.014** | 0.687 ± 0.020 | 0.789 ± 0.012 | 0.686 ± 0.018 | 0.780 ± 0.011 |
| **FERPlus** | 0.577 ± 0.024 | 0.613 ± 0.025 | 0.629 ± 0.022 | 0.591 ± 0.020 | 0.668 ± 0.016 | 0.630 ± 0.017 | **0.669 ± 0.020** |
| **FER2013** | 0.541 ± 0.011 | 0.584 ± 0.015 | 0.640 ± 0.008 | 0.581 ± 0.009 | 0.658 ± 0.006 | **0.665 ± 0.009** | 0.664 ± 0.006 |
| **ExpW**    | 0.392 ± 0.007 | 0.403 ± 0.004 | **0.432 ± 0.005** | 0.398 ± 0.008 | 0.421 ± 0.007 | 0.447 ± 0.003 | 0.416 ± 0.007 |

<br>

## 모델 가중치 파일

- `CK+`, `SFEW2.0`, `RAFDB`, `FERPlus`, `FER2013`, `ExpW` 데이터셋에 대한 위 실험의 가중치 파일이 아래 경로에 있습니다.
- 해당 파일은 총 10번의 iteration 중 가장 validation accuracy가 높은 iteration의 checkpoint 파일입니다.

| Dataset Name | POSTER++ | GSDNet | LNSUNet | POSTER | DDAMFN | Ada-DF | DAN |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 링크 경로 | [Google Drive](https://drive.google.com/drive/folders/1S-xMMA5eYf6H8LIFBZDu9SKZKtxE53Qw?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1luspatzJMjf8Lw3c1NJ9Sl01Thg2YA-A?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1TZgG264-JxDb3tN8LSkilH-NvLqjqcMg?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1tfediz1x1n5oHxzPKgA6LilWwU710igN?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1yqEACstxlrwCVXYWf2WiI24Mtu2t_anS?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1zLCQBnXDPPZg7dUtyQs0ukNsE7tSsFHq?usp=sharing) | [Google Drive](https://drive.google.com/drive/folders/1kgJhwqQDywLRNKwIFF8-_K1nENDu05r6?usp=sharing) |


<br>

## 학습 시

### RAFDB
- RAFDB 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/raf-basic/
    EmoLabel/
        list_patition_label.txt
    Image/aligned/
        train_00001_aligned.jpg
        test_0001_aligned.jpg
        ...
```

### FER2013
- FER2013 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/FER2013/
    fer2013_modified.csv
```

### FERPlus
- FERPlus 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/FERPlus/
    FERPlus_Label_modified.csv
    FERPlus_Image/
        fer0000000.png
        fer0000001.png
        ...
```

### ExpW
- ExpW 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/ExpW/
    label/
        label.lst
    aligned_image/
        afraid_African_214.jpg
        afraid_american_190.jpg
        ...
```

### SFEW2.0
- SFEW2.0 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/SFEW2.0/
    sfew_2.0_labels.csv
    sfew2.0_images/
        image_000000.png
        image_000001.png
        ...
```

### CK+
- CK+ 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/CKPlus/
    ckplus_labels.csv
    ckplus_images/
        image_000000.png
        image_000001.png
        ...
```

<br>

## 사전학습 모델

| **모델 이름** | **사용되는 사전학습 모델 다운로드 링크** | **설명** |
|:---------------:|:----------:|----------|
| POSTER | [ir50 & mobilefacenet](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq) | improved resnet-50(이미지 특징 추출 백본)과 mobilefacenet(랜드마크 특징 추출 백본) |
| DAN | [resnet18](https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view) | msceleb 데이터셋으로 학습시킨 resnet-18 백본 |
| DDAMFN | - | - |
| LNSUNet | [swin transformer](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view) | swin transformer 백본 |
| Ada-DF | [resnet18](https://drive.google.com/file/d/1ByvxPD9QkmWZDWtTmDQ5ta1MiAkXt22T/view) | msceleb 데이터셋으로 학습시킨 resnet-18 백본 |
| POSTER++ | [ir50](https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view), [mobilefacenet](https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view) | improved resnet-50(이미지 특징 추출 백본)과 mobilefacenet(랜드마크 특징 추출 백본) |
| MFER | [resnet18](https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view) | msceleb 데이터셋으로 학습시킨 resnet-18 백본 |
| GSDNet | [resnet50](https://drive.google.com/drive/folders/1OUrrHPYRDneS5OcE6sk8PwbfP_zhmvmJ) | msceleb 데이터셋으로 학습시킨 resnet-50 백본 |

<br>

위 사전학습 모델을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
models/pretrain/
    ir50.pth                            # improved resnet50
    mobilefacenet_model_best.pth        # mobilefacenet
    resnet18_msceleb.pth                # resnet18
    start_0.pth                         # swin transformer
    vgg_msceleb_resnet50_ft_weight.pkl  # resnet50
```
