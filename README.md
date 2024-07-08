<div align="center">

</head>
<body>
  <h3 class="center pink-text">SKN02-2nd-1Team</h3> 
</body>

<p>


    
<body>

<div class="center">
    <h1>고객 이탈 예측 모델</h1>
    <h3>|&nbsp;&nbsp;&nbsp;송문영&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;장준영&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;사재민&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;이재원&nbsp;&nbsp;&nbsp;|</h3>
</div>

<br>
</div>
<h2>기술 스택</h2>
<h3>Python, Pandas, SKlearn, Tensorflow</h3>
</div>

<br><br>


<h2 class="pink-text">목적</h2>
<p>* 구독 서비스 이용 고객들의 이탈 데이터를 분석하여 이탈 예측 모델을 개발 *</p>
</div>

<br><br>


<h2 class="pink-text">한계점</h2>
<p>데이터 출처의 한계: 미상 데이터 출처를 사용</p>
    
</div>

<br><br>


<h2 class="pink-text">분석 모델</h2>



<h3> 1. DecisionTree<br></h3>
<p>
        Tree 구조로 의사 결정을 시각적으로 표현할 수 있어 매우 직관적이고 해석하기가 쉽습니다 .또한 각 노드에 각 특성에 대한 조건을 나타내 최종 예측을 쉽게 따라갈 수 있어 명확한 의사결정 경로를 제공합니다. 또한 특성 간의 비선형 관계를 잘 모델링 할 수 있고 복잡한 상호작용을 자동으로 학습하며 전처리가 필요가 없습니다. 마지막으로 과적합을 쉽게 방지할 수 있습니다.
</p>
</hr>
<h3> 2. RandomForest<br></h3>
<p>
        여러 트리를 앙상블하여 사용하므로, 노이즈에 강하고 안정적인 예측을 할 수 있고, 특성의 중요도를 계산할 수 있습니다.  그리고 연속형 변수와 범주형 변수를 모두 처리할 수 있으며 각 트리를 분할할 때 트리 간의 상관성을 줄이고 모델의 성능을 향상시킬 수 있습니다.
</p>
<h3> 3. XGBoost<br></h3>
<p>
        다양한 평가 지표를 사용하고 자동으로 결측값을 처리해줄 수 있습니다 .또한 정규화를 통해 과적합을 방지하여 높은 예측 정확도와 빠른 속도를 보여 효율적으로 분석이 가능합니다.
</p>
<h3> 4. LightGBM<br></h3>
<p>
        그라디언트 부스팅 기반 모델로, 여러 개의 약한 학습기를 결합하여 강력한 예측 모델을 활용합니다. 그리고 트리 성장 방식으로 leaf-wise를 사용합니다. 이는 기존의 level-wise 방법보다 더 깊고 복잡한 트리를 생성할 수 있어 학습 속도가 빠르고 높은 예측 정확도를 보여줍니다. 또한 LightGBM은 과적합을 방지하기 위한 다양한 하이퍼파라미터 튜닝 옵션을 제공하여 조절이 가능합니다.
</p>
<h3>5. DNN<br><br>
-다층 퍼셉트론</h3>
<p>
여러 층의 뉴런을 쌓아 다양한 구조를 가질 수 있고, 일반적으로 ReLU(Rectified Linear Unit) 함수가 은닉층에서, 출력층에서는 문제 유형에 따라 sigmoid나 softmax 함수가 비선형 활성화 함수로 사용되어 데이터의 복잡한 비선형 관계를 모델링할 수 있습니다. 드롭아웃(Dropout), 배치 정규화(Batch Normalization)와 같은 정규화 기법을 사용하여 과적합을 방지하고 하이퍼파라미터 튜닝**을 통해** 모델의 성능을 최적화할 수 있습니다.
</p>

</div>
<br><br>

<h2 class="pink-text">선택한 ML model</h2>

<h2>"LightGBM"</h2>
<br><br>

<h2>LightGBM 결과</h2>
<p>
Accuracy: 0.9512941831338714
  
                  precision    recall  f1-score   support
           False       0.94      1.00      0.97      2576
            True       0.99      0.84      0.91      1017
  
        accuracy                           0.95      3593
       macro avg       0.96      0.92      0.94      3593
    weighted avg       0.95      0.95      0.95      3593
  
</p>

<h2>기타 모델 결과</h2>
<p>
1. DecisionTree
Accuracy: 0.8883940996381854
  
                  precision    recall  f1-score   support
           False       0.89      0.97      0.93      2576
            True       0.89      0.69      0.78      1017

       accuracy                            0.89      3593
       macro avg       0.89      0.83      0.85      3593
    weighted avg       0.89      0.89      0.88      3593
</p>
<p>
2. Random Forest
Accuracy: 0.9273587531310882
  
                  precision    recall  f1-score   support
           False       0.92      0.98      0.95      2576
            True       0.95      0.78      0.86      1017

        accuracy                           0.93      3593
       macro avg       0.94      0.88      0.91      3593
    weighted avg       0.93      0.93      0.93      3593
</p>
<p>
3. XGBoost
Accuracy: 0.9448928472028946
  
                  precision    recall  f1-score   support

           False       0.94      0.98      0.96      2576
            True       0.96      0.84      0.90      1017

        accuracy                           0.94      3593
       macro avg       0.95      0.91      0.93      3593
    weighted avg       0.95      0.94      0.94      3593
</p>
<p>
4. LightGBM
Accuracy: 0.9512941831338714
    
                  precision    recall  f1-score   support
           False       0.94      1.00      0.97      2576
            True       0.99      0.84      0.91      1017

        accuracy                           0.95      3593
       macro avg       0.96      0.92      0.94      3593
    weighted avg       0.95      0.95      0.95      3593
</p>
<p>
5. LightGBM w/o period
Accuracy: 0.8569440578903423
      
                  precision    recall  f1-score   support
           False       0.86      0.96      0.91      2576
            True       0.86      0.59      0.70      1017

        accuracy                           0.86      3593
       macro avg       0.86      0.78      0.80      3593
    weighted avg       0.86      0.86      0.85      3593
</p>

<p>
6. PCA_XGBoost
Accuracy (PCA Data): 0.7008071249652101
   
                  precision    recall  f1-score   support
           False       0.77      0.84      0.80      2576
            True       0.46      0.35      0.40      1017

        accuracy                           0.70      3593
       macro avg       0.61      0.60      0.60      3593
    weighted avg       0.68      0.70      0.69      3593
</p>
<p>
7. PCA_LightGBM
Accuracy: 0.6927358753131089
        
                  precision    recall  f1-score   support
           False       0.77      0.82      0.79      2576
            True       0.45      0.37      0.40      1017

        accuracy                           0.69      3593
       macro avg       0.61      0.59      0.60      3593
    weighted avg       0.68      0.69      0.68      3593
</p>
<p>
8. MultiLayer Perceptron
Accuracy: 0.9454494714736938 (Batch Size : 256)

                  precision    recall  f1-score   support

           False       0.94      0.99      0.96      2576
            True       0.98      0.83      0.90      1017

        accuracy                           0.95      3593
       macro avg       0.96      0.91      0.93      3593
    weighted avg       0.95      0.95      0.94      3593
</p>


</div>

<br><br>
</body>
</p>
