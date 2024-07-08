<body>
  <h3 class="center pink-text">SKN02-2nd-1Team</h3> 
</body>

<p>


    
<body>

<div class="center">
    <h1>고객 이탈 예측 모델</h1>
    <h3>|&nbsp;&nbsp;&nbsp;송문영&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;장준영&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;사재민&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;이재원&nbsp;&nbsp;&nbsp;|</h3>
</div>

<br><div align="center">
    <h2> 🦋기술 스택🦋</h2>
  <div>
  <a href="https://www.python.org/downloads/release/python-370/"><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-1st-2Team/assets/168343721/034428e1-d5e8-417e-9dd1-617b5b68269c" alt="Untitled (1)" width="120" height="120"></a>
     <a href="https://pandas.pydata.org/"><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-1st-2Team/assets/168343721/2e04b09b-5b08-43f0-9659-21a92917fb1f" alt="Untitled (1)" width="120" height="120"></a>
     <a href="https://scikit-learn.org/stable/"><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/de34781a-f6a4-48bb-ac68-217f17903226" alt="Untitled (1)" width="120" height="120"></a>
    <a href="https://www.tensorflow.org/?hl=ko"><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/69da3041-e258-49cd-88b6-a95870085234" alt="Untitled (1)" width="120" height="120"></a>
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

<h2 class="pink-text">시각화</h2>
<h3> 1. 연령대에 따른 initial과 churn 비율<br></h3>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/459960a8-739b-423d-ae75-a811126850fc" alt="Untitled (1)"></a>
<p>

</p>

<h3> 2. 분기에 따른 Transaction 타입 <br></h3>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/57729781-6d37-47ef-8d15-c10e599f7abe" alt="Untitled (1)"></a>
<p>

</p>

<h3> 3. 진입 경로에 따른 초기 사용자의 구독 플랜 타입 <br></h3>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/3c6d5e41-03c3-41ec-8a8b-c680066c3882" alt="Untitled (1)"></a>
<p>

</p>

<h3> 5. Heat Map <br></h3>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/2b2d4452-4128-4f0a-bb91-868f9fb916bc" alt="Untitled (1)"></a>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/0396e962-6292-42cf-a4ee-b091a4ef8db7" alt="Untitled (1)"></a>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/d7e12dc1-c20b-4731-af75-7f0c9a4b4aeb" alt="Untitled (1)"></a>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/15b56de0-9788-4e40-8023-3a52bcaaa927" alt="Untitled (1)"></a>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/64c164fc-efd1-4eaf-b78f-1349ffb537c5" alt="Untitled (1)"></a>



<p>

</p>

</div>

<br><br>

<h2 class="pink-text">분석 모델</h2>



<h3> 1. DecisionTree<br></h3>
<p>
- Tree 구조로 의사 결정을 시각적 표현, 직관적으로 해석 가능<br>
- 각 노드 특성 조건을 나타내 최종 예측을 쉽계 따라감 → 명확한 의사결정 경로를 제공<br>
- 특성간의 비선형 관계 모델링<br>
- 과적합 방지 필요  
</p>
</hr>
<h3> 2. RandomForest<br></h3>
<p>
- 여러 트리를 앙상블하여 사용 → 노이즈에 강하고 안정적인 예측, 특성의 중요도 계산 가능<br>
- 연속형 변수와 범주형 변수 모두 처리<br>
- 각 트리를 분할할 때 트리 간의 상관성을 줄이고 모델의 성능을 향상<br>
</p>
<h3> 3. XGBoost<br></h3>
<p>
- 비교적 최근 공개<br>
- 자동으로 결측값을 처리<br>
- 정규호를 통한 과적합 방지<br>
- 높은 예측 정확도와 빠른 속도<br>
- 다수의 하이퍼파라미터 필요
</p>
<h3> 4. LightGBM<br></h3>
<p>
- 여러 개의 약한 학습기를 결합하여 강력한 예측 모델을 활용<br>
- 그리고 트리 성장 방식으로 leaf-wise를 사용 → 학습 속도가 빠르고 높은 예측 정확도를 보여줌
</p>
<h3>5. DNN<br><br>
-다층 퍼셉트론</h3>
<p>
- 여러 층의 뉴런을 쌓은 구조<br>
- 일반적 은닉층 : ReLU(Rectified Linear Unit) 함수<br>
- 일반적 출력층 : sigmoid, softmax 함수<br>
- 과적합 방지 : 드롭아웃(Dropout), 배치 정규화(Batch Normalization)와 같은 정규화 기법
</p>

</div>
<br><br>

<h2 class="pink-text">선택한 ML model</h2>

 * LightGBM

<br><br>

<h2>LightGBM 결과<br></h2>
<p>
Accuracy: 0.9512941831338714
  
                  precision    recall  f1-score   support
           False       0.94      1.00      0.97      2576
            True       0.99      0.84      0.91      1017
  
        accuracy                           0.95      3593
       macro avg       0.96      0.92      0.94      3593
    weighted avg       0.95      0.95      0.95      3593
  
</p>
<br><br>
<h2>기타 모델 결과<br></h2>
<p>
<h3>1. DecisionTree<br></h3>
Accuracy: 0.8883940996381854
  
                  precision    recall  f1-score   support
           False       0.89      0.97      0.93      2576
            True       0.89      0.69      0.78      1017

       accuracy                            0.89      3593
       macro avg       0.89      0.83      0.85      3593
    weighted avg       0.89      0.89      0.88      3593
</p>
<p>
<h3>2. Random Forest<br></h3>
Accuracy: 0.9273587531310882
  
                  precision    recall  f1-score   support
           False       0.92      0.98      0.95      2576
            True       0.95      0.78      0.86      1017

        accuracy                           0.93      3593
       macro avg       0.94      0.88      0.91      3593
    weighted avg       0.93      0.93      0.93      3593
</p>
<p>
<h3>3. XGBoost<br></h3>
Accuracy: 0.9448928472028946
  
                  precision    recall  f1-score   support

           False       0.94      0.98      0.96      2576
            True       0.96      0.84      0.90      1017

        accuracy                           0.94      3593
       macro avg       0.95      0.91      0.93      3593
    weighted avg       0.95      0.94      0.94      3593
</p>
<p>
<h3>4. LightGBM<br></h3>
Accuracy: 0.9512941831338714
    
                  precision    recall  f1-score   support
           False       0.94      1.00      0.97      2576
            True       0.99      0.84      0.91      1017

        accuracy                           0.95      3593
       macro avg       0.96      0.92      0.94      3593
    weighted avg       0.95      0.95      0.95      3593
</p>
<p>
<h3>5. LightGBM w/o period<br></h3>
Accuracy: 0.8569440578903423
      
                  precision    recall  f1-score   support
           False       0.86      0.96      0.91      2576
            True       0.86      0.59      0.70      1017

        accuracy                           0.86      3593
       macro avg       0.86      0.78      0.80      3593
    weighted avg       0.86      0.86      0.85      3593
</p>

<p>
<h3>6. PCA_XGBoost<br></h3>
Accuracy (PCA Data): 0.7008071249652101
   
                  precision    recall  f1-score   support
           False       0.77      0.84      0.80      2576
            True       0.46      0.35      0.40      1017

        accuracy                           0.70      3593
       macro avg       0.61      0.60      0.60      3593
    weighted avg       0.68      0.70      0.69      3593
</p>
<p>
<h3>7. PCA_LightGBM<br></h3>
Accuracy: 0.6927358753131089
        
                  precision    recall  f1-score   support
           False       0.77      0.82      0.79      2576
            True       0.45      0.37      0.40      1017

        accuracy                           0.69      3593
       macro avg       0.61      0.59      0.60      3593
    weighted avg       0.68      0.69      0.68      3593
</p>
<p>
<h3>8. MultiLayer Perceptron<br></h3>
Accuracy: 0.9454494714736938 (Batch Size : 256)

                  precision    recall  f1-score   support

           False       0.94      0.99      0.96      2576
            True       0.98      0.83      0.90      1017

        accuracy                           0.95      3593
       macro avg       0.96      0.91      0.93      3593
    weighted avg       0.95      0.95      0.94      3593
</p>


</div>
<div>
<br><br>
<h2>기타 모델 결과<br></h2>
<p><h3> ROC 커브 및 AUC</h3>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN02-2nd-1Team/assets/87643414/27263978-5de2-4260-a164-327803326eb9" alt="Untitled (1)"></a>
</p>

<br><br>
<h2>결론<br></h2>
<p>
  
* LGBM이 가장 높은 성능을 나타냄
* 그러나 다른 모델들 또한 92점 이상의 정확도를 보임
* PCA를 통해 차원 축소를 할 경우 성능이 하락할 수 있음<br>

</p>
</div>
<br><br>
</body>
</p>
