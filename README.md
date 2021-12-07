# 농산물 가격 예측 프로젝트
 분해 시계열의 잔차를 feature로 활용한 LSTM 모델로 농산물 가격을 예측

## 데이터프레임
 - 2016.1.1부터 2020.1.4까지 농산물 거래량과 가격으로 구성된 데이터프레임
 - EDA와 기본 모델을 만드는데 사용
![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/%EB%8D%B0%EC%9D%B4%ED%84%B0%ED%94%84%EB%A0%88%EC%9E%84.png)

## EDA
 1. 품목별 농산물 가격 추세
 > - 계절 패턴이 뚜렷하고 품목별로 확인히 다른 분포를 보임
 > - 품목별 모델 생성
 > - 시계열 반영 모델 생성
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/2.png)
 2. 품목별 농산물 가격 분포
 > - 특이값이 많이 포함되어 있음
 > - 외부 요인에 의한 이상치로 예측해야 하는 값
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/3.png)
 3. 특이값 발생 원인
 > - 건고추의 경우 장마 기간이 길어지면 건조가 안돼서 가격 폭등 발생
 > - 특이값 예측 방안을 마련해야 할 필요가 있음
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/4.png)
 
## 모델링
 1. 첫번째 모델
 > - 대략적인 예측 성능을 알아보기 위한 모델
 > - 일주일 후 배추가격을 예측
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/5.png)
 > - 전반적인 추세는 잘 예측하지만 거래가 발생하지 않는 휴일 전후로 예측성능이 떨어지는 모습을 보임
 > - 큰 폭으로 변동되는 가격은 잘 예측하지 못함
 2. 두번째 모델
 > -feature selection 수행 모델
 > 
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/6.png) ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/7.png)
 > - 배추가격 그래프와 배추거래량 그래프
 > - 정상성을 띄는 시계열 데이터인 거래량과 달리 가격은 비정상 시계열 데이터
 > 
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/8.png)
 > - 상관분석 결과 거래량은 가격과 상관관계가 없음 -> feature에서 제외
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/9.png)
 > - 거래량을 feature에서 제외한 결과 휴일 전후와 특이값을 비교적 잘 예측함
 > - 하지만 거래가 발생하지 않는 휴일을 잘 예측하지 못함 -> 휴일 처리 방안 필요

## feature engineering
 1. 휴일 가격 평균 대체
 > 거래가 발생하지 않은 휴일의 농산물 가격을 하루 전, 하루 후 평균 가격으로 대체한 모델
 > ```
 > df = df[1:].replace(0, np.NaN)
 > df = df.interpolate().fillna(0)
 > df[:10]
 > ```
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/KakaoTalk_20211207_134249946.png)
 > - 예측 결과 전반적으로 준수한 성능을 보임
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/10.png) 
 2. 시계열 분해
 > - 분해 시계열의 잔차를 추가 feature로 활용
 > 
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/11.png)
 > ```
 > stl = STL(df[['date', '배추_가격(원/kg)']].set_index('date'), period=12)
 > res = stl.fit()
 > ```
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/12.png)
 > - 예측 결과 거의 완벽하게 fitting 된 모습을 보임

## 모델 선정
최종 모델을 선정하기 위해 4주후의 농산물 가격을 예측 하는 모델을 만들고 성능을 비교
 1. 랜덤포레스트 모델
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/13.png)
 > - 4주 후를 예측하는 랜덤포레스트 모델은 상대적으로 성능이 떨어짐
 2. LSTM 모델
 > ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/14.png)
 > - LSTM 모델의 경우 상대적으로 긴 기간도 잘 예측하는 모습을 보임
 > - 해당 LSTM 모델을 최종 모델로 선정

## 파라미터 튜닝
> ```
> with tf.device('/device:GPU:0'):
>   model = Sequential()
>   model.add(layers.Activation('relu'))
>   model.add(tf.compat.v1.keras.layers.CuDNNLSTM(100, input_shape=(21,len(feature)), return_sequences=True))
>   model.add(Dropout(0.1))
>   model.add(layers.Dense(30))
>   model.add(Dropout(0.1))
>   model.add(layers.Dense(1))
>   model.compile(optimizer='adam', loss='mse')
>   early_stopping = EarlyStopping(patience=30)
>   model.fit(train_X, train_y, epochs=1000, batch_size=32, validation_split = 0.1, callbacks=[early_stopping], verbose=1)
> ```
> - LSTM 모델은 과거 21일의 데이터를 활용하여 미래의 농산물 가격을 예측
> - 오버피팅을 방지하기 위해 전체 10%의 뉴런을 비활성화하는 Dropout Layer 추가
> - 전체 데이터의 10%를 Validation Set으로 활용하여 오버피팅을 방지

## 파이프라인 구축
 1. 클래스 생성
> ```
> class Nong1:
> 
>   def __init__(self, df, test):
>     #데이터프레임생성
>     #데이터전처리
>     #feature engneering
>     #feature selection
>   def set_feature(self,name):
>     #target의 feature를 설정
>   def set_target(self,week):
>     #target 설정
>     #target feature의 시계열 분해
>   def set_model(self):
>     #scaling
>     #학습 데이터프레임 선언
>     #LSTM Input reshape
>     #모델링
>   def get_plot(self):  
>     #과거 예측
>     #평가지표 MAE
>     #그래프 생성
>   def get_price(self):
>     #target으로 설정한 농산물 가격을 
> ```
 2. 파이프라인
> - 데이터프레임 생성
> ```
> df1 = pd.read_csv('/content/gdrive/MyDrive/nongsan_data/df1.csv', encoding='utf-8')
> ```
> - 데이터 로드
> ```
> response = urllib.request.urlopen(url).read()
> response = json.loads(response)
> ```
> - 데이터 전처리
> ```
> def preprocessing(tsalet_file):
> ```
> - 모델 실행
> ```
> for week in weeks:
>  print(week)
>  for feature in features:
>    my_nong1 = Nong1(df1, df2)
>    my_nong1.set_feature(feature)
>    my_nong1.set_target(week)
>    my_nong1.set_model()
>    if week == 1:
>      week1.append(my_nong1.get_price())
>    if week == 2:
>      week2.append(my_nong1.get_price())
>    if week == 4:
>      week4.append(my_nong1.get_price())
> ```
> - 데이터프레임 업데이트
> ```
> df1 = pd.concat([df1, df2], axis=0)
> df1.to_csv('/content/gdrive/MyDrive/nongsan_data/df1.csv', encoding='utf-8-sig', index=False)
> ```
3. 성과
> ![image](https://github.com/jungsungmoon/nongsan/blob/main/pic/123123.png)
> [데이콘 농산물 가격 예측 AI 경진대회](https://dacon.io/competitions/official/235801/overview/description)
> - 해당 대회에서 5위의 성적으로 
