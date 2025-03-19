# 코드 설명

이 코드는 `input.jpg` 이미지를 불러와서 만화 스타일로 변환하는 파이프라인을 구현합니다.  
전체 파이프라인은 다음과 같은 단계로 구성됩니다.

1. **엣지 마스크 생성 (Edge Mask Generation)**
2. **컬러 팔레트 축소 (Color Palette Reduction)**
3. **어두운 부분 강조 (Enhance Dark Areas)**
4. **엣지와 컬러 이미지 합성 (Image Composition)**
5. **최종 블러 필터 적용 (Final Blurring)**
6. **원본 이미지와 최종 카툰 이미지 동시 출력**

---

## 주요 함수 및 역할

### 0. 원본_이미지
<br>
<img src="https://github.com/user-attachments/assets/9134b597-d6ec-4e70-93bd-52cb8178f2e3" height=400>

### 1. 엣지_마스크_생성 (Edge Mask Generation)

- **역할:**  
  입력된 RGB 이미지를 그레이스케일로 변환한 후, 노이즈를 줄이기 위해 median blur를 적용합니다.  
  이후 adaptive threshold를 적용하여 지역 평균 기반의 이진 엣지 마스크를 생성합니다.

- **핵심 코드:**
  ```python
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray = cv2.medianBlur(gray, 7)
  edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  7, 5)
  ```
- **사진:**
  <br>
 <img src="https://github.com/user-attachments/assets/036025b4-350e-4adb-8b68-ccdeec4bca8c" height=400>
  
### 2. 컬러_팔레트_축소 (Color Palette Reduction)

- **역할:**  
  k-means 클러스터링을 사용하여 이미지의 색상을 k개의 군집으로 단순화합니다.  
  이렇게 축소된 컬러 팔레트는 만화 스타일의 평면적이고 단순한 색감을 만드는데 도움을 줍니다.

- **핵심 코드:**
  ```python
  data = np.float32(image).reshape((-1, 3))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(image.shape)
  ```
- **사진:**
  <br>
  <img src="https://github.com/user-attachments/assets/d8d12839-cde8-4479-baee-b6a3b7ae0ddc" height=400>
  
### 3. 어두운_부분_강조 (Enhance Dark Areas)

- **역할:**  
  컬러 팔레트 축소 후 어두운 부분만 선택적으로 더 어둡게 처리합니다.  
  임계값(`threshold`) 이하인 픽셀에 대해 `factor` 계수를 곱하여 어두운 영역의 대비를 높입니다.

- **핵심 코드:**
  ```python
  result = np.where(image_float < threshold, image_float * factor, image_float)
  result = np.clip(result, 0, 255).astype(np.uint8)
  ```
- **사진:**
  <br>
  <img src="https://github.com/user-attachments/assets/7cd8eecc-4fad-4557-8863-bf783fd083a9" height=400>
  
### 4. 카툰_이미지_합성 (Image Composition)

- **역할:**  
  생성된 엣지 마스크와 단순화된(quantized) 컬러 이미지를 결합합니다.  
  엣지 마스크의 값이 255인 부분(엣지 영역)을 기준으로 bitwise 연산을 수행하여 만화 스타일 이미지를 만듭니다.

- **핵심 코드:**
  ```python
  cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
  ```
- **사진:**
  <br>
  <img src="https://github.com/user-attachments/assets/308c2e32-fed2-44f6-8c01-ec659cbdfc22" height=400>
  
### 5. 블러필터_적용 (Final Blurring)

- **역할:**  
  최종 합성된 만화 스타일 이미지에 Bilateral Filter를 적용하여 약간의 부드러운 효과를 추가합니다.  
  여기서는 sigmaColor와 sigmaSpace 값을 낮춰서 블러 효과를 약하게 적용합니다.

- **핵심 코드:**
  ```python
  blurred = cv2.bilateralFilter(image, 7, 50, 50)
  ```
- **사진:**
  <br>
  <img src="https://github.com/user-attachments/assets/7922fb08-4ef6-4897-a395-802b112937db" height=400>
  
### 6. 카툰_이미지_생성 (Cartoon Image Pipeline)

- **역할:**  
  위의 함수들을 순서대로 호출하여 최종 만화 이미지를 생성합니다.
  
- **파이프라인 순서:**
  1. **엣지 마스크 생성** → `엣지_마스크_생성(image)`
  2. **컬러 팔레트 축소** → `컬러_팔레트_축소(image, 15)`
  3. **어두운 부분 강조** → `어두운_부분_강조(quantized, threshold=100, factor=0.8)`
  4. **엣지와 컬러 이미지 합성** → `카툰_이미지_합성(edges, quantized)`
  5. **최종 블러 적용** → `블러필터_적용(cartoon)`

- **코드 조합:**
  ```python
  def 카툰_이미지_생성(image):
      edges = 엣지_마스크_생성(image)
      quantized = 컬러_팔레트_축소(image, 15)
      quantized = 어두운_부분_강조(quantized, threshold=100, factor=0.8)
      cartoon = 카툰_이미지_합성(edges, quantized)
      final_cartoon = 블러필터_적용(cartoon)
      return final_cartoon
  ```
- **사진:**
  <br>
  <img src="https://github.com/user-attachments/assets/d84ba11b-4c22-44d7-9efb-a3631cca271a" height=400>
  
### 7. 메인 실행 부분

- **역할:**  
  `input.jpg` 파일을 OpenCV로 로드한 후 BGR → RGB 변환을 진행합니다.  
  이후, `카툰_이미지_생성` 함수를 호출하여 만화 스타일 이미지를 생성하고,  
  원본 이미지와 최종 결과를 matplotlib의 subplot으로 한 화면에 동시에 출력합니다.
  
- **최종 결과 저장:**  
  최종 이미지는 RGB에서 BGR로 변환 후 `cartoon_output.jpg`로 저장합니다.

- **코드 예시:**
  ```python
  if __name__ == "__main__":
      image_bgr = cv2.imread("input.jpg")
      if image_bgr is None:
          print("이미지를 불러올 수 없습니다. 파일명을 확인해주세요.")
          exit()
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
      
      cartoon_image = 카툰_이미지_생성(image_rgb)
      
      fig, axs = plt.subplots(1, 2, figsize=(12, 6))
      axs[0].imshow(image_rgb)
      axs[0].set_title("Original Image")
      axs[0].axis("off")
      axs[1].imshow(cartoon_image)
      axs[1].set_title("Final Cartoon Image")
      axs[1].axis("off")
      plt.tight_layout()
      plt.show()
      
      cartoon_bgr = cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR)
      cv2.imwrite("cartoon_output.jpg", cartoon_bgr)
      print("최종 카툰 이미지가 'cartoon_output.jpg'로 저장되었습니다.")
  ```
- **사진:**
  <br>
  <img src="https://github.com/user-attachments/assets/9b595519-8bb5-4b04-bfff-19e394d3fd77" height=400>
  <br>
  <img src="https://github.com/user-attachments/assets/757271e1-f1f3-49a5-9a7e-b6592d73d748" height=400>

---
## 데모 및 한계점
- **잘 나오는 이미지**
  <br>
  <img src="https://github.com/user-attachments/assets/9b595519-8bb5-4b04-bfff-19e394d3fd77" height=400>
- **잘 나오지 않는 이미지**
  <br>
  <img src="https://github.com/user-attachments/assets/17a43139-61f0-47b7-8ce1-20b2fe342b4c" height=400>

- **원인**
  - 엣지 마스크를 생성할때 Adaptive Threshold의 blocksize값과 c값의 변화에 따라 선의 굵기가 달라진다.
  - 위에 잘 나오는 이미지인 드래곤볼 이미지는 blocksize=7, c=5 인 값을 넣은 결과이고
  - 똑같은 값으로 아래 디즈니 공주들 이미지를 변환했을때 선 굵기가 너무 굵다는 것을 알 수 있다.
  - 특히 인물사진의 경우 edge선이 많이 나오기 때문에 선의 굵기를 얇게 가져가야한다. (사진 참고)
    <br>
    <img src="https://github.com/user-attachments/assets/ed13b002-736c-468c-b96c-d05f8aa3538a" height=400>
    <br>
    <img src="https://github.com/user-attachments/assets/828523b3-71b1-4d45-ba35-0c1e029b0b25" height=400>
  - 카툰스타일의 느낌은 선에서 나온다고 생각하기 때문에 이미지에 따라 선 굵기를 조정해줄 필요가 보인다.
  
   
