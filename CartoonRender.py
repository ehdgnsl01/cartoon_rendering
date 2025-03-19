import cv2
import numpy as np
import matplotlib.pyplot as plt

def 엣지_마스크_생성(image):
    # 1. 그레이스케일 변환 (입력 이미지는 RGB라고 가정)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2. 노이즈 감소: Median Blur 적용 (커널 크기 7)
    gray = cv2.medianBlur(gray, 7)
    # 3. Adaptive Threshold 적용 (blockSize=5, C=7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  5, 7)
    return edges

def 컬러_팔레트_축소(image, k):
    # 이미지 데이터를 float32로 변환 후 (Nx3) 행렬로 재구성
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result

def 어두운_부분_강조(image, threshold=100, factor=0.8):
    """
    image: 입력 이미지 (uint8, RGB)
    threshold: 어두운 영역을 결정하는 임계값 (0~255)
    factor: 어두운 영역의 픽셀 값에 곱해질 계수 (1보다 작으면 어둡게 됨)
    """
    image_float = image.astype(np.float32)
    result = np.where(image_float < threshold, image_float * factor, image_float)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def 카툰_이미지_합성(edges, quantized):
    # 엣지 마스크가 255인 영역은 검정색으로 남기고, 나머지는 quantized 색상 이미지 유지
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
    return cartoon

def 블러필터_적용(image):
    # Bilateral Filter를 적용하여 약한 블러 효과 부여 (sigmaColor와 sigmaSpace 값을 낮춤)
    blurred = cv2.bilateralFilter(image, 7, 50, 50)
    return blurred

def 카툰_이미지_생성(image):
    edges = 엣지_마스크_생성(image)
    quantized = 컬러_팔레트_축소(image, 15)
    quantized = 어두운_부분_강조(quantized, threshold=100, factor=0.8)
    cartoon = 카툰_이미지_합성(edges, quantized)
    final_cartoon = 블러필터_적용(cartoon)
    return final_cartoon

if __name__ == "__main__":
    # 이미지 로드 (OpenCV는 기본적으로 BGR 형식으로 읽으므로, matplotlib용 RGB 변환)
    image_bgr = cv2.imread("input.jpg")
    if image_bgr is None:
        print("이미지를 불러올 수 없습니다. 파일명을 확인해주세요.")
        exit()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 카툰 이미지 생성
    cartoon_image = 카툰_이미지_생성(image_rgb)
    
    # 원본과 최종 카툰 이미지를 동시에 출력 (matplotlib subplot 사용)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(cartoon_image)
    axs[1].set_title("Final Cartoon Image")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
    
    # 최종 결과를 저장 (저장 전 RGB -> BGR 변환)
    cartoon_bgr = cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("cartoon_output.jpg", cartoon_bgr)
    print("최종 카툰 이미지가 'cartoon_output.jpg'로 저장되었습니다.")
