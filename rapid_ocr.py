from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "test_small2.png"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")