from paddleocr import PaddleOCR
from tqdm import tqdm
import json
import os

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch"
)

def ocr_image(img_path):
    result = ocr.predict(img_path)
    lines = []

    for res in result:
        texts = res.get("rec_texts", [])
        for t in texts:
            if t.strip():
                lines.append(t.strip())

    return lines

def is_pian_title(text):
    return "篇第" in text and len(text) < 25
def normalize(text):
    return text.replace(" ", "").replace("　", "")

def is_jieyao(text):
    t = normalize(text)
    return t in ["梅自强解要", "【梅自强解要】", "〔梅自强解要〕"]

def is_zhiyi(text):
    t = normalize(text)
    return t in ["廖冬晴直译", "【廖冬晴直译】", "〔廖冬晴直译〕"]
data = []

current = None
section = None

img_files = sorted(os.listdir("pages"))

for img in tqdm(img_files):
    lines = ocr_image(os.path.join("pages", img))

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 新篇
        if is_pian_title(line):
            if current:
                data.append(current)
            current = {
                "篇名": line,
                "原文": "",
                "梅自强解要": "",
                "廖冬晴直译": ""
            }
            section = "原文"
            continue

        if not current:
            continue

        if is_jieyao(line):
            section = "梅自强解要"
            continue

        if is_zhiyi(line):
            section = "廖冬晴直译"
            continue

        current[section] += line + "\n"

# 最后一篇
if current:
    data.append(current)
with open("huangdi_waijing.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 完成，共识别 {len(data)} 篇")
