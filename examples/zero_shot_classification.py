'''
Tính toán độ tương đồng giữa ảnh và văn bản:
    CLIP ánh xạ ảnh và văn bản vào cùng một không gian embedding, sau đó tính toán độ tương đồng giữa chúng để xác định nhãn phù hợp nhất.
    Xác suất được tính toán nhờ softmax để cho ra dự đoán xác suất trực quan hơn.
'''

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    '''
    Tính embedding (vector đặc trưng) của ảnh và text
    '''
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    '''
    Tính độ tương đồng giữa ảnh và mỗi câu văn bản trong danh sách.
        logits_per_image: Điểm số liên kết giữa ảnh và các câu văn bản.
        logits_per_text: Điểm số liên kết giữa văn bản và các ảnh (không sử dụng trong ví dụ này).
    '''
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
