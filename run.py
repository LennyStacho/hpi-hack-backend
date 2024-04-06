from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import requests
import base64
from io import BytesIO

app = Flask(__name__)
torch.no_grad()

processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=7,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load('fine_tuned_ViT.pth',
                      map_location='cpu'), strict=False)
model.eval()


@app.route('/', methods=['GET'])
def predict():
    if request.method == 'GET':
        # data = request.get_json(force=True)

        # base64_image = data.get('image')

        # if not base64_image:
        #     return jsonify({'error': 'No image provided'}), 400

        # if 'data:image' in base64_image and ';base64,' in base64_image:
        #     header, base64_image = base64_image.split(';base64,')

        # image_data = base64.b64decode(base64_image)

        # image = Image.open(BytesIO(image_data))

        model.load_state_dict(torch.load('fine_tuned_ViT.pth'))

        model.eval()

        id2label = {0: 'cars', 1: 'graffiti', 2: 'street',
                    3: 'street_sign', 4: 'traffic_light', 5: 'trash', 6: 'vandalism'}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # Data preprocessing

        image = 'https://previews.123rf.com/images/andreyshevchenko/andreyshevchenko1807/andreyshevchenko180700712/106570073-cars-trabant-in-berlin-germany.jpg'

        image = Image.open(requests.get(image, stream=True).raw)

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        normalized_logits = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_values, top_indices = torch.topk(normalized_logits, k=3, dim=-1)
        top_object = {}
        for index, top_indice in enumerate(top_indices[0].tolist()):
            top_value = top_values[0].tolist()[index]
            category = id2label[top_indice]
            top_object[category] = top_value

        return jsonify(top_object)


if __name__ == '__main__':
    app.run()
