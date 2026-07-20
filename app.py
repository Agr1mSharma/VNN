#Flask server

from flask import Flask, jsonify, request, render_template
from PIL import Image, ImageOps
import base64
from torchvision import transforms
import torch
from cnn import DigitCNN
from io import BytesIO
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


model = DigitCNN()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

activations = {}

def hook_fn_conv1(module, input, output):
    activations['conv1'] = output.detach()

def hook_fn_conv2(module, input, output):
    activations['conv2'] = output.detach()

model.conv1.register_forward_hook(hook_fn_conv1)
model.conv2.register_forward_hook(hook_fn_conv2)

@app.route("/predict", methods = ["POST"])
def predict():
    # 1. get image data from request.json['image']
    data = request.get_json()
    print(data)
    # 2. strip the header "data:image/png;base64," from the string
    image_data = data['image']
    image_data = image_data.split(",")[1]
    #data = data[21::]
    # 3. decode base64 → bytes using base64 library
    decoded_bytes= base64.b64decode(image_data)
    # 4. convert bytes → PIL Image using Image.open(BytesIO(...))
    image = Image.open(BytesIO(decoded_bytes))
    # fix transparency by pasting onto white background
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])  # use alpha channel as mask
    image = background
    image.save("predebug.png")
    # 5. preprocess:
    #    - convert to grayscale
    image = image.convert("L")
    #    - invert colors
    image = ImageOps.invert(image)
    #    - resize to 28x28
    image = image.resize((28, 28))
    image.save("debug.png")
    #    - convert to tensor
    image = transforms.ToTensor()(image)
    #    - normalize
    image = transforms.Normalize((0.1307,), (0.3081,))(image)
    #    - add batch dimension
    image = image.unsqueeze(0)
    # 6. load model, run forward pass
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()
        conv1_features = []
        conv2_features = []
        for i in range(activations['conv1'].shape[1]):  # loop through 32 channels
            feature_map = activations['conv1'][0, i]    # shape: (26, 26)
            conv1_features.append(tensor_to_base64(feature_map))
        for i in range(activations['conv2'].shape[1]):  # loop through 64 channels
            feature_map = activations['conv2'][0, i]    # shape: (26, 26)
            conv2_features.append(tensor_to_base64(feature_map))
    # 7. return jsonify({"prediction": digit})
    return jsonify({"prediction": prediction,
                    "conv1_features": conv1_features,
                    "conv2_features": conv2_features
    })
    #pass


@app.route("/status")
def getStatus():
    ret = {"status": "ok"}
    return jsonify(ret)

@app.route("/hello")
def hello():
    return "Hello World!"

"""
@app.route("/", methods = ['GET', 'POST'])
def index():
    return "Homepage"
"""

def tensor_to_base64(tensor):
    # tensor shape is (height, width) — single feature map
    # 1. convert to numpy
    array = tensor.numpy()
    # 2. normalize to 0-255 range
    #normalized_array = transforms.Normalize(0, 255)(array)
    diff = array.max() - array.min()
    if diff == 0:
        array = np.zeros_like(array, dtype='uint8')
    else:
        array = ((array - array.min()) / diff * 255).astype('uint8')
    # 3. convert to PIL Image
    image = Image.fromarray(array)
    # 4. save to BytesIO as PNG
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    # 5. encode as base64 and return string
    buffer.seek(0)  # rewind to start before reading
    base64_string = base64.b64encode(buffer.read()).decode("utf-8")
    return base64_string

@app.route("/api")
def apiRoute():
    data = {"name": "alice", "age": 25}
    return jsonify(data) 



if __name__ == "__main__":
    app.run()


