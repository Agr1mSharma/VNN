#Flask server

from flask import Flask, jsonify, request, render_template
from PIL import Image, ImageOps
import base64
from torchvision import transforms, torch
from cnn import DigitCNN
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


model = DigitCNN()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

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
    # 7. return jsonify({"prediction": digit})
    return jsonify({"prediction": prediction})
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


@app.route("/api")
def apiRoute():
    data = {"name": "alice", "age": 25}
    return jsonify(data) 



if __name__ == "__main__":
    app.run()


