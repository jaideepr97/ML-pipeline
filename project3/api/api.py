import flask
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from flask import request
import json


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route("/", methods=["POST"])
def home():
    input_image = Image.open(request.files['file'])
    model = torchvision.models.densenet121(pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    data = json.load(open('/api/project3/api/imagenet_class_index.json'))

    #tensor = torch.nn.functional.softmax(output[0], dim=0)
    #print()
    #index = tensor.data.cpu().numpy().argmax()
    #print(index)
    return data[str(int(torch.argmax(output[0]).detach()))][1]

app.run(host='0.0.0.0',debug=True)

