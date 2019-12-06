import flask
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from flask import request
#model = torchvision.models.densenet121(pretrained=True)


app = flask.Flask(__name__)
app.config["DEBUG"] = True

'''
@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
'''

@app.route("/", methods=["POST"])
def home():
    input_image = Image.open(request.files['file'])
    #img.show()
    model = torchvision.models.densenet121(pretrained=True)
    #input_image = Image.open("/Users/aayushgupta/Downloads/1a-2020-kia-telluride-kbb.jpg")
    input_image.show()
    preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
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
    tensor = torch.nn.functional.softmax(output[0], dim=0)
    #print()
    index = tensor.data.cpu().numpy().argmax()
    #print(index)
    return str(index)

app.run()

