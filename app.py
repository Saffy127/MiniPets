from fastai.vision.all import *
import gradio as gr
from fastai.interpret import *
from fastai.vision.learner import _get_precompute

def load_image(img_path):
    img = PILImage.create(img_path)
    img = img.resize((192, 192))
    return img

def classify_image(img):
    # Converting Gradio Image input to FastAI PILImage
    img = PILImage.create(img)
    
    # Prediction
    pred, idx, probs = learn.predict(img)
    
    # Getting explanation using Grad-CAM
    gcam = GradCam.from_one_img(learn, img)
    heatmap, _ = gcam.get_heatmap()
    explanation = gcam.color_heatmap(heatmap)

    # Returning the prediction and explanation
    return dict(zip(categories, map(float, probs))), explanation

learn = load_learner('pet_model.pkl')

categories = ('Dog', 'Cat')

image = gr.inputs.Image(shape=(192, 192), type="pil")
label = gr.outputs.Label(num_top_classes=2)
image_explanation = gr.outputs.Image(type="pil")

examples = [['dog.jpg'], ['cat.jpg'], ['cat1.jpg'], ['dog1.jpg']]

intf = gr.Interface(fn=classify_image,
                    inputs=image,
                    outputs=[label, image_explanation],
                    examples=examples,
                    live=True)  # Live updates

intf.launch(inline=False)
