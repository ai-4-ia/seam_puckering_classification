import streamlit as st
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
import os

st.set_page_config(page_title='Seam Puckering Classifcation')

class_names = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
def load_model(device='cpu'):
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True).to(device)
    mobilenet.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(in_features=1280, out_features=5, bias=True)
    ).to(device)
    if(os.path.isfile('./mobilenet.pt')):
        mobilenet.load_state_dict(torch.load('./mobilenet.pt', map_location=device))
        mobilenet.eval()
        print('Load model successfully!')
    else:
        print('No model found!')

    return mobilenet

def predict_image(model, imageTensor, device='cpu'):
    model.to(device)
    torchVisionTransform = T.ToPILImage()
    img = torchVisionTransform(imageTensor)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    st.write(f'Result: {class_names[target_image_pred_label[0].item()]}')

model = load_model('cpu')

st.title("Seam Puckering Classification App")
open_camera = st.button('Take picture')
close_camera = st.button('Turn off the camera')

if 'is_open_camera' not in st.session_state:
    st.session_state['is_open_camera'] =  False

if close_camera:
    st.session_state['is_open_camera'] =  False

if open_camera or st.session_state['is_open_camera']:
    # Open the camera
    st.session_state['is_open_camera'] =  True
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
            # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
        bytes_data = img_file_buffer.getvalue()
        torch_img = torchvision.io.decode_image(
                torch.frombuffer(bytes_data, dtype=torch.uint8)
            )
        predict_image(model, torch_img)

        # st.write(torch_img)
else:
    st.session_state['is_open_camera'] = False


image_ext = ["png", "jpg", "jpeg", "heic", "heif"]

uploaded_file = st.file_uploader("Or choose a file")
if uploaded_file is not None:
    ext_position = len(uploaded_file.name.split('.')) - 1
    file_ext = uploaded_file.name.split('.')[ext_position]
    if file_ext in image_ext:
        bytes_data = uploaded_file.getvalue()
        torch_img = torchvision.io.decode_image(
                torch.frombuffer(bytes_data, dtype=torch.uint8)
            )

        predict_image(model, torch_img)
