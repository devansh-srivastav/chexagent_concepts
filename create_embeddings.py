# Importing Libraries
from modeling_chexagent import CheXagentVisionEmbeddings
from configuration_chexagent import CheXagentVisionConfig
from PIL import Image
from torchvision import transforms
from mistralai.client import MistralClient
import os
import torch
import pickle
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Mistral API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# Pneumonia Concepts
pneumonia_concepts = [
    'Presence of alveolar consolidation',
    'Air bronchograms within consolidation',
    'Obscured cardiac or diaphragmatic borders',
    'Pleural effusion present',
    'Increased interstitial markings',
    'Lobar, segmental, or subsegmental atelectasis',
    'Hazy opacification',
    'Cavitation within consolidation',
    'Air-fluid levels in the lung',
    'Silhouette sign',
    'Reticular opacities',
    'Bronchial wall thickening',
    'Patchy infiltrates',
    'Localized hyperinflation',
    'Round pneumonia',
    'Segmental consolidation',
    'Interstitial thickening',
    'Tree-in-bud pattern',
    'Homogeneous opacification',
    'Lung abscess formation'
]

# COVID-19 Concepts
covid19_concepts = [
    'Peripheral ground-glass opacities',
    'Bilateral involvement',
    'Multilobar distribution',
    'Crazy-paving pattern',
    'Rare pleural effusion',
    'Increased density in the lung',
    'Localized or diffuse presentation',
    'Ground-glass appearance',
    'Consolidative areas',
    'Presence of nodules or masses',
    'Diffuse opacities',
    'Patchy or widespread distribution',
    'Interstitial abnormalities',
    'Absence of lobar consolidation',
    'Subpleural sparing',
    'Fibrotic streaks',
    'Thickened interlobular septa',
    'Vascular enlargement within lesions',
    'Pleural thickening',
    'Reversed halo sign'
]


# Normal Concepts
normal_concepts = [
    'Clear lung fields with no opacities',
    'Defined cardiac borders',
    'Sharp costophrenic angles',
    'Uniform vascular markings',
    'Normal mediastinal silhouette',
    'Absence of lymphadenopathy',
    'Normal bronchovascular markings',
    'No evidence of pleural thickening',
    'No abnormal lung parenchymal opacities',
    'Normal tracheobronchial tree',
    'Symmetrical diaphragmatic domes',
    'Clear hila',
    'Unremarkable soft tissues and bones',
    'No signs of pulmonary edema',
    'Absence of masses or nodules',
    'Normal aortic arch contour',
    'Lungs are well-aerated',
    'No evidence of pneumothorax',
    'Consistent radiographic density',
    'Regularly spaced rib intervals'
]


# Concept Set C = {C1, C2, ....., CN}
C = pneumonia_concepts + covid19_concepts + normal_concepts

# Embeddings from Mistral
client = MistralClient(api_key=MISTRAL_API_KEY)

embeddings_batch_response = client.embeddings(
    model="mistral-embed",
    input= C,
)

# Text embeddings t
t = []
for concept_embedding in embeddings_batch_response.data:
    t.append(torch.tensor(concept_embedding.embedding)) # Shape t[i] = [1024]


# Loading the Images
pneumonia_dir = [
    'Lung Segmentation Data/Lung Segmentation Data/Train/Non-COVID/images',
    'Lung Segmentation Data/Lung Segmentation Data/Test/Non-COVID/images',
    'Lung Segmentation Data/Lung Segmentation Data/Val/Non-COVID/images'
]

pneumonia_images = []

for dir in pneumonia_dir:
    for file in os.listdir(dir):
        pneumonia_images.append((os.path.join(dir, file), 'pneumonia'))

covid_dir = [
    'Lung Segmentation Data/Lung Segmentation Data/Train/COVID-19/images',
    'Lung Segmentation Data/Lung Segmentation Data/Test/COVID-19/images',
    'Lung Segmentation Data/Lung Segmentation Data/Val/COVID-19/images'
]

covid_images = []

for dir in covid_dir:
    for file in os.listdir(dir):
        covid_images.append((os.path.join(dir, file), 'covid'))

normal_dir = [
    'Lung Segmentation Data/Lung Segmentation Data/Train/Normal/images',
    'Lung Segmentation Data/Lung Segmentation Data/Test/Normal/images',
    'Lung Segmentation Data/Lung Segmentation Data/Val/Normal/images'
]

normal_images = []

for dir in normal_dir:
    for file in os.listdir(dir):
        normal_images.append((os.path.join(dir, file), 'normal'))

all_images = pneumonia_images + covid_images + normal_images


# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
])

# Vision Model
config = CheXagentVisionConfig(
    hidden_size=1024,
    intermediate_size=6144,
    num_hidden_layers=39,
    num_attention_heads=16,
    image_size=224,
    patch_size=14,
    hidden_act="gelu",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
    initializer_range=1e-10,
    qkv_bias=True
)

vision_model = CheXagentVisionEmbeddings(config)

count = 0
for img, label in all_images:
    count += 1
    I = Image.open(img).convert("RGB")
    I = transform(I).unsqueeze(0)
    V = vision_model(I)
    V = V.squeeze(0)
    e = []
    for concept in t:
        s = torch.nn.functional.cosine_similarity(V, concept, dim=1)
        e.append(max(s))
    #  Saving the Data
    datafile = open(f'embedding_similarities/{count}.obj', 'wb')
    data = {"image_path": img, "label": label, "e": e}
    pickle.dump(data, datafile)