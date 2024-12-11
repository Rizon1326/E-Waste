

https://github.com/user-attachments/assets/692029c1-9530-42e0-a31a-35309161f9df

# E-Waste Management Application

This application provides an interactive solution for classifying e-waste using a pre-trained Keras model. Users can input images of waste materials through a webcam or file upload, and the application predicts the type of waste and provides appropriate guidance based on its classification.

## Features
- **Image Classification**: Classifies waste materials into categories such as battery, cardboard, glass, leather, medical, metal, plastic, and wood.
- **Recyclability Guidance**: Informs whether the identified waste type is recyclable or not.
- **Sustainable Development Goals (SDG)**: Displays relevant SDG images related to the waste type.
- **Tutorial Videos**: Provides disposal tutorials for specific waste categories.
- **Interactive UI**: Built with [Streamlit](https://streamlit.io/) for a user-friendly experience.

## How It Works
1. **Input Method**: Users can either:
   - Take a picture using their webcam.
   - Upload an image file (`jpg`, `png`, `jpeg`).

2. **Image Processing**:
   - Images are resized to `224x224` pixels and normalized before being fed to the model.
   - Predictions are made using a pre-trained `keras_model3.h5` file.

3. **Output**:
   - Displays the predicted waste category with a confidence score.
   - Shows relevant SDG images.
   - Indicates recyclability and provides appropriate disposal guidance.

## Installation
### Prerequisites
- Python 3.7 or higher
- Required Python libraries:
  - `tensorflow`
  - `keras`
  - `Pillow`
  - `numpy`
  - `streamlit`
  - `python-dotenv`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/e-waste-management.git
   cd e-waste-management
## Install dependencies:
pip install -r requirements.txt
## Add your .env file:
Create a .env file in the project root and include the GEMINI_API_KEY:
GEMINI_API_KEY=your_api_key
## Place your model file (keras_model3.h5) and labels file (labels3.txt) in the root directory.
## Add supporting files:

Images: SDG images in sdg goals/ directory.
Tutorials: Disposal tutorial videos in videos/ directory.
GIFs: Recycle and non-recycle bin animations in bin_images/ directory.
## Running the Application
Start the Streamlit server by running:

streamlit run app.py
# Live video for this project
https://github.com/user-attachments/assets/0a04394a-e905-4cfd-aad5-00eb935bdb85


## Project Structure
```bash
e-waste-management/
│
├── webcame.py            # Main application script
├── keras_model3.h5       # Pre-trained model file
├── labels3.txt           # Waste classification labels
├── videos/               # Tutorial videos
├── bin_images/           # GIFs for recycling guidance
├── sdg goals/            # SDG-related images
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables


