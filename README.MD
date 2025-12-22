# **Cross-Dataset Transfer Learning with Explainable AI for Heart Disease Prediction**

## **Project Description**

This project develops a transfer learning framework to enhance heart disease prediction models by leveraging knowledge from multiple datasets. The framework aims to mitigate overfitting, improve generalizability, and ensure model transparency using Explainable AI (XAI) techniques.

The project involves fine-tuning models on large and small datasets, utilizing XGBoost and TabNet models, and employing SHAP for interpretability.

## **Technologies Used**

* **Python**
* **XGBoost**
* **TabNet**
* **SHAP (SHapley Additive exPlanations)**
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**
* **Keras & TensorFlow**
* **PyTorch**

## **Installation**

To get started with this project, follow the steps below to set up the environment:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd CardioTransferX
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   * **On Windows**:

     ```bash
     venv\Scripts\activate
     ```
   * **On macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

**Dependencies (in `requirements.txt`):**
```text
keras==3.12.0
keras-tuner==1.4.8
lightgbm==4.6.0
lime==0.2.0.1
MarkupSafe==2.1.5
matplotlib==3.10.8
numpy==1.26.2
opencv-python==4.11.0.86
opt_einsum==3.4.0
optree==0.18.0
optuna==4.6.0
pandas==2.3.3
pandas-profiling==3.1.0
pillow==11.3.0
PyQt5==5.15.10
pytorch-tabnet==4.1.0
requests==2.28.2
scikit-learn==1.1.3
scipy==1.10.0
seaborn==0.11.2
shap==0.50.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.2.0
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.66.1
ultralytics==8.3.161
ultralytics-thop==2.0.14
xgboost==3.1.2
imbalanced-learn==0.10.0
joblib==1.1.0
pytest==7.1.2
pyyaml==6.0
```

## **File Structure**
```sh
CardioTransferX/
│
├── CardioTransferX_Main/
│   ├── Heart_disease_cleveland
│   ├── Cleveland+Hungary+VA_long_beach+Switzerland
│   ├── Main.ipynb  # Main script for training and testing
│
├── CardioTransfer-X_AblationStudy(FeatureSelection)/
│   ├── AblationStudy(FeatureSelection)
├── CardioTransfer-X_AblationStudy(FeatureSelection_and_SMOTE)/
│   ├── AblationStudy(FeatureSelection_and_SMOTE)
├── CardioTransfer-X_AblationStudy(SMOTE)/
│   ├── AblationStudy(SMOTE)
├── requirements.txt
└── README.md
```

## **Datasets**

This project uses two primary datasets for heart disease prediction:

* **Cleveland Dataset**: A widely-used dataset for heart disease prediction containing various medical features.
* **Multi-Hospital Dataset**: A more diverse dataset collected from multiple hospitals to improve the model's generalizability across different populations.

Both datasets are be placed in all folders within the project directory.

## **Usage**

The main script for training and evaluation is `Main.ipynb`. This script handles all tasks, including model training, data splitting, and testing. Simply open and run the Jupyter notebook to execute the following steps:

1. **Import all libraries**: The core libraries will be imported for usage.
1. **Data  Preprocessing**: The datasets will be loaded from the folders and then all preprocessing steps will be carried out.
2. **Data split and class imbalance handling**: The script handles the 80%/10%/10% data split then uses SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution.
3. **Train the models**: The models (XGBoost, TabNet and MLP) will be pre-trained and fine-tuned on both datasets.
4. **Evaluate the models**: The models will be evaluated on test data, and results will be displayed as SHAP plots and ROC-AUC curves.

### Running the Script

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `Main.ipynb` in the Jupyter interface.

3. Run the cells sequentially to train and evaluate the models.

## **Results**

The project generates the following evaluation plots:

* **SHAP Plots**: Visualizations of feature importance and model explainability.
* **ROC-AUC Curve**: Performance of the model in terms of true positive rate and false positive rate.

## **Acknowledgements**

We thank the authors of the datasets for their contributions.