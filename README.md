# NEMO Visitor Prediction Dashboard

This project provides a comprehensive solution for predicting and analyzing visitor patterns at the NEMO Science Museum in Amsterdam. It includes data processing pipelines, machine learning models, and an interactive Streamlit dashboard for visualization and predictions.

## Project Structure

```
AA_NEMO/
├── Data_Sources/        # Raw and processed data files
│   ├── Data_Raw/       # Original data files from Google Drive
│   └── Data_Cleaned/   # Processed and cleaned datasets, examples are also in Google Drive
├── Models/             # Trained machine learning models
├── DashBoards/         # Streamlit dashboard application
│   └── NEMO_Dashboard.py  # Main dashboard application
└── requirements.txt    # Project dependencies
```

## Data Setup (Required)

Before running the application, you need to set up the data files:

1. Download the data files from the provided Google Drive link
2. Create the following directory structure in your project:
   ```
   AA_NEMO/Data_Sources/
   ├── Data_Raw/      # Place the raw data files here
   └── Data_Cleaned/  # Place the cleaned data files here
   ```
3. Place the downloaded files in their respective directories as shown above

⚠️ **Important**: The application will not work without these data files properly placed in the correct directories.

## Features

- Interactive dashboard for visitor prediction and analysis
- Historical visitor data visualization
- Machine learning models for visitor prediction
- Factor analysis for different visitor categories:
  - Student groups
  - Educational visits (VO)
  - Primary education (PO)
  - International recreational visits
  - Domestic recreational visits

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd AA_NEMO
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Streamlit Dashboard

1. Ensure you're in the project directory and your virtual environment is activated

2. Launch the Streamlit app:
```bash
cd AA_NEMO
streamlit run DashBoards/NEMO_Dashboard.py
```

3. The dashboard will open in your default web browser at `http://localhost:8501`

## Dependencies

The project requires the following main packages:
- pandas (2.1.3)
- numpy (1.26.4)
- scikit-learn (1.3.2)
- streamlit (1.28.1)
- plotly (5.17.0)
- matplotlib (3.8.0)
- holidays (0.20.0)
- openpyxl (3.1.5)

## Data Sources

The `Data_Sources` directory contains:
- Raw visitor data
- Processed and cleaned datasets
- Data processing scripts and notebooks

## Models

The `Models` directory contains:
- Trained machine learning models for visitor prediction
- Model evaluation metrics and results
- Model configuration files

## Dashboard Components

The Streamlit dashboard (`NEMO_Dashboard.py`) provides:
- Real-time visitor predictions
- Historical data visualization
- Interactive charts and graphs
- Factor analysis for different visitor categories
- Customizable date ranges and parameters

## License

This project is licensed under the included LICENSE file.

## Contributing

Please read the contribution guidelines before submitting pull requests.

## Support

For questions or issues, please open an issue in the repository.

## Git Workflow for Beginners

Before making any changes to the code, always follow these steps:

1. **Pull the latest changes** (do this every time before starting work):
   ```bash
   git pull origin main
   ```
   This ensures you have the most recent version of the code.

2. **Make your changes** to the code

3. **Save your changes to Git**:
   ```bash
   # Check which files you modified
   git status
   
   # Add your changes to Git
   git add .
   
   # Save your changes with a descriptive message
   git commit -m "Brief description of what you changed"
   ```

4. **Push your changes** so others can see them:
   ```bash
   git push origin main
   ```

⚠️ **Important Tips**:
- Always pull before starting work to avoid conflicts
- If you get any error messages during pull/push, ask for help before proceeding
- Make sure to include a clear commit message describing what you changed
- If you're unsure about any Git commands, it's better to ask for help than risk losing work 