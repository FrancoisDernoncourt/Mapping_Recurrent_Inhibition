# Code for the paper “From Large-Scale Motor Neuron Recordings to Spinal Circuits: Mapping Recurrent Inhibition Across Muscles in Humans”
(bioRxiv link: TBA)

This repository contains:
 - Processing and analysis pipelines for experimental HD-EMG / MUedit spike-train data
 - Spiking network simulations of motor neuron pools and recurrent inhibition
 - A full simulation-based inference (SBI) pipeline using neural density estimation
 - Scripts to reproduce the main figures (Python + R)

A note on AI assistance:
Parts of this code were created with assistance from large language models (OpenAI GPT-o4, GPT-5.x).
All code used in the paper has been manually checked and tested, but you will notice a characteristic "ChatGPT" style in some notebooks.

If you run into issues or bugs, please reach out to François Dernoncourt
Current institutional email: francois.dernoncourt@etu.univ-cotedazur.fr
If the mail address doesn't exist anymore, please refer to the address available at François Dernoncourt's profile at https://…



# Data required to run the scripts
To keep the repository lightweight, only part of the data is stored here: the simulation parameters, the trained neural density estimators, and the posterior samples.

### Minimal data bundle
A minimal dataset allowing you to run the full analysis pipeline end-to-end (from processed data to figures) is available at:
(download link: TBA)
It includes:
 - CSV files with simulation analysis results used in the paper
 - CSV files with experimental analysis results
 - An example of experimental spike-train data (MUedit format)

### Full experimental dataset
The full experimental dataset (edited spike trains in MUedit format, Avrillon et al., 2024) can be downloaded from:
https://…

### Parameters/priors, SBI density estimator networks, posterior samples
Parameter/prior files, trained neural density estimators and posterior samples CSV files are in this repository directly
The full parameter sets used in the paper are stored under Simulation_parameters/... (as .pkl or .json files).
Note: simulator.py exposes more parameters than those reported in the paper (Table 1). Extra parameters correspond to options explored during development but not used in the final results.

### Requirements for full replication
If you want to fully reproduce all figures, you will need:
 - The simulated data batch used for SBI (summary CSV + parameter files)
 - The experimental summary CSVs
 - The saved SBI neural density estimators in Simulation_based_inference/saved_posterior_density_estimators/



# 0. Create a Python environment (Conda)
From a terminal:
 cd path/to/Mapping_Recurrent_Inhibition
 conda env create -f mapping_RI_env.yml
Then activate:
 conda activate mapping_RI_env

or, in Jupyter / VS Code, select the mapping_RI_env kernel.

The mapping_RI_env.yml file pins a Python + PyTorch + SBI + Brian2 combination



# 1. Experimental data: processing and feature extraction
End goal of this section: a single CSV summarizing experimental features per condition, ready plotting (R scripts) and simulation-based inference (Python SBI pipeline)

### 1.1 Process MUedit spike-train files
Notebook: Experimental_data_processing/process_experimental_data.ipynb
Input: .mat files in MUedit format (Avrillon et al., 2024)
Output: .h5 files containing edited and structured spike trains

This notebook can additionally run: Factor analysis, Coherence analysis, Time-frequency analysis, Power spectrum analysis.
These analyses were not used in the paper.

A single MUedit file example is provided in the minimal data bundle.

### 1.2 Build synchronization cross-histograms and features
Notebook: analyze_batch.ipynb (repository root)
Input: Set path_of_files to the folder where the .h5 outputs of step 1.1 were saved (or alternatively, the path where the experimental data was downloaded).
Output: Synchronization cross-histogram analysis results for each input file (as analysis_output.pkl files)

Configure analyzes_params:
 - Comment out the simulation section
 - Uncomment and use the experimental section
For experimental data, this notebook outputs:
 - Computes synchronization cross-histograms
 - Extracts related features (e.g. trough areas, peaks, timing features)

### 1.3 Combine experimental results into a single CSV
Notebook: Experimental_data_processing/organize_experimental_data.ipynb
Input: This script loads the analysis outputs from step 1.2 (recursively looks into the subfolders of the folder provided as input to find analysis_output.pkl files)
Output: a single summary CSV for plotting and SBI

Please double-check that subject and intensity are parsed correctly:
subject   = os.path.basename(pickle_file)[:2]   # e.g. "S1", "S2", or "P1", "P2" # or [:4] if using subject indices such as "DeFr"
intensity = extract_number(pickle_file)         # should be 10.0 or 40.0



# 2. Simulations
This section covers how to run the spiking network simulations and analyze their spike trains.
To re-run the simulations used in the paper, use the parameter files in: Simulation_parameters/
To only reproduce figures and SBI from existing simulations, you can instead use the precomputed simulation CSVs from: Mapping_Recurrent_Inhibition_minimal_data_to_run_scripts (link TBA)

### 2.1 Run simulation batches
Notebook: simulate_batch.ipynb (root)
Uses simulator.py and the SimulationParameters dataclass.
Any parameter defined in SimulationParameters can be overridden in simulate_batch.ipynb.
Parameters not set there fall back to their default values in simulator.py.

Performance note:
Simulations are faster if you have Cython and a working C/C++ toolchain installed (for Brian2’s C++ backend).
On Windows, you may need to follow Microsoft’s instructions for installing build tools.

### 2.2 Analyze simulation batches
Notebook: analyze_batch.ipynb (root)
Input: Set path_of_files to the folder containing simulation outputs from step 2.1.
Output: Analysis results for each input file (as analysis_output.pkl files)
Configure analyzes_params:
 - Comment out the experimental section
 - Uncomment the simulation section
This step computes the same cross-histogram features as for experimental data, but now for all simulated conditions. For the simulated data, it also computes other values such as firing statistics, or coherence.

### 2.3 Build a summary CSV for the full simulation batch
Notebook: Simulation_batch_general_analysis.ipynb (root)
Input: This script loads the simulation outputs from step 2.1 and analysis outputs from step 2.2
(recursively looks into the subfolders of the folder provided as input to find analysis_output.pkl, simulation_output.h5 and sim_parameters.json files)
Output: Aggregates all simulation analysis results into a single CSV.
This CSV is then used:
 - For descriptive plots
 - As input to the SBI training pipeline



# 3. Simulation-based inference (SBI)
This section describes how to:
 - Train and evaluate the neural density estimator on simulated data
 - Infer posteriors for experimental conditions
 - Run posterior predictive checks
 - Reproduce the main SBI results (Fig. 5A–B in the paper)
The code is tailored to this project’s data and feature definitions. Many parts of the code may be adapted to other use cases, but such adaptations may not be very straightforward to implement.

### 3.X Main SBI script description
Notebook: Simulation_based_inference/SBI_main_script.ipynb
This notebook:
 - Loads simulated training data (the CSV file output from step 2.3) and experimental summary data (the CSV file output from step 1.3)
 - Organizes parameters, priors, and features used for inference
 - Loads or trains neural density estimators (sbi package)
 - Perform posterior inference
 - Run posterior-predicted simulations (use inferred posteriors as inputs to the simulator)

Data and model files:
 - Preprocessed data used in the paper are provided in the minimal data bundle (experimental and simulated training data CSVs)
 - Trained neural density estimators used in the paper are stored in Simulation_based_inference/saved_posterior_density_estimators/
 - Parameter and prior definitions are stored in Simulation_parameters/

The script was used twice in the paper: once for the single-muscle case (inferring recurrent inhibition strength and high-frequency common input per muscle), once for the between-synergists case (inferring recurrent inhibition and shared common input across pairs of muscles)
The notebook is configured for the within-muscle case by default. Alternative lines of code for the between-muscle case are present but commented out and can be enabled as needed.

### 3.1 Evaluating inference on held-out simulations
Section in the notebook: "Evaluating inference on held-out simulated data (posterior estimates VS ground truth)"
Pipeline:
 - Randomly hold out ~10% of simulated examples for testing
 - Train a neural density estimator (SNPE) on the remaining 90%
 - Compute, for all held-out examples:
    - Accuracy: how close posterior modes are to ground truth
    - Calibration: whether posterior uncertainty is well-calibrated
    - Resolution: how well different ground-truth values can be distinguished
 - All this is driven from SBI_main_script.ipynb and uses the same features and parameters as the final inference.

### 3.2 Simulation-based inference on experimental data
Section in the notebook: "SBI ON EXPERIMENTAL DATA"
Pipeline:
 - A final neural density estimator network is trained on 100% of the simulated data (or a trained one is loaded)
 - SBI is performed on experimental observations:
    - For individual participants [1 condition = subject × muscle (or muscle pair) × intensity]
    - For grouped participants (experimental data of all participants pooled together) [1 condition = muscle (or muscle pair) × intensity]
 - Posterior samples are saved as CSV files as 'posterior_estimates_each_subject.pkl' and 'posterior_samples_subjects_grouped_df.pkl'

Note that if some experimental features lie far outside the distribution of simulated training data, SBI sampling can run forever. The sbi package emits warnings in such cases.

### 3.3 Posterior predictive checks
Section in SBI_main_script.ipynb: "Posterior predictive checks". Other notebooks are involved too (see pipeline)
Pipeline:
 - Load the parameters used for simulating the training data (sets the simulation parameters that are not inferred by SBI)
 - Sample parameter sets from the inferred posteriors for selected conditions
 - Use these samples as inputs to the simulator (run_simulation() function)
 - Analyze the resulting simulated spike trains and cross-histogram features (run Simulation_batch_general_analysis.ipynb as in step 2.3)
 - Compare posterior-predicted simulated features with experimental features using: Simulation_based_inference/posterior_prediction_simulated_obs_VS_experimental_obs.ipynb
You may need to adjust:
 - Paths to prior files used to fill in non-inferred simulator parameters
 - Output directories for new simulation batches and analyses

### 3.4 Reproducing Figure 5A–B (posterior plots and between-muscle / between-intensity comparisons)
Notebook: Simulation_based_inference/SBI_posteriors_figures.ipynb
Input: this notebook loads the posteriors inferred in step 3.2 (CSV file with posterior samples)
Output:
 - Posterior density plots (Fig. 5A) (also incudes per-participant posterior plots, not shown in the paper)
 - Between-muscle and between-intensity posterior comparisons (Fig. 5B)
The CSVs needed to reproduce the paper’s figures are provided under: Simulation_based_inference/saved_posterior_density_estimators/...



# 4. R figure scripts and additional analyses
All R notebooks are in R_scripts_figures_results/.

### 4.1 Fig. 2 – Proxy features (trough area, peak height, trough timing) vs ground truth parameters (recurrent inhibition strength and time constant, higher-frequency common input amplitude) (simulated data)
Notebook: R_scripts_figures_results/Proxy_features_VS_ground_truth_plots_FIG2.Rmd
Input: simulation batch CSV generated in section 2.3
Output: regression plots relating proxy features to ground-truth physiological parameters (Fig. 2)
The input file (i.e., the simulation batch) to plot are configurable inside the Rmd.

### 4.2 Fig. 5C – Recruitment threshold vs backward trough area
Notebook: R_scripts_figures_results/Plotting_experimental_data_features_FIG5.Rmd
Input: experimental data CSV (output of step 1.3)
Output:
 - plots for chosen (i.e., configurable in the notebook) experimental features (not all shown in the paper)
 - reproduces Fig. 5C (recruitment threshold vs backward trough area regression per participant)