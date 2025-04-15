# Import Flask framework and necessary functions for rendering templates and handling HTTP requests.
from flask import Flask, render_template, request  # type: ignore

# Import the os module for interacting with the operating system (e.g., directory creation).
import os
# Import the uuid module to generate unique identifiers.
import uuid
# Import numpy for numerical operations.
import numpy as np
# Import pandas for data processing, particularly for reading CSV files.
import pandas as pd
# Import all objects from the utilities module (e.g., RollingDefectAnalysis, generate_response_format, generate_content_openai).
from utilities import * 

# The following commented-out imports are an alternative explicit import of specific functions or classes from utilities.
# from utilities import RollingDefectAnalysis, generate_response_format, generate_content_openai

# Initialize the Flask application instance.
app = Flask(__name__)

# Define a route for the root URL ('/').
@app.route('/')
def index() -> str:
    """
    Render the main upload page.

    Returns:
        str: Rendered HTML content for 'upload.html'.
    """
    # Render and return the 'upload.html' template when the root URL is accessed.
    return render_template('upload.html')


# Define a route for processing file uploads via a POST request at the '/upload' URL.
@app.route('/upload', methods=['POST'])
def upload() -> str:
    """
    Process the uploaded CSV file along with form parameters, perform signal analysis,
    generate plots, and render the display page with the report and images.

    Workflow:
      1. Validate mandatory file and form inputs.
      2. Process the CSV file:
         - Ensure the CSV contains exactly one column.
         - Compute the sum of the CSV values.
         - Create a unique subfolder inside the static folder.
      3. Process signal and filter parameters from form inputs.
      4. Configure the analysis parameters and generate plots using a RollingDefectAnalysis object.
      5. Compose a prompt for OpenAI report generation based on extracted features and frequency peaks.
      6. Generate the analysis report.
      7. Render the display page with the generated images and report details.

    Returns:
        str: Rendered HTML content for 'display.html' if processing succeeds; otherwise, renders 'upload.html'
             with an error message.
    """
    # --- Validate Mandatory Inputs ---
    # Retrieve the uploaded file (if any) from the form using the key 'file'.
    file = request.files.get('file')
    # Check if the file is missing or its filename is empty.
    if not file or file.filename == '':
        # Set an error message indicating that a CSV file must be selected.
        error = "Please select a CSV file."
        # Render the upload template again with the error message.
        return render_template('upload.html', error=error)
    
    # Define a list of required form field names.
    required_fields = [
        'rpm', 'sampling_frequency', 
        'bpfi_frequency', 'bpfo_frequency', 
        'ftf_frequency', 'bsf_frequency', 'filter_type'
    ]
    # Check which required fields are missing by filtering those that are not present in the form data.
    missing_fields = [field for field in required_fields if not request.form.get(field)]
    # If any required fields are missing...
    if missing_fields:
        # Create an error message listing the missing fields.
        error = f"Please fill in the mandatory fields: {', '.join(missing_fields)}."
        # Render the upload template with the error message.
        return render_template('upload.html', error=error)
    
    # --- Process the CSV File ---
    try:
        # Reset the file pointer to the beginning of the file.
        file.seek(0)
        # Read the CSV file into a pandas DataFrame.
        df = pd.read_csv(file)
        # Check if the DataFrame contains exactly one column.
        if len(df.columns) != 1:
            # If not, set an error message indicating the CSV should have one column only.
            error = "Error: CSV file should contain exactly one column."
            # Render the upload template with the error message.
            return render_template('upload.html', error=error)

        # Determine the path for the static folder (for saving plot images) using the Flask app's root path.
        static_folder = os.path.join(app.root_path, 'static')
        # Create the static folder if it does not exist.
        os.makedirs(static_folder, exist_ok=True)

        # Generate a unique identifier using UUID, then convert it to a string.
        subfolder = str(uuid.uuid4())
        # Create the full path for a new subfolder within the static folder using the generated UUID.
        subfolder_path = os.path.join(static_folder, subfolder)
        # Create the subfolder for storing this project's images.
        os.makedirs(subfolder_path)
    except Exception as e:
        # If any error occurs during CSV processing, create an error message with the exception details.
        error = f"Error processing CSV file: {e}"
        # Render the upload template with the CSV processing error message.
        return render_template('upload.html', error=error)
    
    # --- Process Signal & Filter Parameters ---
    try:
        # Convert the RPM form value to an integer.
        rpm = int(request.form.get("rpm"))
        # Convert the sampling frequency form value to an integer.
        sampling_frequency = int(request.form.get("sampling_frequency"))
        # Convert the BPFI frequency form value to a float.
        bpfi_frequency = float(request.form.get("bpfi_frequency"))
        # Convert the BPFO frequency form value to a float.
        bpfo_frequency = float(request.form.get("bpfo_frequency"))
        # Convert the FTF frequency form value to a float.
        ftf_frequency = float(request.form.get("ftf_frequency"))
        # Convert the BSF frequency form value to a float.
        bsf_frequency = float(request.form.get("bsf_frequency"))
        # Convert the downsample factor form value to an integer.
        downsample_factor = int(request.form.get("downsample_factor"))
    except Exception as e:
        # If there is an error while converting numerical parameters, set an error message with details.
        error = f"Error processing numerical parameters: {e}"
        # Render the upload template with the numerical parameter error message.
        return render_template('upload.html', error=error)

    # Retrieve the filter type from the form.
    filter_type = request.form.get("filter_type")
    # Check if the "use_decibel" checkbox is on; this will be True if the checkbox was checked.
    use_decibel = (request.form.get("use_decibel") == "on")
    # Check if the "use_walch" checkbox is on; similar logic as above.
    use_walch   = (request.form.get("use_walch") == "on")

    # Generate a short project name by splitting the UUID subfolder name and taking the first segment.
    project = subfolder.split('-')[0]
    # Create a configuration dictionary with all necessary parameters for signal analysis.
    config = {
        "signal": df.values.ravel(),              # Flatten the CSV data to a 1D array.
        "fs": sampling_frequency,                  # Sampling frequency.
        "rpm": rpm,                                # Rotational speed in RPM.
        "bpfi": bpfi_frequency,                    # BPFI frequency.
        "bpfo": bpfo_frequency,                    # BPFO frequency.
        "ftf": ftf_frequency,                      # FTF frequency.
        "bsf": bsf_frequency,                      # BSF frequency.
        "title": project,                          # Title for the project based on the project name.
        "peak_dividant": 50,                       # A divisor to help identify significant frequency peaks.
        "use_walch": use_walch,                    # Boolean flag to use Welch's method.
        "filter_type": filter_type,                # The chosen filter type.
        "root_folder": subfolder_path,             # The path to save output images.
        "downsample_factor": downsample_factor,    # Downsample factor for signal processing.
    }
    
    # --- Generate Plots using RollingDefectAnalysis ---
    # Create an instance of RollingDefectAnalysis using the configuration dictionary (unpacking with **).
    obj = RollingDefectAnalysis(**config)
    # Generate the time domain plot for the first 2 seconds, and then close the plot.
    obj.plot_time(seconds=2, close=True)
    # Generate the frequency domain plot for the raw signal with the option to convert to decibels, then close the plot.
    obj.plot_frequency(db=use_decibel, signal_type='raw', close=True)
    # Generate the frequency domain plot for the filtered signal.
    obj.plot_frequency(db=use_decibel, signal_type='filtered', close=True)
    # Generate the frequency domain plot for the envelope signal.
    obj.plot_frequency(db=use_decibel, signal_type='envelope', close=True)
    # Generate the spectrogram for the raw signal.
    obj.plot_spectrogram(signal_type="raw", close=True)
    # Generate the spectrogram for the filtered signal.
    obj.plot_spectrogram(signal_type="filtered", close=True)
    # Generate the spectrogram for the envelope signal.
    obj.plot_spectrogram(signal_type="envelope", close=True)

    # --- Extract Analysis Information ---
    # Retrieve the frequency peaks identified during analysis from the RollingDefectAnalysis object.
    freq_information = obj.freq_peaks
    # Extract statistical features of the signal (e.g., RMS, kurtosis) using the analysis object.
    features = obj.extract_features()

    # Define the list of image filenames (constructed based on project name and analysis parameters).
    image_filenames = [
        f"{project}_time_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}.png",
        f"{project}_freq_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}_raw.png",
        f"{project}_freq_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}_filtered.png",
        f"{project}_freq_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}_envelope.png",
        f"{project}_spectrogram_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}_raw.png",
        f"{project}_spectrogram_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}_filtered.png",
        f"{project}_spectrogram_{'walch' if use_walch else 'rfft'}_{obj.filter_type.lower()}_envelope.png"
    ]
    
    # Build a list of URLs to the images by combining the subfolder name and each filename.
    image_urls = [f"{subfolder}/{filename}" for filename in image_filenames]

    # --- Compose the Prompt for Report Generation ---
    # Define a system prompt for the OpenAI report generation.
    system_prompt = "You are a helpful assistant, replying in json"
    # Construct a detailed user prompt by interpolating various analysis parameters and extracted features.

    downsampling_comment = f"""
    The signal first downsampled by a factor of {downsample_factor}.
    So the new signal has a sampling frequency of {int(sampling_frequency/downsample_factor)} Hz.
    The signal filter type is {obj.filter_type.upper()}{'.' if obj.filter_type==filter_type else f", which is different from the one selected initially ({filter_type}) since the {filter_type} resulted in all zero or nan filtered signal."}"
    """

    freq_comment = f"""
    The signal sampling frequency is {sampling_frequency} Hz.
    The signal filter type is {obj.filter_type.upper()}{'.' if obj.filter_type==filter_type else f", which is different from the one selected initially ({filter_type}) since the {filter_type} resulted in all zero or nan filtered signal."}"
    """

    user_prompt = f"""
    Vibration signals were recorded from a rotating machine at {rpm} RPM (i.e., ~{int(rpm/60)} Hz). 
    Given the following information, identify potential rolling element bearing defects.

    {freq_comment if downsample_factor==1 else downsampling_comment}
    
    Characteristic defect frequencies:
    - BPFI: {np.round(bpfi_frequency * rpm/60, 2)} Hz
    - BPFO: {np.round(bpfo_frequency * rpm/60, 2)} Hz
    - FTF: {np.round(ftf_frequency * rpm/60, 2)} Hz
    - BSF: {np.round(bsf_frequency * rpm/60, 2)} Hz
    
    Sampling frequency: {sampling_frequency} Hz.
    Filter applied: {filter_type}.
    
    Extracted features:
    - RMS: {np.round(features['rms'], 2)}
    - Kurtosis: {np.round(features['kurtosis'], 2)}
    - Crest Factor: {np.round(features['crest_factor'], 2)}
    - Skew: {np.round(features['skew'], 2)}
    - Impulse: {np.round(features['impulse'], 2)}
    
    Frequency peaks for analysis of the main frequencies and harmonics and modulations:
    - Raw: [{round_freqs(freq_information['raw'])}]
    - Filtered: [{round_freqs(freq_information['filtered'])}]
    - Envelope: [{round_freqs(freq_information['envelope'])}]
    
    Provide a large one-paragraph analysis for each defect:
    - BPFI
    - BPFO
    - FTF
    - BSF

    Then, provide a summary paragraph of your findings.
    In your responses, include details about the signal extracted features, frequency peaks, harmonics, and modulations.
    Do not fabricate details; base your report solely on the provided data.
    """
    
    # Define keys for the analysis report that will be returned.
    keys = ['BPFI', 'BPFO', 'FTF', 'BSF', 'Summary']
    # Generate a response format (i.e., JSON schema) using the keys and descriptions for each key.
    response_format = generate_response_format(keys, [f"One paragraph analysis for {k}" for k in keys])
    # Set the temperature parameter for the OpenAI API (controls randomness of output).
    temperature = 0.1

    # Call the OpenAI API-based function to generate the analysis report using the provided prompt and response format.
    report_json = generate_content_openai(
        user_prompt,
        model="gpt-4o",
        system_prompt=system_prompt,
        response_format=response_format,
        temperature=temperature
    )

    # --- Render the Display Page with Results ---
    # Render the 'display.html' template with all the extracted information and generated image URLs.
    return render_template(
        'display.html',
        image_urls=image_urls,                    # List of image URLs.
        rms=features['rms'],                      # Extracted RMS value.
        kurtosis=features['kurtosis'],            # Extracted kurtosis.
        crest_factor=features['crest_factor'],    # Extracted crest factor.
        skews=features['skew'],                   # Extracted skew value.
        impulse=features['impulse'],              # Extracted impulse value.
        bpfi_report=report_json['BPFI'],            # Analysis report for BPFI.
        bpfo_report=report_json['BPFO'],            # Analysis report for BPFO.
        ftf_report=report_json['FTF'],              # Analysis report for FTF.
        bsf_report=report_json['BSF'],              # Analysis report for BSF.
        summary_report=report_json['Summary'],      # Summary of the overall analysis.
        rpm=rpm,                                  # Rotational speed.
        sampling_frequency=sampling_frequency,    # Sampling frequency.
        bpfi_frequency=bpfi_frequency,            # BPFI frequency.
        bpfo_frequency=bpfo_frequency,            # BPFO frequency.
        ftf_frequency=ftf_frequency,              # FTF frequency.
        bsf_frequency=bsf_frequency,              # BSF frequency.
        filter_type=obj.filter_type,                  # Type of filter applied.
        use_decibel=use_decibel,                  # Boolean flag for decibel conversion.
        use_walch=use_walch,                       # Boolean flag for using Welch's method.
        downsample_factor=downsample_factor       # Downsample factor.
    )


# Check if this script is executed directly (not imported) and run the Flask app.
if __name__ == '__main__':
    # Run the Flask application on all network interfaces at port 8080 with debugging enabled.
    app.run(host='0.0.0.0', port=8080, debug=True)
