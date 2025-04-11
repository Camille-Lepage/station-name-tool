import streamlit as st
import pandas as pd
from collections import defaultdict
from Levenshtein import ratio
import unicodedata
import re
import google.generativeai as genai
import json
import os
import time
from math import ceil

# Set page configuration
st.set_page_config(page_title="Station Name Processing Tool", layout="wide")

# Display title and intro
st.title("Station Name Processing Tool")
st.markdown("""
This application helps process and standardize station names from CSV data. 
The tool offers two main functions:
1. Generate station names using Gemini AI
2. Define cluster station names from similar entries

Upload your CSV file to begin processing.
""")

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase and remove accents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Title case formatting
    text = text.title()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Gemini API configuration
def configure_gemini(api_key):
    """Configure the Gemini API with the provided key"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    return model

# Function to process a batch of addresses with Gemini
def process_batch_with_gemini(model, data_batch, batch_size=10):
    """
    Process a batch of addresses and names with Gemini in a single request

    Args:
        model: The configured Gemini model
        data_batch: List of dictionaries containing 'adress' and 'remote_name'
        batch_size: Batch size to process

    Returns:
        A list of generated station names
    """
    # Prepare the prompt with examples and task explanation
    prompt = """
    Analyze the following pairs of addresses and names and determine the best station name for each.

    Context: These data represent transportation stations (bus, train, etc.) in different languages and countries.

    Here are examples of what is expected:

    Example 1:
    - Address: "Terminal Rodoviário de São Paulo, Avenida Cruzeiro do Sul, 1800, Santana"
    - Remote name: "Station São Paulo"
    - Generated name: "Sao Paulo Terminal - Santana"

    Example 2:
    - Address: "123 Main Street, New York, NY 10001"
    - Remote name: "New York Bus Station"
    - Generated name: "New York - Main Street"

    Example 3:
    - Address: "Gare du Nord, 18 Rue de Dunkerque, 75010 Paris, France"
    - Remote name: "Paris"
    - Generated name: "Paris - North Train Station"

    Example 4:
    - Address: "Av. Alberto Leal Nunes, 1240 - Alto do Balanço, Regeneração - PI, 64490-000, Brazil"
    - Remote name: "Regeneração"
    - Generated name: "Regeneracao - Alto Do Balanco"

    Example 5:
    - Address: "R. Bento Gonçalves, 813 - Centro, Cachoeira do Sul - RS, 96501-151, Brazil"
    - Remote name: "Parada Vargas"
    - Generated name: "Cachoeira Do Sul - Parada Vargas"

    Example 6:
    - Address: "R. Pref. João Costa, 283 - Centro, Unaí - MG, 38610-009, Brazil"
    - Remote name: "Unai"
    - Generated name: "Unai - Centro"

    For each station, please:
    1. Identify the city name
    2. Identify any important specific location (terminal, station, shopping mall, neighborhood, district, historical center, city center etc.)
    3. The remote_name MUST be included in the station_name:
       - If remote_name is the city name, use it as the first part and include the specific location after a dash
       - If remote_name is NOT the city name, include it after the dash as shown in Example 5

    Formatting rules:
    4. Remove all diacritics and accents from names
    5. Ensure the station_name does not exceed 10 words
    6. Translate station-related terms (Terminal, Gare, Rodoviária, etc.) to English
    7. AVOID duplicate names like "Foz Do Jordao - Foz Do Jordao"
    8. When the city name appears in both parts, use it ONCE followed by a distinctive landmark, neighborhood, or feature
    9. If no distinctive feature exists beyond the city name, use only the city name

    Examples of INCORRECT formats (DO NOT USE):
    - "Paris - Paris"
    - "Foz Do Jordao - Foz Do Jordao"
    - "Santa Isabel - Santa Isabel"
    - "São Paulo - Terminal Rodoviário Tietê"
    - "Cachoeira Do Sul - Centro" (incorrect because remote_name "Parada Vargas" is missing)

    Corresponding examples of CORRECT formats:
    - "Paris - North Station" (city + landmark)
    - "Foz Do Jordao" (city name alone when no distinctive feature)
    - "Santa Isabel - Bus Terminal" (city + feature)
    - "Sao Paulo - Tiete Bus Terminal"
    - "Cachoeira Do Sul - Parada Vargas" (city + remote_name)

    Here is the data to process:
    """

    # Add batch data to the prompt
    for i, item in enumerate(data_batch):
        address = item.get('adress', '') if isinstance(item.get('adress', ''), str) else ""
        remote_name = item.get('remote_name', '') if isinstance(item.get('remote_name', ''), str) else ""
        prompt += f"\nStation {i+1}:\n- Address: \"{address}\"\n- Remote name: \"{remote_name}\"\n"

    # Request a response in JSON format
    prompt += """
    Reply only with a JSON array where each element is an object with a single property "station_name"
    containing the generated name for each station, in the same order as the provided data.
    For example:
    [
      {"station_name": "Sao Paulo - Terminal Santana"},
      {"station_name": "New York - Main Street"},
      {"station_name": "Paris - North Train Station"},
      {"station_name": "Regeneracao - Alto Do Balanco"},
      {"station_name": "Cachoeira Do Sul - Parada Vargas"}
    ]

    """

    try:
        # Send request to Gemini
        response = model.generate_content(prompt)
        response_text = response.text

        # Find and extract the JSON
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                results = json.loads(json_str)
                # Extract only the station names and ensure they're properly cleaned
                return [clean_text(item.get("station_name", "")) for item in results]
            except json.JSONDecodeError as je:
                st.error(f"JSON decoding error: {je}")
                st.code(f"Received JSON: {json_str}")
                # Fallback: process manually if the format is incorrect
                return [""] * len(data_batch)
        else:
            st.error(f"No JSON found in the response")
            return [""] * len(data_batch)
    except Exception as e:
        st.error(f"Error with Gemini: {e}")
        return [""] * len(data_batch)

# Function to process station names with Gemini
def process_stations_with_gemini(df, api_key, batch_size=10):
    # Check for required columns
    required_columns = ['adress', 'remote_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Missing columns in CSV: {', '.join(missing_columns)}")
        # Adapt columns if necessary
        for col in missing_columns:
            # Look for alternatives (e.g., 'address' instead of 'adress')
            alternatives = {
                'adress': ['address', 'addr', 'location', 'direccion', 'adresse'],
                'remote_name': ['name', 'station_name', 'remote', 'nom', 'nombre']
            }
            found = False
            for alt in alternatives.get(col, []):
                if alt in df.columns:
                    st.info(f"Using '{alt}' instead of '{col}'")
                    df[col] = df[alt]
                    found = True
                    break
            if not found:
                st.warning(f"Creating empty column for '{col}'")
                df[col] = ""
    
    try:
        # Configure Gemini API
        model = configure_gemini(api_key)
        
        # Process data in batches
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        results = []
        total_rows = len(df)
        
        for i in range(0, total_rows, batch_size):
            end_idx = min(i + batch_size, total_rows)
            progress_text.text(f"Processing batch {i//batch_size + 1}/{ceil(total_rows/batch_size)} (rows {i+1}-{end_idx})...")
            
            # Update progress bar
            progress_bar.progress((i + 1) / ceil(total_rows/batch_size))
            
            # Prepare the batch data
            batch_data = df.iloc[i:end_idx].to_dict('records')
            
            try:
                # Process the batch with Gemini
                batch_results = process_batch_with_gemini(model, batch_data, batch_size)
                results.extend(batch_results)
                
                # Add delay to respect rate limits
                time.sleep(4)
                
            except Exception as e:
                st.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # In case of error, add empty values for this batch
                results.extend([""] * len(batch_data))
        
        # Add results to DataFrame
        df['station_name_new'] = results
        progress_text.text("Processing complete!")
        progress_bar.progress(100)
        
        return df
        
    except Exception as e:
        st.error(f"General error during processing: {e}")
        return df

# Function to define cluster station names
def define_cluster_station_names(merged_df, similarity_threshold=0.6):
    """
    Defines a representative station name for each cluster according to the following rules:
    - Case 1: If all station names in a cluster are identical, keep that name
    - Case 2: If names are similar (one is contained in the other or high similarity), take the longest one
    - Case 3: If names are different, ask the user to choose or input a custom name

    Args:
        merged_df: DataFrame containing columns remote_id, remote_name, cluster_id and station_name_new
        similarity_threshold: Similarity threshold to consider two names as similar

    Returns:
        DataFrame with updated station names
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = merged_df.copy()

    # Create a dictionary to store information by cluster
    clusters = defaultdict(list)

    # Fill the dictionary
    for _, row in df.iterrows():
        # Check for NaN values
        station_name = row['station_name_new']
        if pd.isna(station_name):
            station_name = ""

        # Get remote_name, handle NaN values
        remote_name = row.get('remote_name', '')
        if pd.isna(remote_name):
            remote_name = ""

        clusters[row['cluster_id']].append({
            'remote_id': row['remote_id'],
            'remote_name': remote_name,
            'station_name': station_name
        })

    # Dictionary to store the chosen name for each cluster
    cluster_names = {}

    # Process each cluster
    cluster_data = {}
    for cluster_id, stations in clusters.items():
        # Extract unique station names
        station_names = list(set([s['station_name'] for s in stations]))

        # Case 1: All names are identical
        if len(station_names) == 1:
            cluster_names[cluster_id] = station_names[0]
            continue

        # Case 2: Check if names are similar
        similar_names = False
        longest_name = ""

        for i, name1 in enumerate(station_names):
            for j, name2 in enumerate(station_names):
                if i != j:
                    # Check if one name is contained in the other
                    contained = (name1.lower() in name2.lower() or name2.lower() in name1.lower())

                    # Calculate Levenshtein similarity
                    similarity = ratio(name1.lower(), name2.lower())

                    if contained or similarity > similarity_threshold:
                        similar_names = True
                        # Update the longest name
                        if len(name1) > len(longest_name):
                            longest_name = name1
                        if len(name2) > len(longest_name):
                            longest_name = name2

        if similar_names:
            cluster_names[cluster_id] = longest_name
            continue

        # Case 3: Names are different, prepare data for user selection
        # Group by station name to show all remotes for each station name
        station_to_remotes = defaultdict(list)
        for station in stations:
            station_to_remotes[station['station_name']].append(station['remote_name'])
        
        # Store this cluster's data for user selection
        cluster_data[cluster_id] = {
            'station_names': list(station_to_remotes.keys()),
            'remote_names': {name: sorted(set(remotes)) for name, remotes in station_to_remotes.items()}
        }
    
    # Update station names for Case 1 and Case 2
    df['cluster_station_name'] = df['cluster_id'].map(cluster_names)
    
    # Return both the partially updated DataFrame and the clusters that need user input
    return df, cluster_data

def streamlit_cluster_selection(df, cluster_data):
    """Handle user selection of cluster names via Streamlit widgets"""
    
    if not cluster_data:
        st.success("All clusters processed automatically!")
        return df
    
    st.write("### Manual Cluster Name Selection")
    st.write("The following clusters have multiple different station names. Please select a name for each cluster:")
    
    # Dictionary to store user selections
    user_selections = {}
    
    # Create a form for all selections
    with st.form("cluster_selection_form"):
        # Process each cluster that needs user input
        for cluster_id, data in cluster_data.items():
            st.write(f"**Cluster {cluster_id}**")
            
            # Display all options with their remote names
            for i, name in enumerate(data['station_names']):
                remote_names_str = ", ".join(data['remote_names'][name])
                st.write(f"Option {i+1}: Station: {name} | Remote names: {remote_names_str}")
            
            # Create radio buttons for selection
            options = data['station_names'] + ["Enter a custom name"]
            selection = st.radio(
                f"Choose a name for Cluster {cluster_id}:",
                options=options,
                key=f"cluster_{cluster_id}"
            )
            
            # If custom name selected, show text input
            if selection == "Enter a custom name":
                custom_name = st.text_input(f"Custom name for Cluster {cluster_id}:", key=f"custom_{cluster_id}")
                user_selections[cluster_id] = custom_name
            else:
                user_selections[cluster_id] = selection
            
            st.divider()
        
        # Submit button
        submit_button = st.form_submit_button("Apply Selections")
    
    if submit_button:
        # Update DataFrame with user selections
        for cluster_id, name in user_selections.items():
            # Apply the selection to all rows in this cluster
            df.loc[df['cluster_id'] == cluster_id, 'cluster_station_name'] = name
        
        st.success("Cluster names updated successfully!")
    
    return df

# Main app tabs
tab1, tab2, tab3 = st.tabs(["Upload Data", "Generate Station Names", "Define Cluster Names"])

# Global variable to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'clustered_df' not in st.session_state:
    st.session_state.clustered_df = None

with tab1:
    st.header("Upload CSV Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.success(f"File uploaded successfully! {len(df)} rows found.")
            st.write("### Preview of uploaded data")
            st.dataframe(df.head())
            
            # Display column info
            st.write("### Available columns")
            cols = df.columns.tolist()
            st.write(", ".join(cols))
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

with tab2:
    st.header("Generate Station Names with Gemini AI")
    
    if st.session_state.df is None:
        st.warning("Please upload a CSV file in the 'Upload Data' tab first.")
    else:
        st.write("This process will use Gemini AI to generate station names based on address and remote name data.")
        
        # API key input
        api_key = st.text_input("Enter your Gemini API Key", type="password")
        
        # Batch size selection
        batch_size = st.slider("Batch size for processing", min_value=1, max_value=50, value=10)
        
        # Process button
        if st.button("Generate Station Names"):
            if not api_key:
                st.error("Please enter a valid Gemini API Key.")
            else:
                with st.spinner("Processing data with Gemini AI..."):
                    df = st.session_state.df.copy()
                    processed_df = process_stations_with_gemini(df, api_key, batch_size)
                    st.session_state.processed_df = processed_df
                    
                    # Display results
                    st.write("### Generated Station Names")
                    st.dataframe(processed_df[['remote_name', 'adress', 'station_name_new']].head(10))
                    
                    # Download button
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download processed data as CSV",
                        data=csv,
                        file_name="stations_processed.csv",
                        mime="text/csv",
                    )

with tab3:
    st.header("Define Cluster Station Names")
    
    if st.session_state.processed_df is None:
        st.warning("Please generate station names in the 'Generate Station Names' tab first, or upload a pre-processed CSV.")
        
        # Option to upload pre-processed CSV
        st.write("### Or upload a pre-processed CSV file")
        processed_file = st.file_uploader("Upload processed CSV with station_name_new and cluster_id columns", key="processed_file", type="csv")
        
        if processed_file is not None:
            try:
                processed_df = pd.read_csv(processed_file)
                
                # Check if required columns exist
                required_cols = ['remote_id', 'remote_name', 'cluster_id', 'station_name_new']
                missing_cols = [col for col in required_cols if col not in processed_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.session_state.processed_df = processed_df
                    st.success("Pre-processed file loaded successfully!")
                    st.dataframe(processed_df.head())
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    else:
        st.write("Define representative station names for each cluster")
        
        # Similarity threshold slider
        similarity = st.slider("Similarity threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
        
        # Process clusters button
        if st.button("Process Clusters"):
            with st.spinner("Processing clusters..."):
                df = st.session_state.processed_df.copy()
                
                # Check if cluster_id column exists
                if 'cluster_id' not in df.columns:
                    st.error("The dataframe doesn't have a cluster_id column. Please make sure your data includes cluster information.")
                else:
                    # Process clusters
                    partially_processed_df, clusters_needing_input = define_cluster_station_names(df, similarity)
                    
                    # Now handle the clusters that need user input
                    if clusters_needing_input:
                        st.session_state.partially_processed_df = partially_processed_df
                        st.session_state.clusters_needing_input = clusters_needing_input
                        
                        st.write("### Clusters processed automatically")
                        # Display automatically processed clusters
                        auto_clusters = set(partially_processed_df['cluster_id']) - set(clusters_needing_input.keys())
                        if auto_clusters:
                            auto_examples = partially_processed_df[partially_processed_df['cluster_id'].isin(list(auto_clusters)[:5])]
                            st.dataframe(auto_examples[['cluster_id', 'remote_name', 'station_name_new', 'cluster_station_name']].head())
                        
                        # Process clusters that need user input
                        final_df = streamlit_cluster_selection(partially_processed_df, clusters_needing_input)
                        st.session_state.clustered_df = final_df
                    else:
                        st.success("All clusters processed automatically!")
                        st.session_state.clustered_df = partially_processed_df
                    
                    # Show download button if we have results
                    if st.session_state.clustered_df is not None:
                        st.write("### Final Results")
                        st.dataframe(st.session_state.clustered_df[['cluster_id', 'remote_name', 'station_name_new', 'cluster_station_name']].head(10))
                        
                        # Download button
                        csv = st.session_state.clustered_df.to_csv(index=False)
                        st.download_button(
                            label="Download final clustered data as CSV",
                            data=csv,
                            file_name="stations_clustered.csv",
                            mime="text/csv",
                        )
