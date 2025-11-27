"""
Power BI Streamer - Customized for global_predictions.csv
Your actual CSV with 11 columns and 36 states
"""

import pandas as pd
import requests
import json
import time
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('powerbi_streamer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class GlobalPredictionsStreamer:
    """
    Reads from global_predictions.csv (append-only)
    Tracks position and only processes new rows
    """
    
    def __init__(self, csv_path, powerbi_endpoint, batch_size=1000):
        self.csv_path = csv_path
        self.powerbi_endpoint = powerbi_endpoint
        self.batch_size = batch_size
        self.last_row_count = 0
        
        # Your 36 states
        self.state_codes = [
            'AN', 'AP', 'AR', 'AS', 'BH', 'CH', 'CHH', 'DL', 'DNHD', 'GA',
            'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
            'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PD', 'RJ', 'SK',
            'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
        ]
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize and count existing rows"""
        if not os.path.exists(self.csv_path):
            logging.error(f"CSV file not found: {self.csv_path}")
            logging.error(f"Please ensure your prediction program is running and creating this file")
            return
        
        try:
            # Count existing rows
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            self.last_row_count = len(df)
            logging.info(f"Initialized with {self.last_row_count} existing rows")
            
            # Show date range
            if len(df) > 0:
                df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'])
                logging.info(f"Date range: {df['Timestamp_UTC'].min()} to {df['Timestamp_UTC'].max()}")
                logging.info(f"Unique states in file: {df['State_Code'].nunique()}")
            
        except Exception as e:
            logging.error(f"Error initializing: {e}")
            self.last_row_count = 0
    
    def check_and_push(self):
        """Check for new data and push to Power BI"""
        try:
            # Read CSV
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            current_row_count = len(df)
            
            # Check if new rows exist
            if current_row_count > self.last_row_count:
                # Get only new rows
                new_rows = df.iloc[self.last_row_count:]
                
                logging.info(f"Found {len(new_rows)} new rows")
                
                # Process and push
                if len(new_rows) > 0:
                    cleaned = self.prepare_data(new_rows)
                    if cleaned is not None and len(cleaned) > 0:
                        self.push_to_powerbi(cleaned)
                        self.last_row_count = current_row_count
                        
                        # Log summary
                        unique_states = new_rows['State_Code'].nunique()
                        logging.info(f"Successfully processed! Total rows: {current_row_count}, States in batch: {unique_states}")
            
            elif current_row_count < self.last_row_count:
                # File was reset/truncated
                logging.warning(f"File appears to have been reset. Restarting from beginning.")
                self.last_row_count = 0
                
        except Exception as e:
            logging.error(f"Error in check_and_push: {e}", exc_info=True)
    
    def prepare_data(self, df):
        """Prepare data for Power BI - customized for your CSV structure"""
        try:
            df_clean = df.copy()
            
            # Convert timestamp to Power BI format
            df_clean['Timestamp_UTC'] = pd.to_datetime(df_clean['Timestamp_UTC'])
            df_clean['Timestamp_UTC'] = df_clean['Timestamp_UTC'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            # Calculate prediction error metrics
            df_clean['Prediction_Error_MW'] = df_clean['Gross_Load_MW'] - df_clean['Global_Predicted_Load_MW']
            df_clean['Prediction_Error_Pct'] = (
                abs(df_clean['Prediction_Error_MW']) / df_clean['Gross_Load_MW'] * 100
            ).round(2)
            
            # Add region classification
            df_clean['Region'] = df_clean['State_Code'].apply(self.get_region)
            
            # Add full state name for map visualization (FIXES MAP ISSUE!)
            df_clean['State_Name'] = df_clean['State_Code'].apply(self.get_state_name)
            
            # Add peak hour indicator (7-9 AM or 6-9 PM)
            df_clean['Is_Peak_Hour'] = df_clean['Hour_Of_Day'].apply(
                lambda h: 1 if (7 <= h <= 9) or (18 <= h <= 21) else 0
            )
            
            # Handle NaN values
            df_clean = df_clean.fillna(0)
            
            # Select columns for dashboard (all your columns + calculated ones)
            dashboard_cols = [
                'State_Code', 'State_Name', 'Region', 'Timestamp_UTC',
                'Gross_Load_MW', 'Global_Predicted_Load_MW',
                'Prediction_Error_MW', 'Prediction_Error_Pct',
                'Hour_Of_Day', 'Day_Of_Week', 'Is_Weekend', 'Is_Peak_Hour',
                'Is_Holiday_State', 'Avg_Temp_C', 'Temp_Change_6H', 'Avg_Humidity_Pct'
            ]
            
            # Only keep columns that exist
            available_cols = [col for col in dashboard_cols if col in df_clean.columns]
            df_clean = df_clean[available_cols]
            
            return df_clean
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}", exc_info=True)
            return None
    
    def get_region(self, state_code):
        """Map state codes to regions"""
        regions = {
            'North': ['DL', 'HR', 'HP', 'JK', 'PB', 'RJ', 'UP', 'UK', 'CH', 'LA'],
            'South': ['AP', 'KA', 'KL', 'TN', 'TS', 'PD', 'AN', 'LD'],
            'East': ['BH', 'JH', 'OD', 'WB', 'AR', 'AS', 'ML', 'MN', 'MZ', 'NL', 'SK', 'TR'],
            'West': ['GJ', 'MH', 'GA', 'DNHD'],
            'Central': ['CHH', 'MP']
        }
        
        for region, states in regions.items():
            if state_code in states:
                return region
        return 'Other'
    
    def get_state_name(self, state_code):
        """Convert state code to full name for Power BI map"""
        state_names = {
            'AN': 'Andaman and Nicobar Islands',
            'AP': 'Andhra Pradesh',
            'AR': 'Arunachal Pradesh',
            'AS': 'Assam',
            'BH': 'Bihar',
            'CH': 'Chandigarh',
            'CHH': 'Chhattisgarh',
            'DL': 'Delhi',
            'DNHD': 'Dadra and Nagar Haveli and Daman and Diu',
            'GA': 'Goa',
            'GJ': 'Gujarat',
            'HP': 'Himachal Pradesh',
            'HR': 'Haryana',
            'JH': 'Jharkhand',
            'JK': 'Jammu and Kashmir',
            'KA': 'Karnataka',
            'KL': 'Kerala',
            'LA': 'Ladakh',
            'LD': 'Lakshadweep',
            'MH': 'Maharashtra',
            'ML': 'Meghalaya',
            'MN': 'Manipur',
            'MP': 'Madhya Pradesh',
            'MZ': 'Mizoram',
            'NL': 'Nagaland',
            'OD': 'Odisha',
            'PB': 'Punjab',
            'PD': 'Puducherry',
            'RJ': 'Rajasthan',
            'SK': 'Sikkim',
            'TN': 'Tamil Nadu',
            'TR': 'Tripura',
            'TS': 'Telangana',
            'UK': 'Uttarakhand',
            'UP': 'Uttar Pradesh',
            'WB': 'West Bengal'
        }
        
        return state_names.get(state_code, state_code)
    
    def push_to_powerbi(self, df):
        """Push to Power BI in batches"""
        rows = df.to_dict('records')
        total_rows = len(rows)
        
        for i in range(0, total_rows, self.batch_size):
            batch = rows[i:i + self.batch_size]
            
            try:
                response = requests.post(
                    self.powerbi_endpoint,
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps({"rows": batch}),
                    timeout=10
                )
                
                if response.status_code == 200:
                    logging.info(f"[SUCCESS] Pushed batch {i//self.batch_size + 1}: {len(batch)} rows")
                else:
                    logging.error(f"[FAILED] Batch {i//self.batch_size + 1}: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logging.error(f"Request error: {e}")
            
            # Small delay between batches
            if i + self.batch_size < total_rows:
                time.sleep(0.5)
    
    def run(self, check_interval=15):
        """Main loop - checks every 15 seconds"""
        logging.info("=" * 60)
        logging.info("Global Predictions Streamer Started")
        logging.info(f"CSV: {self.csv_path}")
        logging.info(f"Check interval: {check_interval}s")
        logging.info(f"Monitoring {len(self.state_codes)} states")
        logging.info("=" * 60)
        
        try:
            while True:
                self.check_and_push()
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logging.info("Stopped by user")


def main():
    """Main execution"""
    
    # ============ CONFIGURATION ============
    # UPDATE THESE WITH YOUR ACTUAL VALUES:
    
    CSV_PATH = r"C:\Users\DEV\Desktop\Minor Project\Project\Data\stream_predictions_csv\global_predictions. csv"
    POWERBI_URL = "https://api.powerbi.com/beta/b87386c8-9083-4a27-9ddf-63a3dfa33850/datasets/08f43b30-e66f-4941-9351-e0c4516777e4/rows?experience=power-bi&key=zLN9phs607sCpt5KExeaI06n9A8nkDEOEQ9YLHtUuJkX5PUh%2FJRgtEAYX0PKgHM9xmmtKkgx%2FoZ2P85akOMvnA%3D%3D"  # ← Your Power BI push URL
    CHECK_INTERVAL = 15  # Check every 15 seconds (matches your append interval)
    
    # =====================================
    
    # Validation
    if "YOUR_POWER_BI" in POWERBI_URL:
        logging.error("")
        logging.error("⚠ CONFIGURATION REQUIRED ⚠")
        logging.error("")
        logging.error("Please update POWERBI_URL with your actual streaming dataset URL")
        logging.error("Get it from: Power BI → Workspace → Your Dataset → Settings → Push URL")
        logging.error("")
        return
    
    if not os.path.exists(CSV_PATH):
        logging.error(f"")
        logging.error(f"⚠ CSV FILE NOT FOUND ⚠")
        logging.error(f"")
        logging.error(f"Looking for: {CSV_PATH}")
        logging.error(f"Please verify:")
        logging.error(f"  1. Path is correct")
        logging.error(f"  2. Your prediction program is running")
        logging.error(f"  3. File has been created")
        logging.error(f"")
        return
    
    # Create and run streamer
    streamer = GlobalPredictionsStreamer(CSV_PATH, POWERBI_URL)
    streamer.run(check_interval=CHECK_INTERVAL)


if __name__ == "__main__":
    main()