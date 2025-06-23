import sqlite3
from datetime import datetime
import requests
from logger import logger

class DatabaseHandler:
    def __init__(self, db_path='sackbag_counter.db'):
        self.db_path = db_path
        self._create_table()
        self.api_url = "http://exhibitapi.ttpltech.in/sackbagcounter"  # URL for posting the data

    def _create_table(self):
        # Connect to the database and create the table if it doesn't exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS SackBagCounter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    in_count BOOLEAN NOT NULL,
                    out_count BOOLEAN NOT NULL,
                    status BOOLEAN NOT NULL 
                );
            ''')
            conn.commit()

    def insert_crossing(self, in_count, out_count, camera_id="SB01"):
        # Record the current date and time in the specified format
        datetime_now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        
        # Insert the crossing data into the database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO SackBagCounter (camera_id, datetime, in_count, out_count, status)
                VALUES (?, ?, ?, ?, FALSE);
            ''', (camera_id, datetime_now, in_count, out_count))
            conn.commit()
        logger.info(f"Database entry added: IN = {in_count}, OUT = {out_count}, Camera ID = {camera_id}")

    def post_to_api(self, entry):
        """
        Post a single entry to the API.
        """
        data = {
            "camera_id": entry["camera_id"],
            "datetime": entry["datetime"],
            "in_count": entry["in_count"],
            "out_count": entry["out_count"],
            "status": entry["status"]
        }

        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()  # Raise exception for HTTP errors
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error posting data to API: {e}")
            return False

    def update_status(self, entry_id):
        """
        Update the status to True after the API post is successful.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE SackBagCounter
                SET status = TRUE
                WHERE id = ?
            ''', (entry_id,))
            conn.commit()
        print(f"Status updated to True for entry {entry_id}")

    def post_pending_entries(self):
        """
        Find all entries where status is False and post them to the API.
        Once successfully posted, update the status to True.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, camera_id, datetime, in_count, out_count, status
                FROM SackBagCounter
                WHERE status = FALSE
            ''')
            pending_entries = cursor.fetchall()

            for entry in pending_entries:
                entry_dict = {
                    "id": entry[0],
                    "camera_id": entry[1],
                    "datetime": entry[2],
                    "in_count": entry[3],
                    "out_count": entry[4],
                    "status": bool(entry[5])
                }

                # Try posting to the API
                if self.post_to_api(entry_dict):
                    # If successful, update the status
                    self.update_status(entry[0])
