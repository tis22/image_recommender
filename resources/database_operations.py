import sqlite3
import os

def create_database():
    """
    Creates a SQLite database and an image_paths table if they do not exist.
    """
    if not os.path.exists("database"):
        os.makedirs("database")

    conn = sqlite3.connect("database/bd_database.db")
    curs = conn.cursor()

    curs.execute(
        """CREATE TABLE IF NOT EXISTS image_paths 
                    (ID INTEGER PRIMARY KEY,
                    Path text);"""
    )
    conn.commit()

def save_to_db(df, conn):
    """
    Saves image paths from a DataFrame into the database.

    Args:
        df (pd.DataFrame): DataFrame containing image paths.
        conn (sqlite3.Connection): SQLite connection object.
    """
    curs = conn.cursor()
    for file_path in df["Path"]:
        curs.execute("""INSERT OR IGNORE INTO image_paths (Path) VALUES (?);""", (file_path,))
    conn.commit()

def get_result_paths(curs, similarity_results):
    """
    Retrieves file paths for images based on their IDs from the database.

    Args:
        curs (sqlite3.Cursor): SQLite cursor object.
        similarity_results (list): List of image IDs.

    Returns:
        list: List of file paths corresponding to the provided IDs.
    """
    result_paths = []
    for image_id in similarity_results:
        curs.execute(
            """SELECT Path FROM image_paths WHERE ID == (?);""",
            (image_id,),
        )
        results = curs.fetchall()
        if results:
            result_paths.append(results[0][0])
    return result_paths