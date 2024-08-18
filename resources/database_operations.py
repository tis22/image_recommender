import sqlite3
import os


def create_database():
    """
    Creates a SQLite database and an image_paths table if they do not exist.

    This function checks if the "database" directory exists, creates it if it doesn't,
    and then creates a SQLite database and a table named "image_paths" to store image paths.

    Returns:
        None
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
        df (pd.DataFrame): DataFrame containing image paths under a column named "Path".
        conn (sqlite3.Connection): SQLite connection object to the database.

    Returns:
        None
    """
    curs = conn.cursor()
    for file_path in df["Path"]:
        curs.execute("""INSERT OR IGNORE INTO image_paths (Path) VALUES (?);""", (file_path,))
    conn.commit()


def get_result_paths(curs, similarity_results):
    """
    Retrieves file paths for images based on their IDs from the database.

    Args:
        curs (sqlite3.Cursor): SQLite cursor object used to execute SQL queries.
        similarity_results (list): List of image IDs for which file paths are needed.

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
