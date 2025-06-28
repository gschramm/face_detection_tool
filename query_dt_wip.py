import sqlite3
import pandas as pd
from os.path import expanduser

# —1— Define your tags here. Can be any length (1…n)
desired_tags = [
    "person|HappyHedgehog",
    "person|CuriousCat",
    # …add as many as you like
]

# —2— Paths to your Darktable DBs
library_db = expanduser("~/.config/darktable/library.db")
data_db = expanduser("~/.config/darktable/data.db")

# —3— Connect & attach
conn = sqlite3.connect(library_db)
conn.execute(f"ATTACH DATABASE '{data_db}' AS data")

# —4— Build the IN-list placeholders dynamically
placeholders = ", ".join("?" for _ in desired_tags)

query = f"""
SELECT
  fr.folder   AS folder,
  i.filename  AS filename
FROM images         AS i
JOIN film_rolls    AS fr ON i.film_id = fr.id
JOIN tagged_images AS ti ON ti.imgid  = i.id
JOIN data.tags     AS t  ON ti.tagid  = t.id
WHERE t.name IN ({placeholders})
GROUP BY i.id
HAVING COUNT(DISTINCT t.name) = {len(desired_tags)}
"""

# —5— Run and grab a neat DataFrame
df = pd.read_sql_query(query, conn, params=desired_tags)

conn.close()

# —6— Inspect or export
print(df)
# or df.to_csv("matched_images.csv", index=False)
