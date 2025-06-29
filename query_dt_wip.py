# get data data from Darktable's library.db and data.db
# useful to create a LUT from DT index (used in cached images) to folder and filename

import sqlite3
import pandas as pd
from os.path import expanduser

# —2— Paths to your Darktable DBs
library_db = expanduser("~/.config/darktable/library.db")
data_db = expanduser("~/.config/darktable/data.db")

# —3— Connect & attach
conn = sqlite3.connect(library_db)
conn.execute(f"ATTACH DATABASE '{data_db}' AS data")

query = f"""
SELECT
  i.id        AS id,
  fr.folder   AS folder,
  i.filename  AS filename
FROM images         AS i
JOIN film_rolls    AS fr ON i.film_id = fr.id
GROUP BY i.id
"""

# —5— Run and grab a neat DataFrame
df = pd.read_sql_query(query, conn)
df.set_index("id", inplace=True)

conn.close()

# —6— Inspect or export
print(df)
