
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

db_user = 'postgres'
db_password =quote_plus('root')
db_host ='localhost'
db_port ='5432'
db_name='call_agent'


xl_path = 'cx_details.xlsx'
SHEET_NAME =0
TABLE_NAME ='my_table'

data = pd.read_excel(xl_path,sheet_name=SHEET_NAME)

data.columns=(data.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_", regex=False)
              )

print('column name: ',list(data.columns))

engine = create_engine(
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",echo=True
)

data.to_sql(name=TABLE_NAME,
            con=engine,
            if_exists='replace',
            index=False)










