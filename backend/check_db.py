from sqlalchemy import create_engine
import pandas as pd

def main():
    try:
        # Connect to database
        engine = create_engine('postgresql://bisnet0:RG4J8^%*TWjA*977Y40T81B2@localhost:5432/episcope_db')
        
        # Query table structure
        columns = pd.read_sql(
            'SELECT column_name FROM information_schema.columns WHERE table_name = \'cleaned_arboviroses_cases\'',
            engine
        )
        print('\nColumns in database:', columns['column_name'].tolist())
        
        # Query sample data
        df = pd.read_sql('SELECT * FROM cleaned_arboviroses_cases LIMIT 5', engine)
        print('\nSample data:')
        print(df)
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
