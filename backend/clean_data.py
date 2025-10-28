# backend/clean_data.py (VERSÃO 4 - Sem Vazamento / Undersampling)

import pandas as pd
from sqlalchemy import create_engine, text
import os
import time
from dotenv import load_dotenv

print("Iniciando processo de limpeza (v4 - Sem Vazamento / Undersampling)...")
start_time = time.time()

# 1. Conectar ao banco
load_dotenv()  # Garantir que o .env seja carregado
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_NAME = os.getenv('POSTGRES_DB')
print(f"Tentando conectar com: user={DB_USER}, db={DB_NAME}")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}"
engine = create_engine(DB_URL)
CHUNK_SIZE = 100000

TEMP_TABLE_DENGUE = 'temp_clean_dengue'
TEMP_TABLE_ZIKA = 'temp_clean_zika'
TEMP_TABLE_CHIK = 'temp_clean_chikungunya'
FINAL_TABLE = 'cleaned_arboviroses_cases'

try:
    # --- ETAPA 1: Pré-calcular Mediana Global ---
    print("Pré-calculando a mediana da idade global...")
    idade_series = pd.read_sql_query('SELECT "nu_idade_n" FROM raw_arboviroses_cases', engine)
    global_idade_median = pd.to_numeric(idade_series['nu_idade_n'], errors='coerce').median()
    print(f"Mediana da idade global calculada: {global_idade_median}")
    del idade_series

    # --- ETAPA 2: Limpar em Chunks e Salvar em Tabelas Temporárias ---
    query = 'SELECT * FROM raw_arboviroses_cases'
    reader = pd.read_sql(query, engine, chunksize=CHUNK_SIZE)
    
    is_first_chunk = { TEMP_TABLE_DENGUE: True, TEMP_TABLE_ZIKA: True, TEMP_TABLE_CHIK: True }

    print("Iniciando limpeza em chunks para tabelas temporárias...")
    for i, df_raw in enumerate(reader):
        print(f"Processando chunk {i+1} ({len(df_raw)} registros)...")
        
        # --- MUDANÇA: 'criterio' e 'resul_ns1' REMOVIDOS daqui ---
        symptom_columns = [
            'febre', 'mialgia', 'cefaleia', 'exantema', 'vomito', 'nausea',
            'dor_costas', 'conjuntvit', 'artrite', 'artralgia', 'petequia_n',
            'leucopenia', 'dor_retro'
        ]
        relevant_cols = symptom_columns + ['cs_sexo', 'nu_idade_n', 'doenca_alvo']
        # --- FIM DA MUDANÇA ---
        
        cols_to_select = [col for col in relevant_cols if col in df_raw.columns]
        df = df_raw[cols_to_select].copy()

        df.dropna(subset=['doenca_alvo'], inplace=True)
        if df.empty: continue

        for col in symptom_columns:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: 1 if x == 1 else 0)

        df['sexo_encoded'] = df['cs_sexo'].map({'M': 0, 'F': 1}).fillna(-1)
        df['idade'] = pd.to_numeric(df['nu_idade_n'], errors='coerce')
        df['idade'] = df['idade'].fillna(global_idade_median) 

        # --- MUDANÇA: Codificação de diagnóstico REMOVIDA ---
        # A seção 'ns1_encoded' e 'criterio_num' foi deletada.
        # --- FIM DA MUDANÇA ---
        
        target_map = {'zika': 0, 'dengue': 1, 'chikungunya': 2}
        target_column = 'doenca_alvo'
        df['target_encoded'] = df[target_column].map(target_map)
        df.dropna(subset=['target_encoded'], inplace=True)
        df['target_encoded'] = df['target_encoded'].astype(int)
        
        # --- MUDANÇA: 'ns1_encoded' e 'criterio_' REMOVIDOS daqui ---
        final_feature_columns = symptom_columns + ['sexo_encoded', 'idade']
        # --- FIM DA MUDANÇA ---

        for col in final_feature_columns:
            if col not in df.columns:
                df[col] = 0
        df_clean = df[final_feature_columns + [target_column, 'target_encoded']]

        # --- ETAPA 3: Salvar o chunk limpo na tabela temp correta ---
        df_dengue = df_clean[df_clean['doenca_alvo'] == 'dengue']
        df_zika = df_clean[df_clean['doenca_alvo'] == 'zika']
        df_chik = df_clean[df_clean['doenca_alvo'] == 'chikungunya']

        if not df_dengue.empty:
            df_dengue.to_sql(TEMP_TABLE_DENGUE, engine, if_exists='replace' if is_first_chunk[TEMP_TABLE_DENGUE] else 'append', index=False)
            is_first_chunk[TEMP_TABLE_DENGUE] = False
        if not df_zika.empty:
            df_zika.to_sql(TEMP_TABLE_ZIKA, engine, if_exists='replace' if is_first_chunk[TEMP_TABLE_ZIKA] else 'append', index=False)
            is_first_chunk[TEMP_TABLE_ZIKA] = False
        if not df_chik.empty:
            df_chik.to_sql(TEMP_TABLE_CHIK, engine, if_exists='replace' if is_first_chunk[TEMP_TABLE_CHIK] else 'append', index=False)
            is_first_chunk[TEMP_TABLE_CHIK] = False

    print("...Limpeza em chunks concluída.")

    # --- ETAPA 4: Encontrar o "Número Mágico" (min_count) ---
    print("Calculando contagem mínima para balanceamento...")
    count_dengue = pd.read_sql(f'SELECT COUNT(*) FROM {TEMP_TABLE_DENGUE}', engine).iloc[0,0]
    count_zika = pd.read_sql(f'SELECT COUNT(*) FROM {TEMP_TABLE_ZIKA}', engine).iloc[0,0]
    count_chik = pd.read_sql(f'SELECT COUNT(*) FROM {TEMP_TABLE_CHIK}', engine).iloc[0,0]
    
    min_count = int(min(count_dengue, count_zika, count_chik))
    
    print(f"Contagens: Dengue={count_dengue}, Zika={count_zika}, Chikungunya={count_chik}")
    print(f"Número Mágico (min_count) para undersampling: {min_count}")

    # --- ETAPA 5: Montar o "Dream Team" (Dataset Balanceado) ---
    print(f"Buscando {min_count} amostras aleatórias de cada classe...")
    df_dengue_balanced = pd.read_sql(f"SELECT * FROM {TEMP_TABLE_DENGUE} ORDER BY RANDOM() LIMIT {min_count}", engine)
    df_zika_balanced = pd.read_sql(f"SELECT * FROM {TEMP_TABLE_ZIKA} ORDER BY RANDOM() LIMIT {min_count}", engine)
    df_chik_balanced = pd.read_sql(f"SELECT * FROM {TEMP_TABLE_CHIK} ORDER BY RANDOM() LIMIT {min_count}", engine)
    
    print("Concatenando dataset final balanceado...")
    df_final_balanced = pd.concat([df_dengue_balanced, df_zika_balanced, df_chik_balanced]).sample(frac=1).reset_index(drop=True)
    
    # --- ETAPA 6: Salvar a Tabela Final ---
    print(f"Salvando tabela final '{FINAL_TABLE}' com {len(df_final_balanced)} registros...")
    df_final_balanced.to_sql(FINAL_TABLE, engine, if_exists='replace', index=False)
    
    print("\n--- Limpeza e Balanceamento Concluídos ---")
    print(f"Total de registros na tabela final: {len(df_final_balanced)}")
    print("Distribuição final das doenças:")
    print(df_final_balanced['doenca_alvo'].value_counts())
    
    # --- ETAPA 7: Limpar a Bagunça (Corrigida) ---
    print("Limpando tabelas temporárias...")
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text(f'DROP TABLE IF EXISTS {TEMP_TABLE_DENGUE}'))
            conn.execute(text(f'DROP TABLE IF EXISTS {TEMP_TABLE_ZIKA}'))
            conn.execute(text(f'DROP TABLE IF EXISTS {TEMP_TABLE_CHIK}'))
    
    end_time = time.time()
    print(f"Processo concluído em {end_time - start_time:.2f} segundos.")

except Exception as e:
    print(f"Ocorreu um erro durante o processo de limpeza: {e}")