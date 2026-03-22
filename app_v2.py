import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Espacial Bolsa Família", layout="wide", page_icon="🗺️")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/base_final.csv")
    
    # Criar features faltantes caso não tenham sido totalmente propagadas
    targets = ['qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp']
    if 'total_beneficios' not in df.columns:
        df['total_beneficios'] = df[targets].sum(axis=1)
        
    df2 = df.copy()
    
    # Conversões e eng. features
    cols_para_converter = ['pib_municipal', 'populacao_total', 'populacao_urbana_2010', 'va_agropecuaria', 'va_industria', 'va_adm_publica', 'taxa_alfabetizacao']
    for c in cols_para_converter:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
            
    if 'pib_per_capita' not in df2.columns:
        df2['pib_per_capita'] = (df2['pib_municipal'] * 1000) / df2['populacao_total']
    if 'taxa_urbanizacao' not in df2.columns:
        df2['taxa_urbanizacao'] = (df2['populacao_urbana_2010'] / df2['populacao_total']) * 100
        
    if 'perc_va_agropecuaria' not in df2.columns:
        va_total = df2['va_agropecuaria'] + df2['va_industria'] + df2['va_adm_publica']
        df2['perc_va_agropecuaria'] = (df2['va_agropecuaria'] / va_total) * 100
        df2['perc_va_industria'] = (df2['va_industria'] / va_total) * 100
        df2['perc_va_adm_publica'] = (df2['va_adm_publica'] / va_total) * 100
        
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Treinamento
    colunas_removidas = ['cod_municipio', 'municipio', 'qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp', 'total_beneficios']
    X = df2.drop(columns=[c for c in colunas_removidas if c in df2.columns])
    X = X.fillna(X.median(numeric_only=True))
    y = df2['total_beneficios']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    df2['beneficios_previstos'] = rf.predict(X)
    df2['residuo'] = df2['total_beneficios'] - df2['beneficios_previstos']
    
    limite_superior = df2['residuo'].quantile(0.95)
    limite_inferior = df2['residuo'].quantile(0.05)
    
    def status_auditoria(res):
        if res > limite_superior:
            return "🔴 Sobreconcentração (Risco de Ineficiência/Fraude)"
        elif res < limite_inferior:
            return "🔵 Subatendimento (Possível Falta de Cobertura Rural)"
        else:
            return "🟢 Normal (Dentro do Esperado Macro)"
            
    df2['status'] = df2['residuo'].apply(status_auditoria)
    return df2

@st.cache_data
def load_geodata():
    import geobr
    # Download das fronteiras municipais do Brasil
    geo_df = geobr.read_municipality(code_muni="all", year=2020)
    
    # O código do geobr tem 7 dígitos numéricos
    geo_df['code_muni'] = geo_df['code_muni'].astype(str).str[:7].astype(int)
    
    # Simplificar a geometria reduz drasticamente o tempo de renderização no Plotly (.simplify)
    # Projetar para pseudo-mercator primeiro se necessário, mas o epsg=4674 original pode ser simplificado
    geo_df['geometry'] = geo_df['geometry'].simplify(0.05, preserve_topology=True)
    
    # Plotly Mapbox exige coordenadas WGS84 (EPSG:4326)
    geo_df = geo_df.to_crs(epsg=4326)
    
    return geo_df

# UI Streamlit
st.title("🗺️ Visão Geográfica: Auditoria Bolsa Família")
st.markdown("Análise Espacial das diferenças entre o volume **Real** e a **Previsão (IA)** em nível municipal usando **GeoPandas**.")

with st.spinner("Processando modelo IA e carregando dados base..."):
    df_processed = load_and_prepare_data()
    
with st.spinner("Carregando base cartográfica do Brasil (pode demorar no primeiro acesso)..."):
    try:
        geo_df = load_geodata()
        map_loaded = True
    except Exception as e:
        map_loaded = False
        st.error(f"Erro ao carregar dados espaciais via geobr: {e}")

if map_loaded:
    # Garantir chaves inteiras de cod IBGE para merge
    df_processed['cod_municipio_int'] = df_processed['cod_municipio'].astype(str).str[:7].astype(int)
    
    # Merge espacial e estatístico
    gdf = geo_df.merge(df_processed, left_on='code_muni', right_on='cod_municipio_int', how='inner')
    
    # O Plotly Express choropleth_mapbox performa bem quando as instâncias estão no index
    gdf.set_index('code_muni', inplace=True)
    
    st.subheader("Distribuição Espacial de Resíduos do Modelo")
    
    st.markdown("""
    **Sobre o Mapa:**  
    - Cores em Vermelho (**🔴 Sobreconcentração**) indicam municípios concedendo mais benefícios do que o modelo prevê.  
    - Cores em Azul (**🔵 Subatendimento**) indicam municípios com menos benefícios concedidos que o estimado.
    """)
    
    vis_data = gdf.copy()
    
    # Limitar anomalias extremas visualmente para proteger a escala de cores contínua
    resid_95 = vis_data['residuo'].quantile(0.95)
    resid_05 = vis_data['residuo'].quantile(0.05)
    vmax = max(abs(resid_95), abs(resid_05))
    
    fig = px.choropleth_mapbox(
        vis_data,
        geojson=vis_data.geometry,
        locations=vis_data.index,
        color="residuo",
        hover_name="municipio",
        hover_data={
            "status": True,
            "total_beneficios": ":.0f",
            "beneficios_previstos": ":.0f",
            "residuo": ":.0f",
            "taxa_urbanizacao": ":.1f"
        },
        color_continuous_scale="RdBu_r", 
        range_color=[-vmax, vmax],
        color_continuous_midpoint=0,
        mapbox_style="carto-positron",
        center={"lat": -15.78, "lon": -47.92},
        zoom=3.5,
        opacity=0.7,
        labels={"residuo": "Resíduo (Real - Previsto)"}
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    qtd_sobre = len(vis_data[vis_data['status'].str.contains("Sobreconcentração")])
    qtd_sub = len(vis_data[vis_data['status'].str.contains("Subatendimento")])
    
    col_kpi1.metric("Total Analisado", f"{len(vis_data)} Municípios")
    col_kpi2.metric("🔺 Alertas de Sobreconcentração", f"{qtd_sobre}")
    col_kpi3.metric("🔻 Alertas de Subatendimento", f"{qtd_sub}")
    
else:
    st.warning("A visualização do mapa não pôde ser completada pois a geometria não carregou.")
