import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Estados Bolsa Família", layout="wide", page_icon="🗺️")

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
    
    # Treinamento do Modelo por Município
    colunas_removidas = ['cod_municipio', 'municipio', 'qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp', 'total_beneficios']
    X = df2.drop(columns=[c for c in colunas_removidas if c in df2.columns])
    X = X.fillna(X.median(numeric_only=True))
    y = df2['total_beneficios']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    df2['beneficios_previstos'] = rf.predict(X)
    df2['residuo'] = df2['total_beneficios'] - df2['beneficios_previstos']
    
    # Agrupamento por Estados (UF)
    df2['cod_uf'] = df2['cod_municipio'].astype(str).str[:2].astype(int)
    
    # Agregando os resíduos (soma do município representa o saldo do Estado)
    df_uf = df2.groupby('cod_uf', as_index=False).agg({
        'total_beneficios': 'sum',
        'beneficios_previstos': 'sum',
        'residuo': 'sum'
    })
    
    # Identificar TOP 10 Sobreconcentração e TOP 10 Subatendimento
    top_10_sobre = df_uf.nlargest(10, 'residuo')
    top_10_sub = df_uf.nsmallest(10, 'residuo')
    
    df_uf['status'] = "Normal / Fora do Top 10"
    df_uf.loc[df_uf['cod_uf'].isin(top_10_sobre['cod_uf']), 'status'] = "🔴 Top 10 Sobreconcentração"
    df_uf.loc[df_uf['cod_uf'].isin(top_10_sub['cod_uf']), 'status'] = "🔵 Top 10 Subatendimento"
    
    # Coluna usada para colorir o mapa (esconde quem está fora do Top 10 com NaN)
    df_uf['residuo_mapa'] = np.nan
    mask_destaque = df_uf['cod_uf'].isin(top_10_sobre['cod_uf']) | df_uf['cod_uf'].isin(top_10_sub['cod_uf'])
    df_uf.loc[mask_destaque, 'residuo_mapa'] = df_uf.loc[mask_destaque, 'residuo']
    
    return df_uf

@st.cache_data
def load_state_geodata():
    import geobr
    # Download das fronteiras estaduais do Brasil
    geo_df = geobr.read_state(code_state="all", year=2020)
    
    # O código do geobr tem 2 dígitos numéricos para estados
    geo_df['code_state'] = geo_df['code_state'].astype(int)
    
    geo_df['geometry'] = geo_df['geometry'].simplify(0.01, preserve_topology=True)
    geo_df = geo_df.to_crs(epsg=4326)
    
    return geo_df

# UI Streamlit
st.title("🗺️ Mapa Estadual: Auditoria Bolsa Família (Top 10)")
st.markdown("Agregação por Estado. Mostrando apenas os **Top 10 Estados com maior Sobreconcentração** e os **Top 10 com maior Subatendimento**.")

with st.spinner("Processando modelo IA e agregando por Estado..."):
    df_uf = load_and_prepare_data()
    
with st.spinner("Carregando base cartográfica Estadual..."):
    try:
        geo_df = load_state_geodata()
        map_loaded = True
    except Exception as e:
        map_loaded = False
        st.error(f"Erro ao carregar dados espaciais via geobr: {e}")

if map_loaded:
    # Merge espacial e estatístico
    gdf = geo_df.merge(df_uf, left_on='code_state', right_on='cod_uf', how='inner')
    
    gdf.set_index('code_state', inplace=True)
    
    st.subheader("Distribuição do Volume de Resíduos Agrupado por Estado")
    
    # Filtrar para exibir as UFs
    vis_data = gdf.copy()
    
    vmax = max(abs(vis_data['residuo_mapa'].max(skipna=True)), abs(vis_data['residuo_mapa'].min(skipna=True)))
    
    fig = px.choropleth_mapbox(
        vis_data,
        geojson=vis_data.geometry,
        locations=vis_data.index,
        color="residuo_mapa",
        hover_name="name_state",
        hover_data={
            "status": True,
            "total_beneficios": ":.0f",
            "beneficios_previstos": ":.0f",
            "residuo": ":.0f",
            "residuo_mapa": False
        },
        color_continuous_scale="RdBu_r", 
        range_color=[-vmax, vmax],
        color_continuous_midpoint=0,
        mapbox_style="carto-positron",
        center={"lat": -15.78, "lon": -47.92},
        zoom=3.5,
        opacity=0.8,
        labels={"residuo_mapa": "Resíduo (Real - Previsto)"}
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Rankings
    col1, col2 = st.columns(2)
    with col1:
        st.error("### 🔴 Top 10 Sobreconcentração")
        df_sobre = vis_data[vis_data['status'] == "🔴 Top 10 Sobreconcentração"].sort_values(by="residuo", ascending=False)[['name_state', 'residuo', 'total_beneficios']]
        df_sobre['residuo'] = df_sobre['residuo'].apply(lambda x: f"+{int(x):,}".replace(',','.'))
        df_sobre['total_beneficios'] = df_sobre['total_beneficios'].apply(lambda x: f"{int(x):,}".replace(',','.'))
        st.dataframe(df_sobre.rename(columns={'name_state': 'Estado', 'residuo': 'Resíduo (Acima do IA)', 'total_beneficios': 'CadÚnico (Real)'}).reset_index(drop=True), use_container_width=True)
        
    with col2:
        st.info("### 🔵 Top 10 Subatendimento")
        df_sub = vis_data[vis_data['status'] == "🔵 Top 10 Subatendimento"].sort_values(by="residuo", ascending=True)[['name_state', 'residuo', 'total_beneficios']]
        df_sub['residuo'] = df_sub['residuo'].apply(lambda x: f"{int(x):,}".replace(',','.'))
        df_sub['total_beneficios'] = df_sub['total_beneficios'].apply(lambda x: f"{int(x):,}".replace(',','.'))
        st.dataframe(df_sub.rename(columns={'name_state': 'Estado', 'residuo': 'Resíduo (Abaixo do IA)', 'total_beneficios': 'CadÚnico (Real)'}).reset_index(drop=True), use_container_width=True)
    
else:
    st.warning("A visualização do mapa não pôde ser completada pois a geometria não carregou.")
