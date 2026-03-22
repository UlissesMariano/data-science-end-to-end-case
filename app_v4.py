import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Estados Bolsa Família v4", layout="wide", page_icon="🗺️")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/base_final.csv")
    
    # Criar features faltantes caso não tenham sido totalmente propagadas
    targets = ['qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp']
    if 'total_beneficios' not in df.columns:
        df['total_beneficios'] = df[targets].sum(axis=1)
        
    df2 = df.copy()
    
    # Conversões e eng. features fundamentais
    cols_para_converter = ['pib_municipal', 'populacao_total', 'populacao_urbana_2010', 'va_agropecuaria', 'va_industria', 'va_adm_publica', 'taxa_alfabetizacao']
    for c in cols_para_converter:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
            
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Treinamento do Modelo por Município para calcular resíduos
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
    
    df_uf = df2.groupby('cod_uf', as_index=False).agg({
        'total_beneficios': 'sum',
        'beneficios_previstos': 'sum',
        'residuo': 'sum'
    })
    
    # Identificar TOP 5 Sobreconcentração e TOP 5 Subatendimento (agora TOP 5 para v4)
    top_5_sobre = df_uf.nlargest(5, 'residuo')
    top_5_sub = df_uf.nsmallest(5, 'residuo')
    
    df_uf['status'] = "Normal / Demais Estados"
    df_uf.loc[df_uf['cod_uf'].isin(top_5_sobre['cod_uf']), 'status'] = "🔴 Top 5 Sobreconcentração"
    df_uf.loc[df_uf['cod_uf'].isin(top_5_sub['cod_uf']), 'status'] = "🔵 Top 5 Subatendimento"
    
    # TODOS possuem residuo_mapa para manter o degradê geral
    df_uf['residuo_mapa'] = df_uf['residuo']
    
    # Para colocar a figura BEM EM CIMA DOS MUNICÍPIOS destaques, vamos achar o município 
    # destaque (com maior/menor resíduo) dentro de cada Estado Top 5
    munis_sobre = []
    for uf in top_5_sobre['cod_uf']:
        muni = df2[df2['cod_uf'] == uf].nlargest(1, 'residuo').iloc[0]
        munis_sobre.append(muni)
    df_munis_sobre = pd.DataFrame(munis_sobre)
    
    munis_sub = []
    for uf in top_5_sub['cod_uf']:
        muni = df2[df2['cod_uf'] == uf].nsmallest(1, 'residuo').iloc[0]
        munis_sub.append(muni)
    df_munis_sub = pd.DataFrame(munis_sub)
    
    return df_uf, df_munis_sobre, df_munis_sub

@st.cache_data
def load_geodata(df_munis_sobre, df_munis_sub):
    import geobr
    # Download das fronteiras estaduais do Brasil (borda completa + divisões)
    geo_df = geobr.read_state(code_state="all", year=2020)
    geo_df['code_state'] = geo_df['code_state'].astype(int)
    geo_df['geometry'] = geo_df['geometry'].simplify(0.01, preserve_topology=True)
    geo_df = geo_df.to_crs(epsg=4326)
    
    # Descobrir geometria dos municípios destaques
    codes_munis = list(df_munis_sobre['cod_municipio']) + list(df_munis_sub['cod_municipio'])
    states_needed = list(set([int(str(c)[:2]) for c in codes_munis]))
    
    muni_geos = []
    for st_code in states_needed:
        try:
            mgf = geobr.read_municipality(code_muni=st_code, year=2020)
            muni_geos.append(mgf)
        except:
            pass
            
    if muni_geos:
        muni_df = pd.concat(muni_geos)
        muni_df['code_muni'] = muni_df['code_muni'].astype(int)
        
        # Merge para lat lon
        df_sobre_geo = df_munis_sobre.merge(muni_df, left_on='cod_municipio', right_on='code_muni')
        df_sobre_geo = gpd.GeoDataFrame(df_sobre_geo, geometry='geometry').to_crs(epsg=4326)
        df_sobre_geo['lon'] = df_sobre_geo.geometry.centroid.x
        df_sobre_geo['lat'] = df_sobre_geo.geometry.centroid.y
        df_sobre_geo['name_muni'] = df_sobre_geo['name_muni'].astype(str)
        
        df_sub_geo = df_munis_sub.merge(muni_df, left_on='cod_municipio', right_on='code_muni')
        df_sub_geo = gpd.GeoDataFrame(df_sub_geo, geometry='geometry').to_crs(epsg=4326)
        df_sub_geo['lon'] = df_sub_geo.geometry.centroid.x
        df_sub_geo['lat'] = df_sub_geo.geometry.centroid.y
        df_sub_geo['name_muni'] = df_sub_geo['name_muni'].astype(str)
    else:
        df_sobre_geo = pd.DataFrame()
        df_sub_geo = pd.DataFrame()
        
    return geo_df, df_sobre_geo, df_sub_geo

# ====== UI Streamlit ======
st.title("🗺️ Mapa Estadual e Municipal: Auditoria Bolsa Família (Top 5)")
st.markdown("Visualização com degradê para todos os Estados. Marcadores localizados nos **Municípios Destaque** dos **Top 5 Estados** (Sobreconcentração e Subatendimento).")

with st.spinner("Processando IA e agregando por Município/Estado..."):
    df_uf, df_munis_sobre, df_munis_sub = load_and_prepare_data()
    
with st.spinner("Carregando base cartográfica espacial (Estados e Municípios destaques)..."):
    try:
        geo_df, df_sobre_geo, df_sub_geo = load_geodata(df_munis_sobre, df_munis_sub)
        map_loaded = True
    except Exception as e:
        map_loaded = False
        st.error(f"Erro ao carregar dados espaciais via geobr: {e}")

if map_loaded:
    # Merge espacial estadual
    gdf = geo_df.merge(df_uf, left_on='code_state', right_on='cod_uf', how='inner')
    gdf.set_index('code_state', inplace=True)
    
    st.subheader("Distribuição do Volume de Resíduos com Destaques")
    
    vis_data = gdf.copy()
    vmax = max(abs(vis_data['residuo_mapa'].max(skipna=True)), abs(vis_data['residuo_mapa'].min(skipna=True)))
    
    # Mapa Base (Degradê de Cores dos Estados)
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
        opacity=0.7,
        labels={"residuo_mapa": "Resíduo Estado (Real - Previsto)"}
    )
    
    # Overlay marcadores (Triângulos Amarelos)
    if not df_sobre_geo.empty:
        fig.add_trace(go.Scattermapbox(
            lat=df_sobre_geo['lat'],
            lon=df_sobre_geo['lon'],
            mode='text',
            text=['▲'] * len(df_sobre_geo),
            textfont=dict(size=22, color='darkorange'), # darkorange ou gold (amarelo) para visibilidade
            hoverinfo='text',
            hovertext=df_sobre_geo['name_muni'] + " (Muni. Destaque em Sobreconcentração)",
            showlegend=True,
            name='▲ Top 5 Sobreconcentração'
        ))

    # Overlay marcadores (Círculos Vermelhos)
    if not df_sub_geo.empty:
        fig.add_trace(go.Scattermapbox(
            lat=df_sub_geo['lat'],
            lon=df_sub_geo['lon'],
            mode='markers',
            marker=dict(size=14, color='red', opacity=0.9),
            hoverinfo='text',
            hovertext=df_sub_geo['name_muni'] + " (Muni. Destaque em Subatendimento)",
            showlegend=True,
            name='● Top 5 Subatendimento'
        ))
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=750,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Rankings Top 5
    col1, col2 = st.columns(2)
    with col1:
        st.error("### 🔴 Top 5 Sobreconcentração")
        df_sobre = vis_data[vis_data['status'] == "🔴 Top 5 Sobreconcentração"].sort_values(by="residuo", ascending=False)[['name_state', 'residuo', 'total_beneficios']]
        df_sobre['residuo'] = df_sobre['residuo'].apply(lambda x: f"+{int(x):,}".replace(',','.'))
        df_sobre['total_beneficios'] = df_sobre['total_beneficios'].apply(lambda x: f"{int(x):,}".replace(',','.'))
        st.dataframe(df_sobre.rename(columns={'name_state': 'Estado', 'residuo': 'Resíduo (Acima do IA)', 'total_beneficios': 'CadÚnico (Real)'}).reset_index(drop=True), use_container_width=True)
        
    with col2:
        st.info("### 🔵 Top 5 Subatendimento")
        df_sub = vis_data[vis_data['status'] == "🔵 Top 5 Subatendimento"].sort_values(by="residuo", ascending=True)[['name_state', 'residuo', 'total_beneficios']]
        df_sub['residuo'] = df_sub['residuo'].apply(lambda x: f"{int(x):,}".replace(',','.'))
        df_sub['total_beneficios'] = df_sub['total_beneficios'].apply(lambda x: f"{int(x):,}".replace(',','.'))
        st.dataframe(df_sub.rename(columns={'name_state': 'Estado', 'residuo': 'Resíduo (Abaixo do IA)', 'total_beneficios': 'CadÚnico (Real)'}).reset_index(drop=True), use_container_width=True)
    
else:
    st.warning("A visualização do mapa não pôde ser completada pois a geometria não carregou.")
