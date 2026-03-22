import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Bolsa Família", layout="wide", page_icon="🌐")

# Customização de CSS para aumentar levemente os textos e títulos
st.markdown("""
<style>
/* Títulos maiores (h1, h2, h3) */
h1 {
    font-size: 2.6rem !important;
}
h2 {
    font-size: 2.1rem !important;
}
h3 {
    font-size: 1.5rem !important;
}
/* Títulos das Abas */
button[data-baseweb="tab"] div p {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}
/* Aumentar de leve as fontes de parágrafo / textos corridos */
div[data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/base_final.csv")
    
    targets = ['qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp']
    if 'total_beneficios' not in df.columns:
        df['total_beneficios'] = df[targets].sum(axis=1)
        
    df2 = df.copy()
    
    # Conversões
    cols_para_converter = ['pib_municipal', 'populacao_total', 'populacao_urbana_2010', 'va_agropecuaria', 'va_industria', 'va_adm_publica', 'taxa_alfabetizacao']
    for c in cols_para_converter:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
            
    # PIB e Urbanização
    if 'pib_per_capita' not in df2.columns:
        df2['pib_per_capita'] = (df2['pib_municipal'] * 1000) / df2['populacao_total']
    if 'taxa_urbanizacao' not in df2.columns:
        df2['taxa_urbanizacao'] = (df2['populacao_urbana_2010'] / df2['populacao_total']) * 100
        
    # Composição do Valor Adicionado
    if 'perc_va_agropecuaria' not in df2.columns:
        va_total = df2['va_agropecuaria'] + df2['va_industria'] + df2['va_adm_publica']
        df2['perc_va_agropecuaria'] = (df2['va_agropecuaria'] / va_total) * 100
        df2['perc_va_industria'] = (df2['va_industria'] / va_total) * 100
        df2['perc_va_adm_publica'] = (df2['va_adm_publica'] / va_total) * 100
        
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Treinamento do Modelo para gerar o baseline
    colunas_removidas = ['cod_municipio', 'municipio', 'qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp', 'total_beneficios']
    X = df2.drop(columns=[c for c in colunas_removidas if c in df2.columns])
    X = X.fillna(X.median(numeric_only=True))
    y = df2['total_beneficios']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    df2['beneficios_previstos'] = rf.predict(X)
    df2['residuo'] = df2['total_beneficios'] - df2['beneficios_previstos']
    
    # Status Municipal (para a Aba 1)
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
    
    # Agrupamento Estadual (para a Aba 2)
    df2['cod_uf'] = df2['cod_municipio'].astype(str).str[:2].astype(int)
    df_uf = df2.groupby('cod_uf', as_index=False).agg({
        'total_beneficios': 'sum',
        'beneficios_previstos': 'sum',
        'residuo': 'sum'
    })
    
    # Status Estadual (usando quantis apropriados para Estados - n=27)
    # 5% e 95% em 27 estados pode isolar o Top 1 ou Top 2
    lim_sup_uf = df_uf['residuo'].quantile(0.95)
    lim_inf_uf = df_uf['residuo'].quantile(0.05)
    
    def status_uf(res):
        if res > lim_sup_uf:
            return "🔴 Sobreconcentração Estadual"
        elif res < lim_inf_uf:
            return "🔵 Subatendimento Estadual"
        else:
            return "🟢 Dentro do Esperado"
            
    df_uf['status'] = df_uf['residuo'].apply(status_uf)
    
    # Extrair destaques para colocar as marcações (bolinhas e triângulos)
    estados_sobre = df_uf[df_uf['status'].str.contains("Sobreconcentração")]['cod_uf']
    estados_sub = df_uf[df_uf['status'].str.contains("Subatendimento")]['cod_uf']
    
    munis_sobre = []
    for uf in estados_sobre:
        muni = df2[df2['cod_uf'] == uf].nlargest(1, 'residuo').iloc[0]
        munis_sobre.append(muni)
    df_munis_sobre = pd.DataFrame(munis_sobre) if munis_sobre else pd.DataFrame()
    
    munis_sub = []
    for uf in estados_sub:
        muni = df2[df2['cod_uf'] == uf].nsmallest(1, 'residuo').iloc[0]
        munis_sub.append(muni)
    df_munis_sub = pd.DataFrame(munis_sub) if munis_sub else pd.DataFrame()
    
    return df2, df_uf, df_munis_sobre, df_munis_sub

@st.cache_data
def load_state_geodata(df_munis_sobre, df_munis_sub):
    import geobr
    geo_df = geobr.read_state(code_state="all", year=2020)
    geo_df['code_state'] = geo_df['code_state'].astype(int)
    geo_df['geometry'] = geo_df['geometry'].simplify(0.01, preserve_topology=True)
    geo_df = geo_df.to_crs(epsg=4326)
    
    # Descobrir geometria dos municípios destaques
    codes_munis = []
    if not df_munis_sobre.empty:
        codes_munis += list(df_munis_sobre['cod_municipio'])
    if not df_munis_sub.empty:
        codes_munis += list(df_munis_sub['cod_municipio'])
        
    states_needed = list(set([int(str(c)[:2]) for c in codes_munis]))
    
    muni_geos = []
    for st_code in states_needed:
        try:
            mgf = geobr.read_municipality(code_muni=st_code, year=2020)
            muni_geos.append(mgf)
        except:
            pass
            
    df_sobre_geo = pd.DataFrame()
    df_sub_geo = pd.DataFrame()
    
    if muni_geos:
        muni_df = pd.concat(muni_geos)
        muni_df['code_muni'] = muni_df['code_muni'].astype(int)
        
        if not df_munis_sobre.empty:
            df_sobre_geo = df_munis_sobre.merge(muni_df, left_on='cod_municipio', right_on='code_muni')
            if not df_sobre_geo.empty:
                df_sobre_geo = gpd.GeoDataFrame(df_sobre_geo, geometry='geometry').to_crs(epsg=4326)
                df_sobre_geo['lon'] = df_sobre_geo.geometry.centroid.x
                df_sobre_geo['lat'] = df_sobre_geo.geometry.centroid.y
                df_sobre_geo['name_muni'] = df_sobre_geo['name_muni'].astype(str)
            
        if not df_munis_sub.empty:
            df_sub_geo = df_munis_sub.merge(muni_df, left_on='cod_municipio', right_on='code_muni')
            if not df_sub_geo.empty:
                df_sub_geo = gpd.GeoDataFrame(df_sub_geo, geometry='geometry').to_crs(epsg=4326)
                df_sub_geo['lon'] = df_sub_geo.geometry.centroid.x
                df_sub_geo['lat'] = df_sub_geo.geometry.centroid.y
                df_sub_geo['name_muni'] = df_sub_geo['name_muni'].astype(str)
                
    return geo_df, df_sobre_geo, df_sub_geo

with st.spinner("Processando modelo IA e carregando dados..."):
    df_muni, df_uf, df_munis_sobre, df_munis_sub = load_and_prepare_data()

st.title("📊 Dashboard Bolsa Família")

# Criação das Abas
tab1, tab2 = st.tabs(["Painel Municipal", "Visão Geográfica Estadual"])

# ================= ABA 1 ================= #
with tab1:
    st.markdown("Analise a correlação entre os indicadores do município e a dependência do programa governamental, comparando o **volume real recebido** com a **previsão do modelo de inteligência artificial** (Random Forest) criado pelos dados do IBGE/Censo 2022.")
    
    st.sidebar.header("🔍 Filtros de Busca (Aba 1)")
    df_muni['municipio'] = df_muni['municipio'].astype(str)
    municipios_lista = sorted(df_muni['municipio'].unique())
    cidade_selecionada = st.sidebar.selectbox("Selecione o Município", municipios_lista)
    
    city_data = df_muni[df_muni['municipio'] == cidade_selecionada].iloc[0]
    
    st.markdown(f"### 📍 Análise de **{city_data['municipio']}**")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("População Total (Censo)", f"{city_data['populacao_total']:,.0f}".replace(',','.'))
    col2.metric("Benefícios Ativos (Real)", f"{city_data['total_beneficios']:,.0f}".replace(',','.'))
    col3.metric("Expectativa IA (Base de Riqueza)", f"{city_data['beneficios_previstos']:,.0f}".replace(',','.'))
    
    status_clean = city_data['status'].split(' ')[0] + " " + city_data['status'].split(' ')[1]
    status_color = "normal" if "Normal" in city_data['status'] else ("inverse" if "Sobreconcentração" in city_data['status'] else "off")
    col4.metric("Status da Auditoria", status_clean, delta=f"{city_data['residuo']:,.0f} benefs de diferença", delta_color=status_color)
    
    st.divider()
    
    col_charts1, col_charts2 = st.columns(2)
    with col_charts1:
        st.subheader("Bolsa Família: Real vs Previsto")
        fig = go.Figure(data=[
            go.Bar(name='Dado Real', x=['Comparativo de Benefícios'], y=[city_data['total_beneficios']], marker_color='#1f77b4'),
            go.Bar(name='Dado Previsto', x=['Comparativo de Benefícios'], y=[city_data['beneficios_previstos']], marker_color='#ff7f0e')
        ])
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        if "Sobreconcentração" in city_data['status']:
            st.error("⚠️ **Alerta:** Este município concede muito mais do que a infraestrutura econômica sugere.")
        elif "Subatendimento" in city_data['status']:
            st.info("💡 **Oportunidade Social:** O perfil desta cidade aponta muita pobreza rural.")
        else:
            st.success("✅ **Dentro dos Padrões:** A cidade tem uma quantidade de bolsas operando no ritmo econômico natural.")
            
    with col_charts2:
        st.subheader("Composição Principal do PIB Local")
        labels = ['Agropecuária', 'Indústria', 'Máquina Pública', 'Serviços/Ajustes']
        va_agro = float(city_data.get('perc_va_agropecuaria', 0)) if pd.notna(city_data.get('perc_va_agropecuaria')) else 0.0
        va_ind = float(city_data.get('perc_va_industria', 0)) if pd.notna(city_data.get('perc_va_industria')) else 0.0
        va_adm = float(city_data.get('perc_va_adm_publica', 0)) if pd.notna(city_data.get('perc_va_adm_publica')) else 0.0
        servicos = max(0.0, 100.0 - (va_agro + va_ind + va_adm))
        
        values = [va_agro, va_ind, va_adm, servicos]
        fig2 = px.pie(values=values, names=labels, hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("📊 Raio-X Sóciodemográfico vs Média Brasil")
    df_nacional = df_muni.mean(numeric_only=True)
    indicadores = ['Alfabetização (%)', 'Urbanização (%)']
    city_vals = [
        float(city_data.get('taxa_alfabetizacao', 0)) if pd.notna(city_data.get('taxa_alfabetizacao')) else 0.0, 
        float(city_data.get('taxa_urbanizacao', 0)) if pd.notna(city_data.get('taxa_urbanizacao')) else 0.0
    ]
    br_vals = [
        float(df_nacional.get('taxa_alfabetizacao', 0)) if pd.notna(df_nacional.get('taxa_alfabetizacao')) else 0.0, 
        float(df_nacional.get('taxa_urbanizacao', 0)) if pd.notna(df_nacional.get('taxa_urbanizacao')) else 0.0
    ]
    
    fig3 = go.Figure(data=[
        go.Bar(name='Município Atual', y=indicadores, x=city_vals, orientation='h', marker_color='#2ca02c'),
        go.Bar(name='Média Brasil', y=indicadores, x=br_vals, orientation='h', marker_color='#7f7f7f')
    ])
    fig3.update_layout(barmode='group')
    st.plotly_chart(fig3, use_container_width=True)

# ================= ABA 2 ================= #
with tab2:
    st.markdown("Análise Espacial das diferenças entre o volume **Real** e a **Previsão (IA)** agrupado em nível **Estadual** usando **GeoPandas**.")
    
    with st.spinner("Carregando base cartográfica do Brasil na Aba 2..."):
        try:
            geo_df, df_sobre_geo, df_sub_geo = load_state_geodata(df_munis_sobre, df_munis_sub)
            map_loaded = True
        except Exception as e:
            map_loaded = False
            st.error(f"Erro ao carregar dados espaciais via geobr: {e}")
            
    if map_loaded:
        gdf = geo_df.merge(df_uf, left_on='code_state', right_on='cod_uf', how='inner')
        gdf.set_index('code_state', inplace=True)
        
        st.subheader("Distribuição Espacial de Resíduos do Modelo (Por UF)")
        
        st.markdown("""
        **Sobre as Cores e Marcações no Mapa:**  
        - 🟥 **Cores do Estado em Vermelho:** Indicam Estados concedendo mais benefícios do que o previsto (Sobreconcentração Estadual).
        - 🟦 **Cores do Estado em Azul:** Indicam Estados concedendo menos benefícios do que o previsto (Subatendimento Estadual).
        - 🟡 **Bolinhas Amarelas:** Marca o Município responsável pelo pico de Sobreconcentração no Estado alerta.
        - 🔴 **Bolinhas Vermelhas:** Marca o Município responsável pelo pico de Subatendimento no Estado alerta.
        """)
        
        vis_data = gdf.copy()
        
        resid_95 = vis_data['residuo'].quantile(0.95)
        resid_05 = vis_data['residuo'].quantile(0.05)
        vmax = max(abs(resid_95), abs(resid_05))
        
        # Colorir pelo resíduo do Estado com Mapbox para manter a essência do app_v2.py 
        fig_map = px.choropleth_mapbox(
            vis_data,
            geojson=vis_data.geometry,
            locations=vis_data.index,
            color="residuo",
            hover_name="name_state",
            hover_data={
                "status": True,
                "total_beneficios": ":.0f",
                "beneficios_previstos": ":.0f",
                "residuo": ":.0f"
            },
            color_continuous_scale="RdBu_r", 
            range_color=[-vmax, vmax],
            color_continuous_midpoint=0,
            mapbox_style="carto-positron",
            center={"lat": -15.78, "lon": -47.92},
            zoom=3.5,
            opacity=0.7,
            labels={"residuo": "Resíduo Estado (Real - Previst)"}
        )
        
        # Overlay marcadores (Triângulos Amarelos/Dourados)
        if not df_sobre_geo.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=df_sobre_geo['lat'],
                lon=df_sobre_geo['lon'],
                mode='markers',
                marker=dict(size=14, color='gold', opacity=0.9),
                hoverinfo='text',
                hovertext=df_sobre_geo['name_muni'] + " (Muni. Destaque em Sobreconcentração)",
                showlegend=True,
                name='🟡 Muni. Sobreconcentração'
            ))

        # Overlay marcadores (Círculos Vermelhos)
        if not df_sub_geo.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=df_sub_geo['lat'],
                lon=df_sub_geo['lon'],
                mode='markers',
                marker=dict(size=14, color='red', opacity=0.9),
                hoverinfo='text',
                hovertext=df_sub_geo['name_muni'] + " (Muni. Destaque em Subatendimento)",
                showlegend=True,
                name='🔴 Muni. Subatendimento'
            ))
            
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            height=700,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="white", font=dict(color="black"))
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        qtd_sobre = len(vis_data[vis_data['status'].str.contains("Sobreconcentração")])
        qtd_sub = len(vis_data[vis_data['status'].str.contains("Subatendimento")])
        
        col_kpi1.metric("Total Analisado", f"{len(vis_data)} Estados")
        col_kpi2.metric("🟥 Estados em Sobreconcentração", f"{qtd_sobre}")
        col_kpi3.metric("🟦 Estados em Subatendimento", f"{qtd_sub}")
        
    else:
        st.warning("A visualização do mapa não pôde ser completada pois a geometria não carregou.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Case Especialista**\\nDashboard desenvolvido em Streamlit focando na validação macroeconômica e descoberta de oportunidades de políticas locais.")
