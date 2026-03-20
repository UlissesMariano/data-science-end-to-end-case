import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Bolsa Família", layout="wide", page_icon="🇧🇷")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/base_final.csv")
    
    # Criar features faltantes caso não tenham sido totalmente propagadas
    targets = ['qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp']
    if 'total_beneficios' not in df.columns:
        df['total_beneficios'] = df[targets].sum(axis=1)
        
    df2 = df.copy()
    
    # --- REPLICANDO A ENGENHARIA DE FEATURES DO NB05 ---
    # As VAs e PIB estão em formato numérico? Forçar conversão:
    cols_para_converter = ['pib_municipal', 'populacao_total', 'populacao_urbana_2010', 'va_agropecuaria', 'va_industria', 'va_adm_publica', 'taxa_alfabetizacao']
    for c in cols_para_converter:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
            
    # PIB e Urbanização
    if 'pib_per_capita' not in df2.columns:
        df2['pib_per_capita'] = (df2['pib_municipal'] * 1000) / df2['populacao_total']
    if 'taxa_urbanizacao' not in df2.columns:
        df2['taxa_urbanizacao'] = (df2['populacao_urbana_2010'] / df2['populacao_total']) * 100
        
    # Composição do Valor Adicionado (Setores)
    if 'perc_va_agropecuaria' not in df2.columns:
        va_total = df2['va_agropecuaria'] + df2['va_industria'] + df2['va_adm_publica']
        df2['perc_va_agropecuaria'] = (df2['va_agropecuaria'] / va_total) * 100
        df2['perc_va_industria'] = (df2['va_industria'] / va_total) * 100
        df2['perc_va_adm_publica'] = (df2['va_adm_publica'] / va_total) * 100
        
    # Limpeza de Infinitos causados por divisão por zero
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Treinar modelo on-the-fly para gerar o baseline
    colunas_removidas = ['cod_municipio', 'municipio', 'qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp', 'total_beneficios']
    X = df2.drop(columns=[c for c in colunas_removidas if c in df2.columns])
    X = X.fillna(X.median(numeric_only=True))
    y = df2['total_beneficios']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    df2['beneficios_previstos'] = rf.predict(X)
    df2['residuo'] = df2['total_beneficios'] - df2['beneficios_previstos']
    
    # Definir Status de Auditoria
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
    return df2, df

# Carregar Dados
df_processed, df_raw = load_and_prepare_data()

# Titulo e Header
st.title("📊 Painel de Monitoramento: Bolsa Família e Socioeconomia")
st.markdown("Analise a correlação entre os indicadores do município e a dependência do programa governamental, comparando o **volume real recebido** com a **previsão do modelo de inteligência artificial** (Random Forest) criado pelos dados do IBGE/Censo 2022.")

# Sidebar de Filtros
st.sidebar.header("🔍 Filtros de Busca")

# Garantir que Municipio venha em string limpa
df_processed['municipio'] = df_processed['municipio'].astype(str)
municipios_lista = sorted(df_processed['municipio'].unique())

cidade_selecionada = st.sidebar.selectbox("Selecione o Município", municipios_lista)

city_data = df_processed[df_processed['municipio'] == cidade_selecionada].iloc[0]

# KPIs Top
st.markdown(f"### 📍 Análise de **{city_data['municipio']}**")
col1, col2, col3, col4 = st.columns(4)

col1.metric("População Total (Censo)", f"{city_data['populacao_total']:,.0f}".replace(',','.'))
col2.metric("Benefícios Ativos (Real)", f"{city_data['total_beneficios']:,.0f}".replace(',','.'))
col3.metric("Expectativa IA (Base de Riqueza)", f"{city_data['beneficios_previstos']:,.0f}".replace(',','.'))

status_clean = city_data['status'].split(' ')[0] + " " + city_data['status'].split(' ')[1]
status_color = "normal" if "Normal" in city_data['status'] else ("inverse" if "Sobreconcentração" in city_data['status'] else "off")
    
col4.metric("Status da Auditoria", status_clean, delta=f"{city_data['residuo']:,.0f} benefs de diferença", delta_color=status_color)

st.divider()

# Colunas Metade-Metade
col_charts1, col_charts2 = st.columns(2)

with col_charts1:
    st.subheader("Bolsa Família: Real vs Previsto")
    fig = go.Figure(data=[
        go.Bar(name='CadÚnico / Dado Real', x=['Comparativo de Benefícios'], y=[city_data['total_beneficios']], marker_color='#1f77b4'),
        go.Bar(name='Estimativa da IA (Base IBGE)', x=['Comparativo de Benefícios'], y=[city_data['beneficios_previstos']], marker_color='#ff7f0e')
    ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Texto de Risco
    if "Sobreconcentração" in city_data['status']:
        st.error("⚠️ **Alerta:** Este município concede muito mais do que a infraestrutura econômica sugere. Demandar cruzamento local de CPFs.")
    elif "Subatendimento" in city_data['status']:
        st.info("💡 **Oportunidade Social:** O perfil desta cidade aponta muita pobreza rural. Agendar mutirão de cadastramento (Busca Ativa).")
    else:
        st.success("✅ **Dentro dos Padrões:** A cidade tem uma quantidade de bolsas operando no ritmo econômico natural.")

with col_charts2:
    st.subheader("Composição Principal do PIB Local")
    labels = ['Agropecuária', 'Indústria', 'Máquina Pública', 'Serviços/Ajustes']
    
    # Tratar colunas se NaN ou None
    va_agro = float(city_data.get('perc_va_agropecuaria', 0)) if pd.notna(city_data.get('perc_va_agropecuaria')) else 0.0
    va_ind = float(city_data.get('perc_va_industria', 0)) if pd.notna(city_data.get('perc_va_industria')) else 0.0
    va_adm = float(city_data.get('perc_va_adm_publica', 0)) if pd.notna(city_data.get('perc_va_adm_publica')) else 0.0
    servicos = max(0.0, 100.0 - (va_agro + va_ind + va_adm))
    
    values = [va_agro, va_ind, va_adm, servicos]
    
    fig2 = px.pie(values=values, names=labels, hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

# Indicadores Macro em Barra Horizontal
st.subheader("📊 Raio-X Sóciodemográfico vs Média Brasil")
df_nacional = df_processed.mean(numeric_only=True)

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

st.sidebar.markdown("---")
st.sidebar.markdown("**Case Especialista**\\nDashboard desenvolvido em Streamlit focando na validação macroeconômica e descoberta de oportunidades de políticas locais.")
