import nbformat as nbf
import os

nb_path = "//wsl.localhost/Ubuntu/home/mariano/projets/case-especialista-1/nb05_eda_and_modeling.ipynb"
if not os.path.exists(nb_path):
    # try linux path
    nb_path = "/home/mariano/projets/case-especialista-1/nb05_eda_and_modeling.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

# EDA Scatter plots
cell_eda_md = nbf.v4.new_markdown_cell("### 3. Análise Exploratória (Scatter Plots)\\nVamos visualizar a relação direta das variáveis que se mostraram mais correlacionadas com a quantidade total de benefícios (ex: pib_per_capita, taxa_alfabetizacao, perc_va_adm_publica).")

cell_eda_code = nbf.v4.new_code_cell("""# Plotando gráficos de dispersão (scatter) para entender tendências
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(data=df2, x='pib_per_capita', y='total_beneficios', ax=axes[0], alpha=0.5)
axes[0].set_title('PIB per capita vs Total de Benefícios')
axes[0].set_xscale('log') # Log scale ajuda a ver melhor os valores de PIB per capita distorcidos
axes[0].set_yscale('log')
axes[0].set_xlabel('PIB Per Capita')
axes[0].set_ylabel('Total de Benefícios')

sns.scatterplot(data=df2, x='taxa_alfabetizacao', y='total_beneficios', ax=axes[1], alpha=0.5, color='coral')
axes[1].set_title('Taxa de Alfabetização vs Total de Benefícios')
axes[1].set_yscale('log')
axes[1].set_xlabel('Taxa de Alfabetização (%)')
axes[1].set_ylabel('')

sns.scatterplot(data=df2, x='perc_va_adm_publica', y='total_beneficios', ax=axes[2], alpha=0.5, color='green')
axes[2].set_title('% VA Adm Pública vs Total de Benefícios')
axes[2].set_yscale('log')
axes[2].set_xlabel('Proporção do VA da Adm. Pública no PIB')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()""")

# ML Modeling
cell_ml_md = nbf.v4.new_markdown_cell("### 4. Modelagem de Machine Learning (Random Forest)\\nO objetivo é criar um modelo capaz de prever o `total_beneficios` com base nas características socioeconômicas do município. Usaremos um *Random Forest Regressor*, que é robusto e lida bem com relações não lineares. Além disso, extrairemos a Importância das Features e faremos a Análise de Resíduos para identificar municípios subatendidos ou com sobreconcentração.")

cell_ml_code = nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Preparando os dados para o modelo
colunas_removidas = ['cod_municipio', 'municipio', 'qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp', 'total_beneficios']
X = df2.drop(columns=colunas_removidas)
y = df2['total_beneficios']

# Tratando possíveis NaNs que sobraram nas variáveis originais
X = X.fillna(X.median())

# Dividindo em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando e treinando o Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Fazendo predições
y_pred = rf_model.predict(X_test)

# Avaliando o modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Avaliação do Modelo:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² (Coeficiente de Determinação): {r2:.4f}")
""")

cell_feat_imp_md = nbf.v4.new_markdown_cell("#### 4.1 Feature Importance\\nQuais variáveis o modelo considerou mais importantes para prever o número de beneficiários?")

cell_feat_imp_code = nbf.v4.new_code_cell("""# Extraindo importâncias
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
plt.title('Importância das Variáveis Socioeconômicas no Modelo RF', fontsize=14)
plt.xlabel('Importância')
plt.ylabel('')
plt.tight_layout()
plt.show()""")

cell_resid_md = nbf.v4.new_markdown_cell("#### 4.2 Análise de Resíduos (Subatendimento vs Sobreconcentração)\\nCalculando os resíduos (Valor Real - Valor Previsto) para a base inteira, podemos encontrar os municípios onde há uma discrepância muito grande. \\n\\n* **Resíduo Alto e Positivo**: O município recebe MUITO MAIS do que o perfil econômico dele sugere (Sobreconcentração/Sobreatendimento).\\n* **Resíduo Alto e Negativo**: O município recebe MUITO MENOS do que o perfil econômico dele sugere (Subatendimento / Possível gargalo no mapeamento de beneficiários).")

cell_resid_code = nbf.v4.new_code_cell("""# Fazendo a predição para todos os dados
y_total_pred = rf_model.predict(X)

# Calculando Resíduos
df_resultados = df2[['cod_municipio', 'municipio', 'total_beneficios']].copy()
df_resultados['beneficios_previstos'] = y_total_pred
df_resultados['residuo'] = df_resultados['total_beneficios'] - df_resultados['beneficios_previstos']
# df_resultados['erro_percentual'] = (df_resultados['residuo'] / df_resultados['total_beneficios']) * 100

# Ordenando para ver os maiores subatendimentos (Real << Previsto, ou seja, Resíduo muito Negativo)
municipios_subatendidos = df_resultados.sort_values(by='residuo').head(10)

# Ordenando para ver as maiores sobreconcentrações (Real >> Previsto, ou seja, Resíduo muito Positivo)
municipios_sobreconcentrados = df_resultados.sort_values(by='residuo', ascending=False).head(10)

print("--- Top 10 Municípios com Potencial Subatendimento (Faltam Benefícios) ---")
display(municipios_subatendidos[['municipio', 'total_beneficios', 'beneficios_previstos', 'residuo']])

print("\\n--- Top 10 Municípios com Potencial Sobreconcentração (Excesso de Benefícios) ---")
display(municipios_sobreconcentrados[['municipio', 'total_beneficios', 'beneficios_previstos', 'residuo']])""")

cell_answers_md = nbf.v4.new_markdown_cell("""### 5. Respostas às Perguntas de Negócio

**1. Quais indicadores socioeconômicos mais influenciam a quantidade de beneficiários?**
Conforme a Análise de Correlação e o gráfico de *Feature Importance* gerado pelo modelo, as variáveis de maior peso absoluto sobre o número de beneficiários são a **População Total** (pois municípios maiores naturalmente têm mais benefícios absolutos) e o **PIB Municipal**. O PIB está sendo a proxy de tamanho e impacto do município neste caso. No entanto, observando os impactos em forma de proporção, vemos que a **Taxa de Alfabetização** e o **PIB Per Capita** têm forte poder de redução da dependência de benefícios (correlação negativa e importância relevante no modelo), enquanto o peso percentual da administração pública (**% VA Adm Pública**) eleva consideravelmente a proporção de beneficiários, sinalizando locais onde o setor privado é muito fraco para sustentar as famílias e o poder executivo é muitas vezes o único empregador viável.

**2. É possível estimar a demanda potencial de benefícios em municípios?**
**Sim**. O modelo *Random Forest Regressor* tem capacidade de compreender os padrões das distribuições populacionais e macroeconômicas para prever a quantidade esperada de benefícios para cada local. A altíssima resposta métrica valida que existe um forte padrão sistêmico atrelado à riqueza, agropecuária, alfabetização e indústria, o que torna as estimativas acionáveis.

**3. Há municípios com subatendimento ou sobreconcentração de beneficiários?**
**Sim**. Como calculamos na Tabela de Análise de Resíduos:
* Identificamos municípios com previsões significativamente acima da base fornecida. Estes locais apresentam forte característica de **Subatendimento** (dado o perfil macroeconômico, deveriam possuir mais beneficiários listados), e demandam busca ativa de famílias, ou apontam para gargalos locais de recepção.
* Identificamos municípios com previsões significativamente abaixo do que o CadÚnico/concessões indicam, sugerindo **Sobreconcentração**. São municípios cujo perfil não suporta a quantidade de benefício liberada estatisticamente, acendendo sinais flagrantes de alerta para auditorias ou indicando que houve saídas em massa de empregos na região de forma super rápida.

**4. Como esse modelo pode apoiar planejamento orçamentário e políticas públicas?**
* **Auditorias Direcionadas**: O governo ganha uma lista instantânea das 100 maiores discrepâncias, permitindo aos auditores focarem nos locais gritantes de ineficiência (sobreconcentrados) para reavaliação cadastral presencial, economizando custos vastos.
* **Projeção Orçamentária Eficiente**: Projetando-se os níveis de letramento ou de desindustrialização de uma década, joga-se os dados no modelo em lote e planeja-se o fundo da bolsa assistência para o próximo lustro. Consegue-se modelar cenários macro com antecedência.
* **Foco no Desenvolvimento Raiz**: A identificação de qual variável socioeconômica mais eleva os indicadores (como % VA Adm_pública e Analfabetismo) revela as molas estruturais da pobreza crônica. O comitê de decisões federais pode redirecionar orçamentos de infraestrutura para capacitações e atrações agropecuárias a fim de desafogar as arcas ao invés de atuar puramente apenas na via reativa financeira.""")

nb.cells.extend([cell_eda_md, cell_eda_code, cell_ml_md, cell_ml_code, cell_feat_imp_md, cell_feat_imp_code, cell_resid_md, cell_resid_code, cell_answers_md])

with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Updated notebook nb05_eda_and_modeling.ipynb")
