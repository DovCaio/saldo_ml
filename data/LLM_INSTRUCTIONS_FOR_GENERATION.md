# Geração de dataset para classificação de transações financeiras brasileiras

Atue como um especialista em Machine Learning, processamento de linguagem natural (NLP) e dados financeiros brasileiros.

Estou desenvolvendo um classificador de transações financeiras para um sistema pessoal de controle financeiro chamado Saldo. O modelo utiliza BERTimbau (`neuralmind/bert-base-portuguese-cased`) para gerar embeddings e Logistic Regression para classificar as transações.

Minha base atual possui 1339 registros, distribuídos igualmente entre 9 categorias. A validação cruzada apresentou F1 Macro de aproximadamente 85,7%.

Quero ampliar significativamente a diversidade semântica do dataset, melhorando a capacidade de generalização do classificador para transações bancárias reais.

## 1. Objetivo

Gere exatamente **900 novas transações financeiras brasileiras**, distribuídas da seguinte maneira:

- 100 registros para cada uma das 9 categorias.
- Não repita descrições.
- Não crie apenas variações superficiais de frases existentes.
- Priorize diversidade de significado, estabelecimentos, serviços, padrões bancários e formas de descrição.

Se eu fornecer o CSV da base atual, utilize-o como referência para evitar duplicatas exatas e semânticas.

## 2. Categorias

Utilize exclusivamente estas categorias:

### education
Despesas relacionadas à educação:
- Mensalidades escolares e universitárias.
- Cursos profissionalizantes e preparatórios.
- Cursos de idiomas.
- Plataformas de ensino.
- Material didático e livros acadêmicos.
- Matrículas e certificações educacionais.

### emi
Parcelas, empréstimos e financiamentos:
- Parcelas de empréstimos.
- Financiamentos de veículos e imóveis.
- Crédito consignado.
- Crediários.
- Parcelamentos de compras e faturas.
- Prestação de financiamentos.

Não classifique compras comuns à vista nesta categoria.

### entertainment
Entretenimento e lazer:
- Streaming de filmes e músicas.
- Jogos digitais.
- Cinemas e teatros.
- Shows e festivais.
- Eventos e atividades recreativas.
- Assinaturas de entretenimento.

### food
Alimentação:
- Restaurantes.
- Lanchonetes.
- Padarias.
- Supermercados e mercearias.
- Açougues e hortifrutis.
- Delivery de alimentos.
- Cafeterias e estabelecimentos alimentícios.

### healthcare
Saúde e cuidados médicos:
- Hospitais e clínicas.
- Consultas médicas.
- Consultas odontológicas.
- Exames laboratoriais.
- Farmácias e medicamentos.
- Planos de saúde.
- Fisioterapia e tratamentos médicos.

### investment
Investimentos e aplicações financeiras:
- Aplicações em CDB, LCI e LCA.
- Tesouro Direto.
- Ações e fundos imobiliários.
- ETFs.
- Fundos de investimento.
- Previdência privada.
- Aportes em corretoras.

Não classifique transferências genéricas como investimentos sem evidência de sua finalidade.

### shopping
Compras de produtos e bens de consumo:
- Roupas e calçados.
- Eletrônicos e celulares.
- Eletrodomésticos.
- Móveis.
- Acessórios.
- Compras em marketplaces.
- Lojas físicas e virtuais.

Evite incluir alimentos, medicamentos e serviços de entretenimento nesta categoria.

### travel
Transporte e mobilidade:
- Aplicativos de transporte.
- Táxis.
- Combustível.
- Postos de abastecimento.
- Pedágios.
- Estacionamentos.
- Passagens de ônibus e avião.
- Transporte público.
- Aluguel de veículos.

### utilities
Contas e serviços essenciais:
- Energia elétrica.
- Água e saneamento.
- Gás.
- Internet residencial.
- Telefonia móvel e fixa.
- Provedores de internet.
- Serviços recorrentes de infraestrutura doméstica.

## 3. Diversidade obrigatória

Para cada categoria, procure contemplar diferentes tipos de descrição:

1. Nomes de estabelecimentos e empresas.
2. Descrições genéricas de extratos bancários.
3. Descrições abreviadas.
4. Descrições com identificadores de pagamento.
5. Nomes comerciais seguidos de informações adicionais.
6. Descrições com linguagem informal.
7. Diferentes formas de representar a mesma operação financeira.
8. Variações de acentuação, caixa e pontuação, quando naturais.
9. Transações de diferentes regiões do Brasil.
10. Descrições que se aproximem dos padrões encontrados em extratos de bancos e cartões.

Não produza apenas exemplos artificiais como "PAGAMENTO DE X", "COMPRA DE X" e "MENSALIDADE DE X".

Quero exemplos realistas, variados e que representem a maneira como estabelecimentos e instituições aparecem nas descrições de transações bancárias.

## 4. Distribuição da diversidade

Dentro dos 100 exemplos de cada categoria, procure equilibrar:

- 30 exemplos com nomes de estabelecimentos, empresas ou serviços.
- 25 exemplos com descrições bancárias genéricas.
- 20 exemplos com abreviações e identificadores.
- 15 exemplos com variações de escrita e nomenclaturas comerciais.
- 10 exemplos com situações menos comuns, mas ainda representativas da categoria.

Esses números são orientações de diversidade, não motivos para produzir frases artificiais.

## 5. Regras de qualidade

- Todas as descrições devem estar em português brasileiro, exceto nomes comerciais e identificadores que naturalmente apareçam em outros idiomas.
- Não invente estabelecimentos apresentados como se fossem empresas reais.
- Não utilize descrições excessivamente longas.
- Não gere exemplos sem relação clara com a categoria.
- Não atribua categorias com base apenas em palavras isoladas.
- Não inclua transferências Pix genéricas cuja finalidade seja impossível de identificar.
- Não utilize a mesma estrutura de frase repetidamente.
- Não produza exemplos ambíguos apenas para completar a quantidade.
- Não reutilize descrições da base existente.
- Cada registro deve representar uma transação independente.

Se uma descrição puder pertencer razoavelmente a duas categorias, substitua-a por outra mais clara.

## 6. Formato de saída

Gere um CSV com exatamente duas colunas:

description,category

Exemplo:

description,category
UBER *TRIP HELP,travel
MENSALIDADE PLATAFORMA DE ENSINO,education
DROGARIA CENTRAL COMPRA,healthcare
APLICACAO CDB BANCO,investment

Requisitos:
- Exatamente 900 registros, sem contar o cabeçalho.
- Exatamente 100 registros por categoria.
- Não inclua índices.
- Não inclua explicações dentro do CSV.
- Não inclua colunas adicionais.
- Utilize UTF-8.
- Escape corretamente vírgulas e aspas conforme o padrão CSV.

## 7. Verificação final

Antes de entregar, verifique:

1. Se existem exatamente 900 registros.
2. Se cada categoria possui exatamente 100 registros.
3. Se existem descrições duplicadas.
4. Se existem descrições semanticamente equivalentes.
5. Se alguma descrição recebeu uma categoria incorreta.
6. Se os exemplos representam diferentes padrões de transações brasileiras.

Se não conseguir entregar os 900 registros em uma única resposta, divida a geração em lotes por categoria, preservando a contagem e sem repetir registros entre os lotes.

O objetivo não é criar um dataset artificialmente perfeito, mas construir uma base diversificada e realista que ajude o classificador a generalizar para transações financeiras que ele ainda não encontrou.
