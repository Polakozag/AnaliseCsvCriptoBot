# Dashboard de Análise de Estratégias - Criptobot

## Visão Geral
Dashboard em Streamlit para análise de trades históricos e busca de oportunidades em pares de criptomoedas, com foco em volatilidade realista (baseada na duração média das operações do bot).

---

## Funcionalidades
- Upload e análise de trades históricos (CSV)
- Resumo de resultados, gráficos e tabelas por par, dia e hora
- Cálculo da duração média das operações
- Filtros para análise de pares potenciais (idade, market cap, volume)
- Busca de dados de mercado em tempo real (CoinGecko e Binance)
- Cálculo de volatilidade para 1d e 1h, ajustado para a janela média do bot
- Debug detalhado dos motivos de exclusão de pares
- Cache incremental de candles (1d e 1h) em arquivos CSV por par/timeframe na pasta `cache_klines/`
- Cache de klines por mês em CSV (cache_klines/{symbol}_{interval}_{YYYYMM}.csv)
- Score de pares conforme critérios de volatilidade
- Cálculo de volatilidade aparada (2,5% e 5%) e diferença para média simples
- Cálculo de média aparada (2,5%) para lucro e rentabilidade dos trades

---

## Mudanças recentes
- Layout dos gráficos e tabelas ajustado para visual mais limpo e alinhado, com uso de colunas lado a lado para melhor comparação.
- No ranking de pares potenciais:
  - Coluna "Moeda" exibe apenas o nome da moeda (sem USDT).
  - Colunas Score, 7d, 30d e 90d com degradê visual para os top 5.
- Nos gráficos de volatilidade, o eixo X mostra apenas a moeda (sem USDT).
- Mantida a lógica de score e filtros conforme backup.
- Todas as melhorias visuais e de usabilidade documentadas.

---

## Cálculos e Constantes
- **Volatilidade (std):** Desvio padrão dos log-returns, anualizado para 1d e ajustado para 1h (multiplicado por sqrt(24))
- **Volatilidade aparada:** Desvio padrão removendo 2,5% e 5% das pontas (outliers)
- **Média aparada:** Média removendo 2,5% das pontas
- **Score:** (vol 7d > mediana) + (vol 30d > mediana) + (vol 7d > vol 30d)
- **Constantes:**
  - STABLECOINS = ['USDC', 'FDUSD', 'BUSD', 'USD1']
  - Cores de métricas: lucros, trades, rentabilidade, etc.
  - Períodos: 7d, 30d, 60d, 90d, 180d, 365d

---

## Como funciona o corte (trimmed mean/std)
- Para cada série (lucro, rentabilidade, volatilidade):
  - Ordena os valores
  - Remove 2,5% (ou 5%) dos menores e maiores valores
  - Calcula a média ou desvio padrão do restante
- Usado para eliminar spikes e dar visão mais robusta

---

## Estrutura do Projeto
- `app.py`: Código principal do dashboard
- `cache_klines/`: Cache de candles por par/timeframe/mês
- `extrato-criptobot*.csv`: Arquivos de trades históricos
- `PROJETO.md`: Documentação de decisões e backlog
- `README.md`: Este arquivo, visão geral e cálculos

---

## Como usar
1. Suba seu arquivo de trades CSV
2. Ajuste os filtros conforme desejado
3. Analise os resultados, gráficos e tabelas
4. Use as colunas de média/volatilidade aparada para avaliar robustez

---

## Observações
- O corte de 2,5% é padrão para médias aparadas, mas pode ser ajustado no código
- O score é apenas um guia inicial, pode ser refinado
- O sistema está preparado para grandes volumes de dados, mas pode ser otimizado ainda mais

---

> Mantenha este README atualizado a cada evolução relevante do projeto.
