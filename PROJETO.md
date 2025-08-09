# Documentação do Projeto: Dashboard de Análise de Estratégias - Criptobot

## Visão Geral
Dashboard em Streamlit para análise de trades históricos e busca de oportunidades em pares de criptomoedas, com foco em volatilidade realista (baseada na duração média das operações do bot).

---

## O que já foi implementado
- Upload e análise de trades históricos (CSV)
- Resumo de resultados, gráficos e tabelas por par, dia e hora
- Cálculo da duração média das operações
- Filtros para análise de pares potenciais (idade, market cap, volume)
- Busca de dados de mercado em tempo real (CoinGecko e Binance)
- Cálculo de volatilidade para 1d e 1h, ajustado para a janela média do bot
- Debug detalhado dos motivos de exclusão de pares
- Cache incremental de candles (1d e 1h) em arquivos CSV por par/timeframe na pasta `cache_klines/`. O sistema busca só o que falta e atualiza o cache automaticamente. Essa funcionalidade já está integrada na análise de pares potenciais.
- Cache de klines por mês em CSV (cache_klines/{symbol}_{interval}_{YYYYMM}.csv). O sistema busca e atualiza só o mês atual, lê todos os meses para montar o histórico. Se não houver histórico, busca tudo da API e cria os arquivos mensais.

---

## Mudanças recentes (2025-08-08)
- Layout dos gráficos e tabelas ajustado para visual mais limpo e alinhado, com uso de colunas lado a lado para melhor comparação.
- No ranking de pares potenciais:
	- Coluna "Moeda" exibe apenas o nome da moeda (sem USDT).
	- Colunas Score, 7d, 30d e 90d com degradê visual para os top 5.
- Nos gráficos de volatilidade, o eixo X mostra apenas a moeda (sem USDT).
- Mantida a lógica de score e filtros conforme backup.
- Todas as melhorias visuais e de usabilidade documentadas.
  
---

## Histórico de Mudanças
...existing code...

## Próximos Passos (Backlog)
- [PLANEJADO] Se necessário, reativar cortes de outliers para estudo de robustez.
- [PLANEJADO] Permitir ajuste fácil do percentual de corte e análise do impacto na tabela (se voltar a ser relevante).
- Otimizar busca de candles usando o cache (só buscar o que falta)
- Melhorar performance e UX para grandes volumes de dados
- (Opcional futuro) Migrar cache para SQLite para maior robustez
- Documentar funções utilitárias e APIs usadas
- Adicionar testes unitários para funções críticas
- Melhorar interface e visualização dos resultados

---

## Decisões Técnicas
- **Cache em CSV**: cada par/timeframe terá seu próprio arquivo em uma pasta `cache_klines/`.
- **Limite da API Binance**: candles serão buscados em lotes de 1000, com paginação automática.
- **Filtros**: idade só exclui se conhecida e menor que o mínimo; market cap e volume conforme sliders.
- **Debug**: motivos de exclusão de pares são exibidos no painel para facilitar troubleshooting.
- **Volatilidade aparada**: Para cada janela, calcular volatilidade normal, volatilidade aparada (2,5% e 5%) e diferença. Exibir todas para estudo.
- **Score**: Score = (vol 7d > mediana) + (vol 30d > mediana) + (vol 7d > vol 30d), conforme backup. Exibir na tabela principal.

---

## Como contribuir/usar
- Adicione novas ideias e tarefas na seção "Próximos Passos"
- Use o cache local para evitar sobrecarga de requisições
- Sempre documente mudanças relevantes neste arquivo

---

## Histórico de Mudanças
- 2025-08-08: Adicionado debug detalhado e proposta de cache incremental
- 2025-08-08: Ajuste do filtro de idade e lógica de exclusão
- 2025-08-07: Implementação do comparativo de volatilidade 1d x 1h
- 2025-08-08: Implementado cache incremental de candles (1d e 1h) em arquivos CSV por par/timeframe na pasta `cache_klines/`. Busca só o que falta e atualiza o cache automaticamente. Integração na análise de pares potenciais.
- 2025-08-08: Implementado cache de klines por mês em CSV (cache_klines/{symbol}_{interval}_{YYYYMM}.csv). Busca e atualiza só o mês atual, lê todos os meses para montar o histórico. Se não houver histórico, busca tudo da API e cria os arquivos mensais.
- 2025-08-08: Implementado cálculo de volatilidade aparada (trimmed std) para 2,5% e 5% de corte, exibindo ambas na tabela, além da diferença para a média simples.
- 2025-08-08: Implementado score conforme backup: Score = (vol 7d > mediana) + (vol 30d > mediana) + (vol 7d > vol 30d).
- 2025-08-08: Removidas as colunas de volatilidade aparada (2,5% e 5%) da análise de pares e trades. Agora só é exibido o valor real (cru). Debug dos filtros de pares simplificado. Gráficos de volatilidade adicionados abaixo da tabela de pares. Degrade visual nos top 5 da tabela de pares. Possibilidade de reativar cortes de outliers documentada para estudos futuros.

---

> **Este documento deve ser atualizado a cada mudança relevante no projeto.**
