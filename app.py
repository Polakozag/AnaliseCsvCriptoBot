import streamlit as st
import pandas as pd
import io
import numpy as np
import plotly.express as px
from datetime import datetime
import requests
from pycoingecko import CoinGeckoAPI
import re

# --- CONFIGURAÇÕES INICIAIS E ESTILO ---
st.set_page_config(
    page_title="Dashboard Criptobot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Definindo cores e constantes
PROFIT_COLOR = '#28a745'
TRADES_COLOR = '#1f77b4'
RENTABILIDADE_COLOR = '#ff7f0e'
CUMULATIVE_COLOR = '#ff7f0e'
METRIC_BG_COLOR = '#262730'
TEXT_COLOR = '#FAFAFA'
STABLECOINS = ['USDC', 'FDUSD', 'BUSD', 'USD1']

# Estilo CSS
st.markdown(f"""
<style>
.metric-card {{
    background-color: {METRIC_BG_COLOR};
    border: 1px solid #3c3f4a;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}}
.metric-card-label {{
    font-size: 1rem;
    color: #a9a9a9;
}}
.metric-card-value {{
    font-size: 2rem;
    font-weight: bold;
}}
div.stButton > button:first-child {{ display: block; margin: 0 auto; }}
</style>
""", unsafe_allow_html=True)


def format_currency(value):
    if pd.isna(value): return "N/A"
    value = float(value)
    if value >= 1_000_000_000_000: return f"$ {value / 1_000_000_000_000:,.2f} tri"
    if value >= 1_000_000_000: return f"$ {value / 1_000_000_000:,.2f} bi"
    if value >= 1_000_000: return f"$ {value / 1_000_000:,.2f} M"
    if value >= 1_000: return f"$ {value / 1_000:,.2f} mil"
    return f"$ {value:,.2f}"

def parse_duration_to_seconds(duration_str):
    try:
        parts = str(duration_str).replace('"', '').split()
        if len(parts) == 3:
            h = int(parts[0].replace('h', ''))
            m = int(parts[1].replace('m', ''))
            s = int(parts[2].replace('s', ''))
            return h * 3600 + m * 60 + s
        return 0
    except (ValueError, IndexError):
        return 0

# --- FUNÇÕES DE API ---
def get_coingecko_market_data():
    try:
        cg = CoinGeckoAPI()
        return cg.get_coins_markets(vs_currency='usd', per_page=250, page=1)
    except Exception as e:
        st.error(f"Erro ao buscar dados da CoinGecko: {e}")
        return None

@st.cache_data(ttl=3600)
def get_binance_exchange_info():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar dados da Binance: {e}")
        return None

@st.cache_data(ttl=3600)
def get_binance_klines(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': str(limit)}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=3600)
def get_coingecko_coin_details(coin_id):
    try:
        cg = CoinGeckoAPI()
        return cg.get_coin_by_id(coin_id)
    except Exception:
        return None
        
# --- TÍTULO ---
st.title("🤖 Dashboard de Análise de Estratégias - Criptobot")
st.markdown("---")

# ==============================================================================
# SEÇÃO 1: ANÁLISE DE TRADES HISTÓRICOS
# ==============================================================================
st.header("1. Análise de Trades Históricos")
uploaded_file = st.file_uploader("Carregue seu arquivo de trades (.csv)", type="csv")

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        lines = file_content.strip().split('\n')
        
        correct_header = ['Data', 'Par', 'Preço Compra ($)', 'Preço Venda ($)', 'Duração', 'Lucro ($)', 'Rentabilidade (%)']
        
        data_lines = lines[1:]
        processed_data_list_of_lists = []
        for line in data_lines:
            if line.strip():
                fields = line.split(',')
                combined_datetime = f"{fields[0].strip()} {fields[1].strip()}"
                rest_of_fields = [field.strip() for field in fields[2:]]
                processed_row = [combined_datetime] + rest_of_fields
                
                if len(processed_row) == len(correct_header):
                    processed_data_list_of_lists.append(processed_row)

        df = pd.DataFrame(processed_data_list_of_lists, columns=correct_header)
        
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        numeric_cols = ['Preço Compra ($)', 'Preço Venda ($)', 'Lucro ($)', 'Rentabilidade (%)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Duração' in df.columns:
            df['Duração em Segundos'] = df['Duração'].apply(parse_duration_to_seconds)

        df.dropna(subset=['Data', 'Par', 'Lucro ($)'], inplace=True)
        
        if df.empty:
            st.error("Nenhuma linha com dados válidos foi encontrada no arquivo CSV após o processamento.")
            st.stop()

    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV. Verifique o formato. Erro: {e}")
        st.stop()
    
    st.subheader("Filtros de Análise Histórica")
    
    col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
    with col_filtro1:
        pares_disponiveis = ['Todos'] + sorted(df['Par'].unique().tolist())
        par_selecionado = st.selectbox("Selecione um Par:", pares_disponiveis)
    with col_filtro2:
        data_min = df['Data'].min().date()
        data_max = df['Data'].max().date()
        start_date = st.date_input("Data de Início", value=data_min, min_value=data_min, max_value=data_max)
    with col_filtro3:
        end_date = st.date_input("Data de Fim", value=data_max, min_value=data_min, max_value=data_max)

    df_filtrado = df.copy()
    if par_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Par'] == par_selecionado]
    
    df_filtrado = df_filtrado[(df_filtrado['Data'].dt.date >= start_date) & (df_filtrado['Data'].dt.date <= end_date)]

    if not df_filtrado.empty:
        st.subheader("Resumo dos Resultados")
        lucro_col = 'Lucro ($)'
        rent_col = 'Rentabilidade (%)'

        col1, col2, col3 = st.columns(3)
        total_lucro = df_filtrado[lucro_col].sum()
        total_trades = len(df_filtrado)
        rentabilidade_media = df_filtrado[rent_col].mean()
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Lucro Total</div><div class="metric-card-value" style="color:{PROFIT_COLOR};">${total_lucro:,.2f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Total de Trades</div><div class="metric-card-value" style="color:{TRADES_COLOR};">{total_trades}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Rentabilidade Média</div><div class="metric-card-value" style="color:{RENTABILIDADE_COLOR};">{rentabilidade_media:.2f}%</div></div>', unsafe_allow_html=True)
        
        st.subheader("Lucro Cumulativo")
        daily_profit = df_filtrado.groupby(df_filtrado['Data'].dt.date)[lucro_col].sum().reset_index()
        daily_profit['Lucro Cumulativo'] = daily_profit[lucro_col].cumsum()
        fig_cumulative = px.line(daily_profit, x='Data', y='Lucro Cumulativo', template='plotly_dark', color_discrete_sequence=[PROFIT_COLOR])
        st.plotly_chart(fig_cumulative, use_container_width=True)

        st.markdown("---")
        st.header("Análises Detalhadas por Par")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Lucro Total por Par")
            lucro_por_par = df_filtrado.groupby('Par')[lucro_col].sum().reset_index().sort_values(by=lucro_col, ascending=False)
            fig = px.bar(lucro_por_par, x='Par', y=lucro_col, color_discrete_sequence=[PROFIT_COLOR], template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.subheader("Rentabilidade Média por Par")
            rentabilidade_por_par = df_filtrado.groupby('Par')[rent_col].mean().reset_index().sort_values(by=rent_col, ascending=False)
            fig = px.bar(rentabilidade_por_par, x='Par', y=rent_col, color_discrete_sequence=[RENTABILIDADE_COLOR], template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("⏰ Análises de Tempo e Frequência")
        
        dias_mapeados = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
        dias_ordem = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        df_filtrado['Dia da Semana'] = df_filtrado['Data'].dt.day_name().map(dias_mapeados)
        
        bins_horas = np.arange(0, 26, 2)
        labels_horas = [f'{h:02d}:00-{(h+2)%24:02d}:00' for h in np.arange(0, 24, 2)]
        df_filtrado['Intervalo de Horas'] = pd.cut(df_filtrado['Data'].dt.hour, bins=bins_horas, labels=labels_horas, right=False, ordered=True)

        col_day_1, col_day_2 = st.columns(2)
        with col_day_1:
            st.subheader("Lucro Total por Dia da Semana")
            lucro_por_dia = df_filtrado.groupby('Dia da Semana')[lucro_col].sum().reindex(dias_ordem).fillna(0).reset_index()
            fig = px.bar(lucro_por_dia, x='Dia da Semana', y=lucro_col, color_discrete_sequence=[PROFIT_COLOR], template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_day_2:
            st.subheader("Trades por Dia da Semana")
            trades_por_dia = df_filtrado.groupby('Dia da Semana')[lucro_col].count().reindex(dias_ordem).fillna(0).reset_index(name='Quantidade de Trades')
            fig = px.bar(trades_por_dia, x='Dia da Semana', y='Quantidade de Trades', color_discrete_sequence=[TRADES_COLOR], template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

        col_hour_1, col_hour_2 = st.columns(2)
        with col_hour_1:
            st.subheader("Lucro Total por Intervalo de Horas")
            lucro_por_hora = df_filtrado.groupby('Intervalo de Horas', observed=True)[lucro_col].sum().reset_index()
            fig = px.bar(lucro_por_hora, x='Intervalo de Horas', y=lucro_col, color_discrete_sequence=[PROFIT_COLOR], template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_hour_2:
            st.subheader("Trades por Intervalo de Horas")
            trades_por_hora = df_filtrado.groupby('Intervalo de Horas', observed=True)[lucro_col].count().reset_index(name='Quantidade de Trades')
            fig = px.bar(trades_por_hora, x='Intervalo de Horas', y='Quantidade de Trades', color_discrete_sequence=[TRADES_COLOR], template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Tabela de Trades")
        st.dataframe(df_filtrado.drop(columns=['Duração em Segundos'], errors='ignore'), use_container_width=True)
    else:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")

st.markdown("---")

# ==============================================================================
# SEÇÃO 2: ANÁLISE DE PARES POTENCIAIS
# ==============================================================================
st.header("2. Análise de Pares Potenciais")
st.markdown("Use os filtros para encontrar novas oportunidades de trading com base em dados de mercado em tempo real.")

def calculate_volatility_score(row, median_vol_7d, median_vol_30d):
    score = 0
    vol_7d = row.get('Vol (7d)')
    vol_30d = row.get('Vol (30d)')

    if pd.notna(vol_7d) and vol_7d > median_vol_7d:
        score += 1
    if pd.notna(vol_30d) and vol_30d > median_vol_30d:
        score += 1
    if pd.notna(vol_7d) and pd.notna(vol_30d) and vol_7d > vol_30d:
        score += 1
    return score

def highlight_top5(s):
    colors = ['#1a593a', '#14482f', '#0f3723', '#0a2618', '#05150d']
    
    if s.dtype in ['float64', 'int64']:
        top5_values = s.dropna().nlargest(5).unique()
        color_map = {val: color for val, color in zip(top5_values, colors)}
        return [f'background-color: {color_map.get(v, "")}' for v in s]
    return ['' for _ in s]

st.subheader("Filtros para Pares")
col1, col2, col3 = st.columns(3)
with col1:
    min_age_years = st.slider("Idade Mínima (anos)", min_value=1, max_value=10, value=5, step=1)
with col2:
    min_market_cap_mil = st.slider("Market Cap Mínimo (M$)", min_value=0, max_value=1000, value=1000, step=50)
with col3:
    min_volume_mil = st.slider("Volume Mínimo 24h (M$)", min_value=0, max_value=1000, value=500, step=50)

if st.button("Buscar e Analisar Pares", use_container_width=True):
    periods = {'Vol (7d)': 7, 'Vol (30d)': 30, 'Vol (60d)': 60, 'Vol (90d)': 90, 'Vol (6m)': 180, 'Vol (1a)': 365}

    if 'pares_potenciais_df' in st.session_state:
        del st.session_state['pares_potenciais_df']
        
    with st.spinner("Analisando o mercado... Este processo pode levar um momento."):
        market_data = get_coingecko_market_data()
        if not market_data:
            st.error("Falha ao buscar dados da CoinGecko.")
            st.stop()
        
        df_gecko = pd.DataFrame(market_data)[['id', 'symbol', 'name', 'market_cap', 'total_volume']]
        df_gecko['market_cap'] = pd.to_numeric(df_gecko['market_cap'], errors='coerce')
        df_gecko['total_volume'] = pd.to_numeric(df_gecko['total_volume'], errors='coerce')
        df_gecko.dropna(inplace=True)
        df_gecko['symbol'] = df_gecko['symbol'].str.upper()
        
        filtered_gecko = df_gecko[
            (df_gecko['market_cap'] >= min_market_cap_mil * 1_000_000) &
            (df_gecko['total_volume'] >= min_volume_mil * 1_000_000) &
            (~df_gecko['symbol'].isin(STABLECOINS))
        ].copy()

        exchange_info = get_binance_exchange_info()
        if not exchange_info:
            st.error("Falha ao buscar dados da Binance.")
            st.stop()
        
        binance_usdt_pairs = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        filtered_gecko['binance_symbol'] = filtered_gecko['symbol'] + 'USDT'
        final_candidates = filtered_gecko[filtered_gecko['binance_symbol'].isin(binance_usdt_pairs)]

        results = []
        for _, row in final_candidates.iterrows():
            coin_details = get_coingecko_coin_details(row['id'])
            age_in_years = np.nan
            if coin_details and coin_details.get('genesis_date'):
                genesis_date = datetime.strptime(coin_details['genesis_date'], "%Y-%m-%d")
                age_in_years = (datetime.now() - genesis_date).days / 365.25
            
            if pd.notna(age_in_years) and age_in_years < min_age_years:
                continue

            klines = get_binance_klines(row['binance_symbol'], '1d', 365)
            
            if not klines or len(klines) < 365:
                continue

            pair_result = {
                'Par': row['binance_symbol'], 'Idade (anos)': age_in_years,
                'Market Cap Num': row['market_cap'], 'Volume 24h Num': row['total_volume']
            }

            df_klines = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
            df_klines['close'] = pd.to_numeric(df_klines['close'])
            df_klines['log_return'] = np.log(df_klines['close'] / df_klines['close'].shift(1))
            
            for name, days in periods.items():
                if len(df_klines) >= days:
                    volatility = np.std(df_klines['log_return'].tail(days)) * 100
                    pair_result[name] = volatility
                else:
                    pair_result[name] = np.nan
            
            results.append(pair_result)
        
        if results:
            st.success(f"Análise concluída! {len(results)} pares potenciais encontrados.")
            df_final = pd.DataFrame(results)
            
            if not df_final.empty and 'Vol (7d)' in df_final.columns and 'Vol (30d)' in df_final.columns:
                median_vol_7d = df_final['Vol (7d)'].median()
                median_vol_30d = df_final['Vol (30d)'].median()
                df_final['Pontuação'] = df_final.apply(calculate_volatility_score, axis=1, median_vol_7d=median_vol_7d, median_vol_30d=median_vol_30d)
            else:
                df_final['Pontuação'] = 0

            st.session_state['pares_potenciais_df'] = df_final
        else:
            st.warning("Nenhum par encontrado que satisfaça todos os critérios de filtragem.")

if 'pares_potenciais_df' in st.session_state:
    df_to_display = st.session_state['pares_potenciais_df'].copy()
    
    df_to_display.sort_values(by=['Pontuação', 'Market Cap Num'], ascending=[False, False], inplace=True)
    
    st.subheader("Ranking de Pares Potenciais")
    
    df_style = df_to_display.copy()
    
    df_style['Market Cap'] = df_style['Market Cap Num'].apply(format_currency)
    df_style['Volume 24h'] = df_style['Volume 24h Num'].apply(format_currency)

    columns_to_style = ['Pontuação', 'Vol (7d)', 'Vol (30d)']
    styler = df_style.style.apply(highlight_top5, subset=columns_to_style)
    
    vol_columns = ['Vol (7d)', 'Vol (30d)', 'Vol (60d)', 'Vol (90d)', 'Vol (6m)', 'Vol (1a)']
    format_dict = {'Idade (anos)': '{:.1f}'}
    for col in vol_columns:
        format_dict[col] = '{:.2f}%'
    
    styler.format(format_dict, na_rep="N/A")
    styler.hide(['Market Cap Num', 'Volume 24h Num'], axis="columns")
    
    # <<< CORREÇÃO: Adiciona a coluna 'Par' à lista de exibição >>>
    display_columns = ["Par", "Pontuação", "Idade (anos)", "Market Cap", "Volume 24h"] + vol_columns
    
    st.dataframe(styler, use_container_width=True, column_order=display_columns)
    
    st.subheader("Gráficos de Volatilidade por Período")
    
    vol_7d_col = 'Vol (7d)'
    vol_30d_col = 'Vol (30d)'

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        df_chart_7d = df_to_display.dropna(subset=[vol_7d_col]).sort_values(vol_7d_col, ascending=False)
        if not df_chart_7d.empty:
            fig7d = px.bar(df_chart_7d.head(15), x='Par', y=vol_7d_col, title='Top 15 - Volatilidade Média (7 dias)',
                           template='plotly_dark', color_discrete_sequence=[RENTABILIDADE_COLOR])
            st.plotly_chart(fig7d, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para o gráfico de 7 dias.")
    
    with g_col2:
        df_chart_30d = df_to_display.dropna(subset=[vol_30d_col]).sort_values(vol_30d_col, ascending=False)
        if not df_chart_30d.empty:
            fig30d = px.bar(df_chart_30d.head(15), x='Par', y=vol_30d_col, title='Top 15 - Volatilidade Média (30 dias)',
                            template='plotly_dark', color_discrete_sequence=[TRADES_COLOR])
            st.plotly_chart(fig30d, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para o gráfico de 30 dias.")

    other_vol_columns = {'60 dias': 'Vol (60d)', '90 dias': 'Vol (90d)', '6 meses': 'Vol (6m)', '1 ano': 'Vol (1a)'}
    if any(col in df_to_display for col in other_vol_columns.values()):
        tabs = st.tabs(other_vol_columns.keys())
        
        for i, (tab_name, col_name) in enumerate(other_vol_columns.items()):
            with tabs[i]:
                df_chart = df_to_display.dropna(subset=[col_name]).sort_values(col_name, ascending=False)
                
                if not df_chart.empty:
                    fig = px.bar(df_chart.head(15), x='Par', y=col_name, title=f'Top 15 - Volatilidade Média ({tab_name})',
                                 template='plotly_dark', color_discrete_sequence=[PROFIT_COLOR])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Não há dados suficientes para o gráfico de {tab_name}.")