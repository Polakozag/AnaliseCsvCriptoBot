# === UTILITÁRIO DE CACHE DE KLINES ===

# === CACHE DE KLINES POR MÊS ===
import os
from datetime import datetime, timedelta
def get_binance_klines_monthly_cache(symbol, interval, total_needed):
    """
    Busca candles da Binance com cache incremental em CSV por mês.
    Salva/atualiza em cache_klines/{symbol}_{interval}_{YYYYMM}.csv
    Retorna lista de candles (mais recentes primeiro).
    """
    cache_dir = 'cache_klines'
    os.makedirs(cache_dir, exist_ok=True)
    # Descobre meses necessários (do mais antigo ao atual)
    now = datetime.utcnow()
    months_needed = []
    dt = now
    candles_per_month = {}
    # Aproximação: 1d=30, 1h=30*24
    if interval == '1d':
        step = timedelta(days=30)
        candles_per_month = 30
    elif interval == '1h':
        step = timedelta(days=30)
        candles_per_month = 30*24
    else:
        step = timedelta(days=30)
        candles_per_month = 30
    months = []
    for i in range((total_needed // candles_per_month) + 2):
        ym = dt.strftime('%Y%m')
        months.append(ym)
        dt = dt - step
    months = sorted(set(months))
    # Lê todos os arquivos do par/timeframe
    all_klines = []
    for ym in months:
        cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_{ym}.csv")
        if os.path.exists(cache_file):
            try:
                kl = pd.read_csv(cache_file, header=None).values.tolist()
                all_klines += kl
            except Exception:
                pass
    # Se não houver nada, busca tudo da API (paginado)
    if len(all_klines) == 0:
        # Busca do mais antigo ao mais recente
        end_time = int(now.timestamp() * 1000)
        candles = []
        while len(candles) < total_needed:
            limit = min(1000, total_needed - len(candles))
            params = {'symbol': symbol, 'interval': interval, 'limit': str(limit), 'endTime': end_time}
            url = "https://api.binance.com/api/v3/klines"
            try:
                resp = requests.get(url, params=params)
                resp.raise_for_status()
                new_klines = resp.json()
                if not new_klines:
                    break
                candles = new_klines + candles
                end_time = int(new_klines[0][0]) - 1
            except Exception:
                break
        # Salva por mês
        for k in candles:
            dt_k = datetime.utcfromtimestamp(int(k[0]) / 1000)
            ym = dt_k.strftime('%Y%m')
            cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_{ym}.csv")
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, header=None)
                df = pd.concat([df, pd.DataFrame([k])], ignore_index=True)
            else:
                df = pd.DataFrame([k])
            df.drop_duplicates(subset=[0], inplace=True)
            df.sort_values(by=0, inplace=True)
            df.to_csv(cache_file, index=False, header=False)
        all_klines = candles
    else:
        # Atualiza só o mês atual
        ym_now = now.strftime('%Y%m')
        cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_{ym_now}.csv")
        last_time = None
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, header=None)
                if not df.empty:
                    last_time = int(df.iloc[-1, 0])
            except Exception:
                last_time = None
        # Busca novos candles se necessário
        if last_time:
            while True:
                params = {'symbol': symbol, 'interval': interval, 'limit': '1000', 'startTime': last_time + 1}
                url = "https://api.binance.com/api/v3/klines"
                try:
                    resp = requests.get(url, params=params)
                    resp.raise_for_status()
                    new_klines = resp.json()
                    if not new_klines:
                        break
                    df_new = pd.DataFrame(new_klines)
                    df_new.to_csv(cache_file, mode='a', index=False, header=False)
                    all_klines += new_klines
                    last_time = int(new_klines[-1][0])
                    if len(new_klines) < 1000:
                        break
                except Exception:
                    break
        # Releitura para garantir tudo atualizado
        all_klines = []
        for ym in months:
            cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_{ym}.csv")
            if os.path.exists(cache_file):
                try:
                    kl = pd.read_csv(cache_file, header=None).values.tolist()
                    all_klines += kl
                except Exception:
                    pass
    # Ordena e retorna os mais recentes
    all_klines = sorted(all_klines, key=lambda x: int(x[0]))
    return all_klines[-total_needed:]
import streamlit as st
import pandas as pd
import io
import numpy as np
import plotly.express as px
from datetime import datetime
import requests
from pycoingecko import CoinGeckoAPI
import re

# Funções utilitárias de estatística aparada

def get_trimmed_std(series, trim_perc=0.05):
    """Calcula o desvio padrão aparado, removendo trim_perc das pontas."""
    arr = np.sort(series.dropna())
    n = len(arr)
    k = int(n * trim_perc)
    if n < 2 * k + 1:
        return np.nan
    arr = arr[k:n - k]
    return np.std(arr)

def get_trimmed_mean(series, trim_perc=0.025):
    arr = np.sort(series.dropna())
    n = len(arr)
    k = int(n * trim_perc)
    if n < 2 * k + 1:
        return np.nan
    arr = arr[k:n - k]
    return np.mean(arr)

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

def format_duration_horas(dur_horas):
    if pd.isna(dur_horas) or dur_horas < 0:
        return "N/A"
    total_min = int(round(dur_horas * 60))
    dias = total_min // (24*60)
    horas = (total_min % (24*60)) // 60
    minutos = total_min % 60
    if dias > 0:
        return f"{dias}d {horas}h {minutos}min"
    elif horas > 0:
        return f"{horas}h {minutos}min"
    else:
        return f"{minutos}min"

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
    # Hide USDT from pair names for display
    df['Par_Display'] = df['Par'].str.replace('USDT$', '', regex=True)
    
    col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
    with col_filtro1:
        pares_disponiveis = ['Todos'] + sorted(df['Par_Display'].unique().tolist())
        par_selecionado = st.selectbox("Selecione um Par:", pares_disponiveis)
    with col_filtro2:
        data_min = df['Data'].min().date()
        data_max = df['Data'].max().date()
        # Use Brazilian date format
        start_date = st.date_input("Data de Início", value=data_min, min_value=data_min, max_value=data_max, format="DD/MM/YYYY")
    with col_filtro3:
        end_date = st.date_input("Data de Fim", value=data_max, min_value=data_min, max_value=data_max, format="DD/MM/YYYY")

    df_filtrado = df.copy()
    if par_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Par_Display'] == par_selecionado]
    
    df_filtrado = df_filtrado[(df_filtrado['Data'].dt.date >= start_date) & (df_filtrado['Data'].dt.date <= end_date)]

    if not df_filtrado.empty:
        st.subheader("Resumo dos Resultados")
        lucro_col = 'Lucro ($)'
        rent_col = 'Rentabilidade (%)'

        col1, col2, col3, col4 = st.columns(4)
        total_lucro = df_filtrado[lucro_col].sum()
        total_trades = len(df_filtrado)
        rentabilidade_media = df_filtrado[rent_col].mean()
        duracao_media_horas = df_filtrado['Duração em Segundos'].mean() / 3600 if 'Duração em Segundos' in df_filtrado else 1
        duracao_media_str = format_duration_horas(duracao_media_horas)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Lucro Total</div><div class="metric-card-value" style="color:{PROFIT_COLOR};">${total_lucro:,.2f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Total de Trades</div><div class="metric-card-value" style="color:{TRADES_COLOR};">{total_trades}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Rentabilidade Média</div><div class="metric-card-value" style="color:{RENTABILIDADE_COLOR};">{rentabilidade_media:.2f}%</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-card-label">Duração Média dos Trades</div><div class="metric-card-value" style="color:#4fa3e3;">{duracao_media_str}</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="text-align:center; margin-top:2em; margin-bottom:-1.5em;"><h4>Lucro Cumulativo</h4></div>', unsafe_allow_html=True)
        daily_profit = df_filtrado.groupby(df_filtrado['Data'].dt.date)[lucro_col].sum().reset_index()
        daily_profit['Lucro Cumulativo'] = daily_profit[lucro_col].cumsum()
        daily_profit['Data'] = pd.to_datetime(daily_profit['Data']).dt.strftime('%d %b %y')
        fig_cumulative = px.line(daily_profit, x='Data', y='Lucro Cumulativo', template='plotly_dark', color_discrete_sequence=[PROFIT_COLOR])
        fig_cumulative.update_traces(hovertemplate='%{x}: %{y:.2f}')
        fig_cumulative.update_layout(xaxis_title=None)
        st.plotly_chart(fig_cumulative, use_container_width=True)

        col_chart1, col_chart2 = st.columns(2, gap="medium")
        with col_chart1:
            st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Lucro Total por Par</h4></div>', unsafe_allow_html=True)
            lucro_por_par = df_filtrado.groupby('Par_Display')[lucro_col].sum().reset_index().sort_values(by=lucro_col, ascending=False)
            fig = px.bar(lucro_por_par, x='Par_Display', y=lucro_col, color_discrete_sequence=[PROFIT_COLOR], template='plotly_dark')
            fig.update_traces(hovertemplate='%{x}: %{y:.2f}')
            fig.update_layout(xaxis_title=None, margin=dict(l=30, r=10, t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)
        with col_chart2:
            st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Rentabilidade Média por Par</h4></div>', unsafe_allow_html=True)
            rentabilidade_por_par = df_filtrado.groupby('Par_Display')[rent_col].mean().reset_index().sort_values(by=rent_col, ascending=False)
            fig = px.bar(rentabilidade_por_par, x='Par_Display', y=rent_col, color_discrete_sequence=[RENTABILIDADE_COLOR], template='plotly_dark')
            fig.update_traces(hovertemplate='%{x}: %{y:.2f}')
            fig.update_layout(xaxis_title=None, margin=dict(l=30, r=10, t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

        dias_mapeados = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
        dias_ordem = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        df_filtrado['Dia da Semana'] = df_filtrado['Data'].dt.day_name().map(dias_mapeados)
        bins_horas = np.arange(0, 26, 2)
        labels_horas = [f'{h:02d}-{(h+2)%24:02d}h' for h in np.arange(0, 24, 2)]
        df_filtrado['Intervalo de Horas'] = pd.cut(df_filtrado['Data'].dt.hour, bins=bins_horas, labels=labels_horas, right=False, ordered=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Lucro Total por Dia da Semana</h4></div>', unsafe_allow_html=True)
            lucro_por_dia = df_filtrado.groupby('Dia da Semana')[lucro_col].sum().reindex(dias_ordem).fillna(0).reset_index()
            fig = px.bar(lucro_por_dia, x='Dia da Semana', y=lucro_col, color_discrete_sequence=[PROFIT_COLOR], template='plotly_dark')
            fig.update_traces(hovertemplate='%{y:.2f}')
            fig.update_layout(xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Trades por Dia da Semana</h4></div>', unsafe_allow_html=True)
            trades_por_dia = df_filtrado.groupby('Dia da Semana')[lucro_col].count().reindex(dias_ordem).fillna(0).reset_index(name='Quantidade de Trades')
            fig = px.bar(trades_por_dia, x='Dia da Semana', y='Quantidade de Trades', color_discrete_sequence=[TRADES_COLOR], template='plotly_dark')
            fig.update_traces(hovertemplate='%{y:d}')
            fig.update_layout(xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Lucro Total por Intervalo de Horas</h4></div>', unsafe_allow_html=True)
            lucro_por_hora = df_filtrado.groupby('Intervalo de Horas', observed=True)[lucro_col].sum().reset_index()
            fig = px.bar(lucro_por_hora, x='Intervalo de Horas', y=lucro_col, color_discrete_sequence=[PROFIT_COLOR], template='plotly_dark')
            fig.update_traces(hovertemplate='%{y:.2f}')
            fig.update_layout(xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Trades por Intervalo de Horas</h4></div>', unsafe_allow_html=True)
            trades_por_hora = df_filtrado.groupby('Intervalo de Horas', observed=True)[lucro_col].count().reset_index(name='Quantidade de Trades')
            fig = px.bar(trades_por_hora, x='Intervalo de Horas', y='Quantidade de Trades', color_discrete_sequence=[TRADES_COLOR], template='plotly_dark')
            fig.update_traces(hovertemplate='%{y:d}')
            fig.update_layout(xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div style="text-align:center; margin-bottom:0.5em;"><h4>Tabela de Trades</h4></div>', unsafe_allow_html=True)
        df_trades = df_filtrado.drop(columns=['Duração em Segundos'], errors='ignore').copy()
        if 'Data' in df_trades.columns:
            df_trades['Data'] = pd.to_datetime(df_trades['Data']).dt.strftime('%d %b %y')
        if 'Par_Display' in df_trades.columns:
            df_trades = df_trades.rename(columns={'Par_Display': 'Par'})
            if list(df_trades.columns).count('Par') > 1:
                df_trades = df_trades.loc[:,~df_trades.columns.duplicated()]
        st.dataframe(df_trades, use_container_width=True)
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
# Só libera análise de pares após upload do CSV
if uploaded_file is None or 'df' not in locals():
    st.info("Para analisar novos pares, primeiro carregue seu arquivo de trades históricos (CSV) na seção 1.")
    st.stop()

st.subheader("Filtros para Pares")
col1, col2, col3 = st.columns(3)
with col1:
    min_age_years = st.slider("Idade Mínima (anos)", min_value=1, max_value=10, value=5, step=1, key="slider_idade_pares")
with col2:
    min_market_cap_mil = st.slider("Market Cap Mínimo (M$)", min_value=0, max_value=1000, value=1000, step=50, key="slider_marketcap_pares")
with col3:
    min_volume_mil = st.slider("Volume Mínimo 24h (M$)", min_value=0, max_value=1000, value=500, step=50, key="slider_volume_pares")

# Botão único para análise comparativa
if st.button("Buscar e Analisar Pares", use_container_width=True, key="buscar_pares_unico"):
    import time
    periods = {'7d': 7, '30d': 30, '60d': 60, '90d': 90, '6m': 180, '1a': 365}
    if 'pares_potenciais_df' in st.session_state:
        del st.session_state['pares_potenciais_df']
    consulta_start = time.time()
    with st.spinner("Analisando o mercado... Isso pode levar alguns segundos ou minutos."):
        market_data = get_coingecko_market_data()
        if not market_data:
            st.error("Falha ao buscar dados da CoinGecko.")
            st.stop()
        df_gecko = pd.DataFrame(market_data)[['id', 'symbol', 'name', 'market_cap', 'total_volume']]
        df_gecko['market_cap'] = pd.to_numeric(df_gecko['market_cap'], errors='coerce')
        df_gecko['total_volume'] = pd.to_numeric(df_gecko['total_volume'], errors='coerce')
        df_gecko.dropna(inplace=True)
        df_gecko['symbol'] = df_gecko['symbol'].str.upper()
        # Filtro market cap, volume e stablecoins
        filtered_gecko = df_gecko[
            (df_gecko['market_cap'] >= min_market_cap_mil * 1_000_000) &
            (df_gecko['total_volume'] >= min_volume_mil * 1_000_000)
        ].copy()
        # Filtro de stablecoins antes de tudo
        stablecoins = set(['USDC', 'FDUSD', 'BUSD', 'USD1', 'USDT', 'BSC-USD'])
        filtered_gecko = filtered_gecko[~filtered_gecko['symbol'].isin(stablecoins)]
        exchange_info = get_binance_exchange_info()
        if not exchange_info:
            st.error("Falha ao buscar dados da Binance.")
            st.stop()
        binance_usdt_pairs = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        filtered_gecko['binance_symbol'] = filtered_gecko['symbol'] + 'USDT'
        # Só mantém pares listados na Binance
        final_candidates = filtered_gecko[filtered_gecko['binance_symbol'].isin(binance_usdt_pairs)]
        # === DEBUG MELHORADO PARA FILTRO DE PARES ===
        # (Removido: não exibir candidatos após filtro market cap, volume e Binance durante o processamento)
        # Após todos os filtros, exibir apenas candidatos finais aprovados
        aprovados = []
        results = []
        for _, row in final_candidates.iterrows():
            coin_details = get_coingecko_coin_details(row['id'])
            age_in_years = np.nan
            if coin_details and coin_details.get('genesis_date'):
                genesis_date = datetime.strptime(coin_details['genesis_date'], "%Y-%m-%d")
                age_in_years = (datetime.now() - genesis_date).days / 365.25
            if pd.notna(age_in_years) and age_in_years < min_age_years:
                continue
            aprovados.append(row['binance_symbol'])
            # Volatilidade diária (1d)
            klines_1d = get_binance_klines_monthly_cache(row['binance_symbol'], '1d', 370)
            klines_1h = get_binance_klines_monthly_cache(row['binance_symbol'], '1h', 370*24)
            if not klines_1d or len(klines_1d) < 365:
                continue
            if not klines_1h or len(klines_1h) < 365*24:
                continue
            pair_result = {
                'Par': row['binance_symbol'],
                'Idade (anos)': float(age_in_years) if isinstance(age_in_years, float) and not pd.isna(age_in_years) else None,
                'Market Cap Num': row['market_cap'],
                'Volume 24h Num': row['total_volume']
            }
            df_klines_1d = pd.DataFrame(klines_1d, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
            df_klines_1d['close'] = pd.to_numeric(df_klines_1d['close'])
            df_klines_1d['log_return'] = np.log(df_klines_1d['close'] / df_klines_1d['close'].shift(1))
            df_klines_1h = pd.DataFrame(klines_1h, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
            df_klines_1h['close'] = pd.to_numeric(df_klines_1h['close'])
            df_klines_1h['log_return'] = np.log(df_klines_1h['close'] / df_klines_1h['close'].shift(1))
            for name, days in periods.items():
                hours = int(days * 24)
                if len(df_klines_1h) >= hours:
                    vol_1h = df_klines_1h['log_return'].tail(hours).std() * np.sqrt(24) * 100
                    pair_result[f'{name}'] = vol_1h
                else:
                    pair_result[f'{name}'] = None
            results.append(pair_result)
    # Exibição dos resultados
    if results:
        st.markdown(f"Análise concluída! {len(aprovados)} pares potenciais encontrados: " + ", ".join([f"`{p}`" for p in aprovados]))
        df_final = pd.DataFrame(results)
        st.session_state['pares_potenciais_df'] = df_final
    else:
        st.warning("Nenhum par encontrado que satisfaça todos os critérios de filtragem.")




if 'pares_potenciais_df' in st.session_state:
    df_to_display = st.session_state['pares_potenciais_df'].copy()
    # Hide USDT from pair names for display
    df_to_display['Par_Display'] = df_to_display['Par'].str.replace('USDT$', '', regex=True)
    # Score baseado apenas nas colunas 1h (agora sem sufixo)
    if 'Score' not in df_to_display:
        if '7d' in df_to_display and '30d' in df_to_display:
            med_7d = df_to_display['7d'].median()
            med_30d = df_to_display['30d'].median()
            df_to_display['Score'] = (
                (df_to_display['7d'] > med_7d).astype(int) +
                (df_to_display['30d'] > med_30d).astype(int) +
                (df_to_display['7d'] > df_to_display['30d']).astype(int)
            )
        else:
            df_to_display['Score'] = 0
    if 'Score' in df_to_display:
        df_to_display.sort_values(by=["Score", "Market Cap Num"], ascending=[False, False], inplace=True)

    # Criar coluna 'Moeda' sem o sufixo USDT
    df_to_display['Moeda'] = df_to_display['Par'].str.replace('USDT$', '', regex=True)
    df_style = df_to_display.copy()
    df_style['Market Cap'] = df_style['Market Cap Num'].apply(format_currency)
    df_style['Volume 24h'] = df_style['Volume 24h Num'].apply(format_currency)
    periods = {'7d': 7, '30d': 30, '60d': 60, '90d': 90, '6m': 180, '1a': 365}
    vol_cols = list(periods.keys())
    if not df_style.empty:
        med_7d = df_style['7d'].median() if '7d' in df_style else 0
        med_30d = df_style['30d'].median() if '30d' in df_style else 0
        df_style['Score'] = (
            (df_style['7d'] > med_7d).astype(int) +
            (df_style['30d'] > med_30d).astype(int) +
            (df_style['7d'] > df_style['30d']).astype(int)
        )
    else:
        df_style['Score'] = 0
    if 'Score' not in df_style:
        df_style['Score'] = 0
    df_style.sort_values(by=["Score", "Market Cap Num"], ascending=[False, False], inplace=True)
    # Remover coluna 'Par' duplicada se existir
    if list(df_style.columns).count('Par') > 1:
        df_style = df_style.loc[:,~df_style.columns.duplicated()]
    df_style = df_style.reset_index(drop=True)
    # Nova ordem: Moeda, Score, 7d, 30d, 60d, 90d, 6m, 1a, Idade (anos), Market Cap, Volume 24h
    display_columns = ["Moeda", "Score"] + vol_cols + ["Idade (anos)", "Market Cap", "Volume 24h"]
    def highlight_gradient(s):
        colors = ['#1a593a', '#14482f', '#0f3723', '#0a2618', '#05150d']
        if s.dtype in ['float64', 'int64']:
            top5 = s.dropna().nlargest(5).unique()
            color_map = {val: color for val, color in zip(top5, colors)}
            return [f'background-color: {color_map.get(v, "")}' for v in s]
        return ['' for _ in s]
    styler = df_style[display_columns].style.apply(highlight_gradient, subset=["Score", "7d", "30d", "90d"])
    st.dataframe(styler, use_container_width=True)

    st.markdown('<h4 style="text-align:center;margin-bottom:0.5em;">Gráficos de Volatilidade por Período</h4>', unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        df_chart_7d = df_style[display_columns].dropna(subset=['7d']).sort_values('7d', ascending=False)
        if not df_chart_7d.empty:
            fig7d = px.bar(df_chart_7d.head(15), x='Moeda', y='7d', title='Top 15 - 7 dias', template='plotly_dark', color_discrete_sequence=['#ff7f0e'],
                hover_data={'7d':':.2f'})
            fig7d.update_traces(hovertemplate='%{x}: %{y:.2f}%')
            fig7d.update_layout(title_x=0.5, xaxis_title=None)
            st.plotly_chart(fig7d, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para o gráfico de 7 dias.")
    with g_col2:
        df_chart_30d = df_style[display_columns].dropna(subset=['30d']).sort_values('30d', ascending=False)
        if not df_chart_30d.empty:
            fig30d = px.bar(df_chart_30d.head(15), x='Moeda', y='30d', title='Top 15 - 30 dias', template='plotly_dark', color_discrete_sequence=['#1f77b4'],
                hover_data={'30d':':.2f'})
            fig30d.update_traces(hovertemplate='%{x}: %{y:.2f}%')
            fig30d.update_layout(title_x=0.5, xaxis_title=None)
            st.plotly_chart(fig30d, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para o gráfico de 30 dias.")
    other_vol_columns = {'60 dias': '60d', '90 dias': '90d', '6 meses': '6m', '1 ano': '1a'}
    if any(col in df_style[display_columns] for col in other_vol_columns.values()):
        tabs = st.tabs(other_vol_columns.keys())
        for i, (tab_name, col_name) in enumerate(other_vol_columns.items()):
            with tabs[i]:
                df_chart = df_style[display_columns].dropna(subset=[col_name]).sort_values(col_name, ascending=False)
                if not df_chart.empty:
                    fig = px.bar(df_chart.head(15), x='Moeda', y=col_name, title=f'Top 15 - {tab_name}', template='plotly_dark', color_discrete_sequence=['#28a745'],
                        hover_data={col_name:':.2f'})
                    fig.update_traces(hovertemplate='%{x}: %{y:.2f}%')
                    fig.update_layout(title_x=0.5, xaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Não há dados suficientes para o gráfico de {tab_name}.")