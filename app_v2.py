import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import json

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Trợ lý Đầu tư AI V2", layout="wide", page_icon="🤖")

st.title("🤖 Trợ lý Đầu tư AI Đa Năng - Phân Tích & Chatbot")
st.write("Hệ thống phân tích kỹ thuật nâng cao kết hợp tin tức thời gian thực và Trợ lý ảo AI (Không phụ thuộc yfinance).")

# Khởi tạo các trạng thái trong bộ nhớ hệ thống (Session State) để tránh mất dữ liệu khi chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = {}
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "watchlist_results" not in st.session_state:
    st.session_state.watchlist_results = []
if "is_watchlist_scanned" not in st.session_state:
    st.session_state.is_watchlist_scanned = False

# --- CÁC HÀM XỬ LÝ TOÁN HỌC & KỸ THUẬT ---
def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    
    return df

def generate_strategy(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = float(latest['Close'])
    rsi = float(latest['RSI'])
    ema20 = float(latest['EMA20'])
    ema50 = float(latest['EMA50'])
    macd = float(latest['MACD'])
    signal = float(latest['Signal_Line'])
    bb_upper = float(latest['BB_Upper'])
    bb_lower = float(latest['BB_Lower'])
    
    latest_vol = float(latest['Volume'])
    avg_vol = float(latest['Vol_MA20'])
    vol_ratio = latest_vol / (avg_vol + 1e-9)
    
    trend = "TĂNG" if ema20 > ema50 else "GIẢM"
    action = "THEO DÕI"
    reason = []
    
    if trend == "TĂNG" and rsi < 42:
        action = "LONG / MUA HOLD"
        reason.append("Xu hướng tăng trung hạn, giá điều chỉnh về vùng hỗ trợ (RSI thấp).")
    elif macd > signal and prev['MACD'] <= prev['Signal_Line'] and trend == "TĂNG":
        action = "LONG (Thuận xu hướng)"
        reason.append("Giao cắt MACD hướng lên (Bullish Crossover) trong xu hướng tăng.")
    elif close <= bb_lower:
        action = "CÂN NHẮC MUA / LONG"
        reason.append("Giá chạm dải dưới Bollinger Bands (Quá bán kỹ thuật).")
        
    elif trend == "GIẢM" and rsi > 58:
        action = "SHORT / BÁN CHỐT LỜI"
        reason.append("Xu hướng giảm trung hạn, giá hồi phục ngắn hạn (RSI cao).")
    elif macd < signal Glen and prev['MACD'] >= prev['Signal_Line'] and trend == "GIẢM":
        action = "SHORT (Bán khống)"
        reason.append("Giao cắt MACD hướng xuống (Bearish Crossover) trong xu hướng giảm.")
    elif close >= bb_upper:
        action = "CÂN NHẮC BÁN / SHORT"
        reason.append("Giá chạm dải trên Bollinger Bands (Quá mua kỹ thuật).")
        
    vol_status = "Bình thường"
    if vol_ratio > 1.5:
        vol_status = "Đột biến"
        reason.append(f"Khối lượng giao dịch đột biến gấp {round(vol_ratio, 1)} lần trung bình 20 ngày.")
        
    return {
        "price": round(close, 2),
        "rsi": round(rsi, 2),
        "trend": trend,
        "vol_ratio": round(vol_ratio, 2),
        "vol_status": vol_status,
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "action": action,
        "reason": " & ".join(reason) if reason else "Chưa xuất hiện tín hiệu kích hoạt rõ ràng."
    }

# --- KẾT NỐI CHATBOT GEMINI AI ---
def call_gemini_api(api_key, model_name, system_prompt, user_question):
    api_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"System Context: {system_prompt}\n\nUser Question: {user_question}"}
                ]
            }
        ]
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Lỗi kết nối AI (Mã lỗi: {response.status_code}).\nChi tiết: {response.text}\n\nMẹo: Bạn hãy thử đổi dòng mô hình AI sang 'gemini-2.5-flash-lite' ở sườn trái để có hạn mức cao hơn."
    except Exception as e:
        return f"Không thể kết nối đến máy chủ AI: {e}"

# --- GỬI THÔNG BÁO DISCORD ---
def send_discord_notification(webhook_url, ticker, result, market_type):
    webhook_url = webhook_url.strip()
    if not webhook_url:
        return False, "Chưa nhập Webhook URL."
        
    if result['action'] == "THEO DÕI":
        return True, "Trạng thái hiện tại là 'THEO DÕI' (Không gửi thông báo để tránh loãng phòng chat)."

    is_buy = "LONG" in result['action'] or "MUA" in result['action']
    color_code = 3066993 if is_buy else 15158332 

    price_label = f"${result['price']}" if "Crypto" in market_type else f"{result['price']}k"

    payload = {
        "username": "AI Trading Assistant",
        "embeds": [
            {
                "title": f"📈 TÍN HIỆU GIAO DỊCH MỚI: {ticker}",
                "color": color_code,
                "fields": [
                    {"name": "Giá hiện tại", "value": f"**{price_label}**", "inline": True},
                    {"name": "RSI (14)", "value": f"{result['rsi']}", "inline": True},
                    {"name": "Xu hướng chủ đạo", "value": f"{result['trend']}", "inline": True},
                    {"name": "KHUYẾN NGHỊ HÀNH ĐỘNG", "value": f"🚨 **{result['action']}**", "inline": False},
                    {"name": "Lý do phân tích kỹ thuật", "value": result['reason'], "inline": False}
                ],
                "footer": {
                    "text": "Trading Assistant Bot | Thông báo tự động"
                }
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        if response.status_code == 204:
            return True, "Đã gửi tín hiệu giao dịch mới tới Discord thành công!"
        else:
            return False, f"Lỗi gửi Discord (Mã lỗi: {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"Không thể kết nối đến máy chủ Discord: {e}"


# --- HÀM LẤY DỮ LIỆU GIÁ KHÔNG DÙNG YFINANCE ---
CRYPTO_COINGECKO_ID = {
    "BTC": "bitcoin", "BTC-USD": "bitcoin",
    "ETH": "ethereum", "ETH-USD": "ethereum",
    "BNB": "binancecoin", "BNB-USD": "binancecoin",
    "SOL": "solana", "SOL-USD": "solana",
    "XRP": "ripple", "XRP-USD": "ripple",
    "ADA": "cardano", "ADA-USD": "cardano",
    "DOGE": "dogecoin", "DOGE-USD": "dogecoin",
    "AVAX": "avalanche-2", "AVAX-USD": "avalanche-2",
    "DOT": "polkadot", "DOT-USD": "polkadot",
    "LINK": "chainlink", "LINK-USD": "chainlink",
    "LTC": "litecoin", "LTC-USD": "litecoin",
    "TRX": "tron", "TRX-USD": "tron",
    "TON": "the-open-network", "TON-USD": "the-open-network",
}

def _empty_ohlcv():
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

def fetch_crypto_from_coingecko(ticker_symbol, days=180):
    symbol = ticker_symbol.upper().strip()
    coin_id = CRYPTO_COINGECKO_ID.get(symbol)
    if not coin_id:
        return _empty_ohlcv()

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return _empty_ohlcv()

        df_price = pd.DataFrame(prices, columns=["Date", "Close"])
        df_volume = pd.DataFrame(volumes, columns=["Date", "Volume"])
        df = pd.merge(df_price, df_volume, on="Date", how="left")
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df = df.drop_duplicates("Date").set_index("Date").sort_index()

        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df.dropna()
    except Exception as e:
        st.warning(f"Không lấy được dữ liệu CoinGecko cho {ticker_symbol}: {e}")
        return _empty_ohlcv()

def fetch_stock_from_stooq(ticker_symbol):
    symbol = ticker_symbol.upper().strip()
    if "." not in symbol:
        stooq_symbol = f"{symbol.lower()}.us"
    else:
        stooq_symbol = symbol.lower()

    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        if "No data" in response.text or not response.text.strip():
            return _empty_ohlcv()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if df.empty or not required.issubset(df.columns):
            return _empty_ohlcv()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index().tail(180)
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df.dropna()
    except Exception as e:
        st.warning(f"Không lấy được dữ liệu Stooq cho {ticker_symbol}: {e}")
        return _empty_ohlcv()

def fetch_vn_stock_from_vnstock(ticker_symbol):
    try:
        from vnstock import Market
        m = Market()
        df = m.equity(ticker_symbol.upper().strip()).ohlcv(length=180, interval="1D")
        if df.empty:
            return _empty_ohlcv()
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df.dropna()
    except Exception as e:
        st.warning(f"Không lấy được dữ liệu vnstock cho {ticker_symbol}: {e}")
        return _empty_ohlcv()

def load_market_data(ticker_symbol, market_type):
    symbol = ticker_symbol.upper().strip()
    if "Chứng khoán Việt Nam" in market_type:
        return fetch_vn_stock_from_vnstock(symbol)
    if symbol in CRYPTO_COINGECKO_ID:
        return fetch_crypto_from_coingecko(symbol)
    return fetch_stock_from_stooq(symbol)

# --- TỰ ĐỘNG NHẬN DIỆN THỊ TRƯỜNG ---
def auto_detect_market(ticker_symbol):
    ticker_symbol = ticker_symbol.upper().strip()
    if len(ticker_symbol) == 3 and "-" not in ticker_symbol:
        return "Chứng khoán Việt Nam"
    return "Crypto & Quốc tế"

# --- HÀM LẤY TIN TỨC TỪ GOOGLE RSS FREE ---
def clean_news_title(title):
    if not title:
        return ""
    return title.strip()

def fetch_google_news_rss(query, source_name="Google News", limit=5, lang="vi", country="VN"):
    news_items = []
    try:
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl={lang}&gl={country}&ceid={country}:{lang}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            rss_source = item.findtext("source", default=source_name).strip() or source_name
            if title and link:
                news_items.append({
                    "title": clean_news_title(title),
                    "link": link,
                    "source": rss_source,
                    "published": pub_date,
                })
    except Exception:
        pass
    return news_items

def get_news_sources_for_ticker(ticker_symbol, market_type):
    ticker_clean = ticker_symbol.replace(".VN", "").upper()

    if "Chứng khoán Việt Nam" in market_type:
        return [
            ("Google News", f'"{ticker_clean}" cổ phiếu OR chứng khoán'),
            ("CafeF", f'{ticker_clean} site:cafef.vn'),
            ("Vietstock", f'{ticker_clean} site:vietstock.vn'),
            ("FireAnt", f'{ticker_clean} site:fireant.vn'),
            ("Stockbiz", f'{ticker_clean} site:stockbiz.vn OR site:en.stockbiz.vn'),
            ("SSI iBoard", f'{ticker_clean} site:iboard.ssi.com.vn'),
            ("NDH", f'{ticker_clean} site:ndh.vn'),
        ]

    ticker_no_suffix = ticker_symbol.upper()
    company_query = ticker_no_suffix.replace("-USD", "")
    return [
        ("Google News", f'{ticker_no_suffix} stock OR crypto'),
        ("Nasdaq", f'{ticker_no_suffix} site:nasdaq.com'),
        ("MarketWatch", f'{ticker_no_suffix} site:marketwatch.com'),
        ("CNBC", f'{company_query} site:cnbc.com'),
        ("CoinDesk", f'{company_query} site:coindesk.com'),
    ]

def aggregate_free_news(ticker_symbol, market_type, limit=8):
    all_news = []
    seen_links = set()
    seen_titles = set()

    all_sources_news = []
    for source_name, query in get_news_sources_for_ticker(ticker_symbol, market_type):
        all_sources_news.extend(fetch_google_news_rss(query, source_name=source_name, limit=4))

    for item in all_sources_news:
        title_key = item.get("title", "").lower().strip()
        link_key = item.get("link", "").strip()
        if not title_key or not link_key:
            continue
        if link_key in seen_links or title_key in seen_titles:
            continue
        seen_links.add(link_key)
        seen_titles.add(title_key)
        all_news.append(item)
        if len(all_news) >= limit:
            break

    return all_news

def render_source_shortcuts(ticker_symbol, market_type):
    ticker_clean = ticker_symbol.replace(".VN", "").upper()
    if "Chứng khoán Việt Nam" in market_type:
        st.markdown("**Mở nhanh nguồn tin free:**")
        st.markdown(
            f"[CafeF](https://www.google.com/search?q={quote_plus(ticker_clean + ' CafeF')}) | "
            f"[Vietstock](https://www.google.com/search?q={quote_plus(ticker_clean + ' Vietstock')}) | "
            f"[FireAnt](https://www.google.com/search?q={quote_plus(ticker_clean + ' FireAnt')}) | "
            f"[Stockbiz](https://www.google.com/search?q={quote_plus(ticker_clean + ' Stockbiz')}) | "
            f"[SSI iBoard](https://iboard.ssi.com.vn/) | "
            f"[Google News](https://news.google.com/search?q={quote_plus(ticker_clean + ' cổ phiếu')})"
        )
    else:
        st.markdown("**Mở nhanh nguồn tin free:**")
        st.markdown(
            f"[Google Finance](https://www.google.com/finance/quote/{ticker_clean}) | "
            f"[Google News](https://news.google.com/search?q={quote_plus(ticker_clean)}) | "
            f"[Nasdaq](https://www.google.com/search?q={quote_plus(ticker_clean + ' Nasdaq')}) | "
            f"[MarketWatch](https://www.google.com/search?q={quote_plus(ticker_clean + ' MarketWatch')}) | "
            f"[CNBC](https://www.google.com/search?q={quote_plus(ticker_clean + ' CNBC')})"
        )

# --- GIAO DIỆN SƯỜN TRÁI (SIDEBAR) ---
st.sidebar.header("⚙️ Cấu hình Hệ thống")

# Khóa API và Webhook Discord
gemini_api_key = st.sidebar.text_input("Dán Gemini API Key vào đây:", type="password")
st.sidebar.markdown("[Lấy Gemini API Key miễn phí tại đây](https://aistudio.google.com/)")

discord_webhook_url = st.sidebar.text_input("Dán Discord Webhook URL vào đây (Tùy chọn):", type="password", help="Để nhận thông báo tự động về điện thoại khi có tín hiệu Mua/Bán.")

# Mô hình Gemini năm 2026
gemini_model = st.sidebar.selectbox(
    "Chọn dòng mô hình AI:", 
    ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash")
)

st.sidebar.divider()
st.sidebar.subheader("🌟 PHÂN TÍCH ĐƠN LẺ")
market_type = st.sidebar.radio("Chọn thị trường đơn lẻ:", ("Crypto & Quốc tế", "Chứng khoán Việt Nam"))

if market_type == "Crypto & Quốc tế":
    ticker = st.sidebar.text_input("Nhập mã đơn lẻ (BTC-USD, ETH-USD, AAPL, TSLA...):", "BTC-USD").upper()
else:
    ticker = st.sidebar.text_input("Nhập mã cổ phiếu VN đơn lẻ (FPT, HPG, TCB...):", "FPT").upper()

single_btn = st.sidebar.button("🚀 Phân tích Mã đơn lẻ")

st.sidebar.divider()
st.sidebar.subheader("⭐ DANH SÁCH YÊU THÍCH (WATCHLIST)")
watchlist_input = st.sidebar.text_area(
    "Nhập các mã ngăn cách bằng dấu phẩy:", 
    value="BTC-USD, ETH-USD, FPT, HPG",
    help="Hệ thống tự nhận diện Crypto hay cổ phiếu VN. Ví dụ: BTC-USD, ETH-USD, FPT, HPG"
)
watchlist_btn = st.sidebar.button("🔍 Quét Toàn bộ Danh sách")


# --- XỬ LÝ SỰ KIỆN NHẤN NÚT PHÂN TÍCH ĐƠN LẺ ---
if single_btn:
    st.session_state.is_watchlist_scanned = False
    with st.spinner("Đang xử lý dữ liệu chuyên sâu..."):
        try:
            df = load_market_data(ticker, market_type)
            
            if df.empty:
                st.error("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã.")
                st.session_state.analyzed = False
            else:
                # Tính toán kỹ thuật
                df = calculate_indicators(df)
                result = generate_strategy(df)
                
                # Gửi thông báo Discord nếu có cấu hình
                discord_status = None
                if discord_webhook_url.strip():
                    success, msg = send_discord_notification(discord_webhook_url, ticker, result, market_type)
                    discord_status = ("success", msg) if success else ("error", msg)
                
                # Lưu thông tin số liệu hiện tại
                st.session_state.current_analysis = {
                    "ticker": ticker,
                    "price": result['price'],
                    "rsi": result['rsi'],
                    "trend": result['trend'],
                    "vol_ratio": result['vol_ratio'],
                    "action": result['action'],
                    "reason": result['reason']
                }
                
                # Lưu kết quả
                st.session_state.analysis_results = {
                    "result": result,
                    "chart_data": df[['Close', 'BB_Upper', 'BB_Lower']].tail(60),
                    "market_type": market_type,
                    "ticker": ticker,
                    "discord_status": discord_status
                }
                st.session_state.analyzed = True
                
        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
            st.session_state.analyzed = False


# --- XỬ LÝ SỰ KIỆN QUÉT DANH SÁCH YÊU THÍCH (WATCHLIST) ---
if watchlist_btn:
    st.session_state.analyzed = False
    watchlist_tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(watchlist_tickers):
        status_text.text(f"⏳ Đang quét mã: {t} ({i+1}/{len(watchlist_tickers)})")
        progress_bar.progress((i + 1) / len(watchlist_tickers))
        
        m_type = auto_detect_market(t)
        try:
            df = load_market_data(t, m_type)
            
            if df.empty:
                results.append({"Mã": t, "Giá hiện tại": "Lỗi tải", "RSI (14)": "N/A", "Xu hướng": "N/A", "Đề xuất hành động": "Sai mã hoặc lỗi mạng", "Discord": "N/A"})
            else:
                df = calculate_indicators(df)
                res = generate_strategy(df)
                
                # Tự động gửi Discord cho từng mã
                discord_status = "Chưa cấu hình"
                if discord_webhook_url.strip():
                    if res['action'] != "THEO DÕI":
                        success, msg = send_discord_notification(discord_webhook_url, t, res, m_type)
                        discord_status = "✅ Đã gửi tín hiệu" if success else f"❌ Lỗi: {msg}"
                    else:
                        discord_status = "⏳ Theo dõi (Không gửi)"
                        
                results.append({
                    "Mã": t,
                    "Giá hiện tại": f"${res['price']}" if m_type == "Crypto & Quốc tế" else f"{res['price']}k",
                    "RSI (14)": res['rsi'],
                    "Xu hướng": res['trend'],
                    "Đề xuất hành động": res['action'],
                    "Discord": discord_status
                })
        except Exception as e:
            results.append({"Mã": t, "Giá hiện tại": "Lỗi", "RSI (14)": "N/A", "Xu hướng": "N/A", "Đề xuất hành động": f"Lỗi hệ thống: {e}", "Discord": "N/A"})
            
    progress_bar.empty()
    status_text.empty()
    st.session_state.watchlist_results = results
    st.session_state.is_watchlist_scanned = True


# --- HIỂN THỊ KẾT QUẢ DANH SÁCH YÊU THÍCH (WATCHLIST) ---
if st.session_state.is_watchlist_scanned and st.session_state.watchlist_results:
    st.subheader("📋 Bảng Tổng hợp Kết quả Quét Danh sách Yêu thích")
    st.write("Hệ thống đã quét nhanh dữ liệu kỹ thuật qua nguồn ngoài và gửi thông báo Discord cho các mã có tín hiệu.")
    
    df_watchlist = pd.DataFrame(st.session_state.watchlist_results)
    st.dataframe(df_watchlist, use_container_width=True)


# --- HIỂN THỊ KẾT QUẢ PHÂN TÍCH ĐƠN LẺ ---
if st.session_state.analyzed and st.session_state.analysis_results:
    data = st.session_state.analysis_results
    result = data["result"]
    ticker_name = data["ticker"]
    m_type = data["market_type"]
    
    st.success(f"Phân tích hoàn tất cho mã: {ticker_name}")
    
    # Hiển thị trạng thái thông báo Discord nếu được cấu hình
    discord_status = data.get("discord_status")
    if discord_status:
        status_type, status_msg = discord_status
        if status_type == "success":
            if "Không gửi thông báo" in status_msg:
                st.info(f"ℹ️ {status_msg}")
            else:
                st.success(f"🔔 {status_msg}")
        else:
            st.error(f"❌ {status_msg}")
            
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá hiện tại", f"${result['price']}" if "Crypto" in m_type else f"{result['price']}k")
    col2.metric("RSI (14)", f"{result['rsi']}")
    col3.metric("Xu hướng chính", result['trend'])
    col4.metric("Đột biến Volume", f"{result['vol_ratio']}x")
    
    st.info(f"👉 **Đề xuất chiến lược:** {result['action']}")
    st.write(f"📝 **Lý do chi tiết:** {result['reason']}")
    
    st.subheader("📊 Biểu đồ giá & Dải biến động (Bollinger Bands)")
    st.line_chart(data["chart_data"])
    
    # Tải tin tức liên quan từ nhiều nguồn free
    st.subheader("📰 Tin tức mới cập nhật về mã này")
    with st.spinner("Đang quét tin từ Google News RSS và các nguồn miễn phí..."):
        valid_news = aggregate_free_news(ticker_name, m_type, limit=8)

    if valid_news:
        for item in valid_news:
            st.markdown(f"🔹 **[{item.get('title')}]({item.get('link')})**")
            source_text = item.get('source', 'Tin tức')
            published_text = item.get('published', '')
            if published_text:
                st.caption(f"Nguồn: {source_text} | Thời gian: {published_text}")
            else:
                st.caption(f"Nguồn: {source_text}")
    else:
        st.warning("Chưa tìm thấy bài viết mới từ các nguồn tự động. Bạn có thể mở nhanh các nguồn bên dưới để kiểm tra thủ công.")
        render_source_shortcuts(ticker_name, m_type)


# --- PHẦN KHUNG CHAT TRÒ CHUYỆN AI ---
st.divider()
st.subheader("💬 Trò chuyện với Trợ lý ảo AI về mã này")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Nhận câu hỏi mới từ người dùng
if user_input := st.chat_input("Hãy hỏi trợ lý AI (ví dụ: 'Mã này có rủi ro gì không?', 'Giải thích chỉ số RSI của mã hiện tại')..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        analysis_data = st.session_state.current_analysis
        
        if not analysis_data:
            response_text = "Chào bạn! Hãy nhấn nút '🚀 Phân tích Mã đơn lẻ' ở sườn trái trước, sau đó tôi sẽ có dữ liệu cụ thể để tư vấn kỹ hơn cho bạn nhé!"
        elif not gemini_api_key:
            response_text = f"""⚠️ **Chatbot AI chuyên sâu chưa được kích hoạt.**  
            Số liệu kỹ thuật hiện tại của mã gần nhất **{analysis_data['ticker']}**:  
            * **Giá:** {analysis_data['price']}  
            * **RSI:** {analysis_data['rsi']} (Xu hướng chính: {analysis_data['trend']})  
            * **Khuyến nghị:** {analysis_data['action']}  
            
            *Để tôi tư vấn thông minh và giải thích chuyên sâu hơn, bạn hãy dán **Gemini API Key** ở sườn trái vào nhé!*"""
        else:
            with st.spinner("AI đang phân tích dữ liệu..."):
                system_prompt = f"""
                Bạn là một chuyên gia phân tích tài chính AI chuyên nghiệp. Bạn đang hỗ trợ người dùng phân tích mã giao dịch {analysis_data['ticker']}.
                Dưới đây là các thông số kỹ thuật thực tế hiện tại của mã này:
                - Giá hiện tại: {analysis_data['price']}
                - RSI (14 ngày): {analysis_data['rsi']}
                - Xu hướng chính (Đường trung bình EMA): {analysis_data['trend']}
                - Khối lượng giao dịch so với trung bình: gấp {analysis_data['vol_ratio']} lần.
                - Đề xuất kỹ thuật cơ bản: {analysis_data['action']} (Lý do: {analysis_data['reason']})
                
                Hãy kết hợp kiến thức tài chính sâu rộng của bạn và các số liệu trên để giải thích, đưa ra lời khuyên sâu sắc, phân tích các rủi ro hoặc cơ hội khi người dùng hỏi. Trả lời bằng tiếng Việt, giọng điệu chuyên nghiệp, khách quan. Luôn nhắc nhở người dùng quản lý vốn và đặt cắt lỗ (Stop loss).
                """
                response_text = call_gemini_api(gemini_api_key, gemini_model, system_prompt, user_input)
        
        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})