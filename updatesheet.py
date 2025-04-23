import gspread
from google.oauth2.service_account import Credentials
import requests
import time
import yfinance as yf
import pandas as pd
import numpy as np
from functools import wraps
from datetime import datetime, timedelta
import logging

# --- Config ---
SPREADSHEET_ID       = "1bWANjyQeU6srKRZO0fWNFTItd_gR3kbdjsycToaJXKo"
MAIN_SHEET           = "Sheet1"
TEMP_SHEET           = "_Temp"
SERVICE_ACCOUNT_FILE = r"C:\Users\zerou\OneDrive\Documents\StockSpikeReplicator\credentials.json"

API_KEYS = {
    'finnhub':       'YOUR_FINNHUB_KEY',
    'alpha_vantage': 'YOUR_ALPHA_VANTAGE_KEY',
    'openai':        'YOUR_OPENAI_KEY',
    'gemini':        'YOUR_GEMINI_KEY'
}
LIMITS = {'finnhub':30,'alpha_vantage':5,'openai':60,'gemini':60}

# Set up logging
logging.basicConfig(
    filename='stock_update.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rate-limit state
api_calls    = {k: [] for k in LIMITS}
api_failures = {k: {'cnt':0,'until':None,'dead':False} for k in LIMITS}

# Main‐sheet cols
C_PRICE   = "L"
C_TODAY   = "O"
C_SUPPORT = "H"
C_RESIST  = "I"
C_SCORE   = "U"
C_EXP     = "Z"
C_OPENAI  = "P"
C_GEMINI  = "Q"
C_RSI     = "R"   # New column for RSI
C_SHARPE  = "S"   # New column for Sharpe ratio
C_SQUEEZE = "T"   # New column for Squeeze indicator
C_DRAWDOWN= "V"   # New column for Drawdown%
C_BETA    = "W"   # New column for Beta
C_MA_RATIO= "X"   # New column for MA50/200
C_MOMENTUM= "Y"   # New column for Momentum%
C_PE      = "AA"  # New column for P/E
C_DE      = "AB"  # New column for D/E

# Temp header row
HEADER = [
    'Symbol','Support (20d)','Resistance (20d)','RSI','Sharpe',
    'Squeeze?','Drawdown%','Beta','MA50/200','Momentum%',
    'P/E','D/E','Score','Verdict','OpenAI','Gemini'
]

# --- Utility Functions ---
def validate_numeric(value, default=0):
    """Ensure a value is numeric and not None/empty"""
    if value is None:
        return default
    try:
        val = float(value)
        return val if not np.isnan(val) else default
    except:
        return default
        
def validate_string(value, default='N/A'):
    """Ensure a value is a valid string"""
    if value is None or value == '':
        return default
    return str(value)

def check_rate(api):
    now = time.time()
    api_calls[api] = [t for t in api_calls[api] if now-t < 60]
    info = api_failures[api]
    if info['dead'] or (info['until'] and datetime.now()<info['until']):
        return False
    if len(api_calls[api]) < LIMITS[api]:
        api_calls[api].append(now)
        return True
    return False

def record_fail(api, code):
    info = api_failures[api]
    if code == 429:
        info['cnt'] += 1
        if api in ['openai','gemini'] and info['cnt'] > 2:
            info['dead'] = True
        elif info['cnt'] == 3:
            info['until'] = datetime.now() + timedelta(minutes=1)

def retry(api):
    def decorator(fn):
        @wraps(fn)
        def wrapper(sym):
            if not check_rate(api):
                return 'RateLimit'
            try:
                return fn(sym)
            except requests.HTTPError as e:
                record_fail(api, e.response.status_code)
                logging.error(f"HTTP Error in {fn.__name__} for {sym}: {e.response.status_code}")
                return 'N/A' if e.response.status_code == 404 else 'Error'
            except Exception as e:
                record_fail(api, 429)
                logging.error(f"Exception in {fn.__name__} for {sym}: {str(e)}")
                return 'Error'
        return wrapper
    return decorator

# --- Indicators ---
def calculate_rsi(p):
    try:
        if len(p) < 15:
            return 50  # Default to neutral if not enough data
            
        deltas = np.diff(p)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)
        avg_up = ups[-14:].mean()
        avg_down = downs[-14:].mean() or 1e-5
        rs = avg_up/avg_down
        return 100 - (100/(1+rs))
    except Exception as e:
        logging.error(f"RSI calculation error: {str(e)}")
        return 50

def calculate_sharpe(r):
    try:
        rf = (1+0.01)**(1/252) - 1
        m, s = r.mean(), r.std()
        return 0 if s==0 else ((m-rf)/s)*np.sqrt(252)
    except Exception as e:
        logging.error(f"Sharpe calculation error: {str(e)}")
        return 0

def detect_squeeze(p):
    try:
        s = pd.Series(p)
        if len(s) < 21: 
            return False
        bw = s.rolling(20).std() * 4
        return bw.iloc[-1] < bw.mean()*0.6
    except Exception as e:
        logging.error(f"Squeeze detection error: {str(e)}")
        return False

def max_drawdown(p):
    try:
        if not p or len(p) < 2:
            return 0
        peak, md = p[0], 0
        for x in p:
            peak = max(peak, x)
            md = max(md, (peak-x)/peak) if peak > 0 else 0
        return md*100
    except Exception as e:
        logging.error(f"Max drawdown calculation error: {str(e)}")
        return 0

def calc_moving_avg_ratio(p):
    try:
        s = pd.Series(p)
        if len(s) < 200: 
            # Return 1.0 as neutral if not enough data
            if len(s) >= 50:
                # Can at least calculate MA50
                return s.rolling(50).mean().iloc[-1] / s.iloc[-1]
            return 1.0
        ma50 = s.rolling(50).mean().iloc[-1]
        ma200 = s.rolling(200).mean().iloc[-1]
        if ma200 == 0:
            return 1.0
        return ma50 / ma200
    except Exception as e:
        logging.error(f"Moving average ratio calculation error: {str(e)}")
        return 1.0

def momentum(p):
    try:
        if len(p) > 20:
            if p[-20] == 0:
                return 0
            return (p[-1]/p[-20] - 1)*100
        return 0
    except Exception as e:
        logging.error(f"Momentum calculation error: {str(e)}")
        return 0

# --- Enhanced Technical Indicators ---
def calculate_macd(prices):
    try:
        if len(prices) < 26:
            return 0, 0
            
        df = pd.DataFrame({'close': prices})
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]
    except Exception as e:
        logging.error(f"MACD calculation error: {str(e)}")
        return 0, 0

def calculate_bollinger_bands(prices, window=20):
    try:
        if len(prices) < window:
            return prices[-1] if len(prices) > 0 else 0, 0, 0
            
        series = pd.Series(prices)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        if rolling_mean.iloc[-1] > 0:
            return rolling_mean.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]
        return 0, 0, 0
    except Exception as e:
        logging.error(f"Bollinger bands calculation error: {str(e)}")
        return 0, 0, 0

# --- Data fetchers ---
@retry('finnhub')
def fetch_profile(sym):
    r = requests.get(
        "https://finnhub.io/api/v1/stock/profile2",
        params={'symbol':sym,'token':API_KEYS['finnhub']}
    )
    r.raise_for_status()
    return r.json() or {}

@retry('alpha_vantage')
def fetch_overview(sym):
    r = requests.get(
        "https://www.alphavantage.co/query",
        params={'function':'OVERVIEW','symbol':sym,'apikey':API_KEYS['alpha_vantage']}
    )
    r.raise_for_status()
    return r.json() or {}

@retry('alpha_vantage')
def fetch_pe(sym):
    data = fetch_overview(sym)
    pe = data.get('PERatio', '0')
    if pe == 'None' or not pe:
        return 0
    try:
        return float(pe)
    except:
        return 0

@retry('alpha_vantage')
def fetch_de(sym):
    data = fetch_overview(sym)
    de = data.get('DebtToEquity', '0')
    if de == 'None' or not de:
        return 0
    try:
        return float(de)
    except:
        return 0

@retry('openai')
def fetch_openai(sym):
    payload = {
        'model':'gpt-3.5-turbo',
        'messages':[{'role':'user','content':f"Buy or Sell {sym}? Reply 'Buy' or 'Sell'."}]
    }
    h = {'Authorization':f"Bearer {API_KEYS['openai']}"}
    r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=h)
    r.raise_for_status()
    content = r.json()['choices'][0]['message']['content'].strip()
    # Normalize the response
    if 'buy' in content.lower():
        return 'Buy'
    elif 'sell' in content.lower():
        return 'Sell'
    else:
        return 'Hold'  # Default if response isn't clear

@retry('gemini')
def fetch_gemini(sym):
    body = {
        'model':'chat-bison-001',
        'prompt':{'messages':[{'author':'user','content':f"Buy or Sell {sym}? Reply 'Buy' or 'Sell'."}]}
    }
    h = {'Authorization':f"Bearer {API_KEYS['gemini']}"}
    r = requests.post('https://gemini.googleapis.com/v1/models/chat-bison-001:generateMessage', json=body, headers=h)
    r.raise_for_status()
    content = r.json()['candidates'][0]['content'].strip()
    # Normalize the response
    if 'buy' in content.lower():
        return 'Buy'
    elif 'sell' in content.lower():
        return 'Sell'
    else:
        return 'Hold'  # Default if response isn't clear

# --- Sheets helpers ---
def init_sheets():
    try:
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
        )
        client = gspread.authorize(creds)
        ss = client.open_by_key(SPREADSHEET_ID)
        return ss.worksheet(MAIN_SHEET), ss.worksheet(TEMP_SHEET)
    except Exception as e:
        logging.error(f"Failed to initialize sheets: {str(e)}")
        raise

def retry_on_429(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        for i in range(3):
            try:
                return fn(*a, **k)
            except gspread.exceptions.APIError as e:
                if e.response.status_code == 429:
                    wait_time = 2**i
                    logging.warning(f"Google Sheets API rate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Google Sheets API error: {e}")
                    raise
        logging.error("Google Sheets API 429 retry failed after 3 attempts")
        raise Exception("Sheets 429 retry failed")
    return wrapper

@retry_on_429
def batch_write(ws, updates):
    if updates:
        ws.batch_update(updates)
        time.sleep(1)

# --- Enhanced Stock Analysis Functions ---
def calculate_comprehensive_score(metrics):
    """Calculate a more comprehensive stock score using multiple metrics"""
    try:
        # Weights for different factors
        weights = {
            'technical': 0.6,
            'fundamental': 0.3,
            'sentiment': 0.1
        }
        
        # Technical components
        rsi_score = 0
        if metrics['rsi'] <= 30:  # Oversold
            rsi_score = 0.8
        elif metrics['rsi'] >= 70:  # Overbought
            rsi_score = 0.2
        else:  # Neutral zone gets middle score
            rsi_score = 0.5
            
        # Handle sharpe ratio - higher is better
        sharpe_score = min(max(metrics['sharpe'], 0), 3) / 3
        
        # Drawdown - lower is better
        drawdown_score = 1 - min(metrics['drawdown'] / 100, 1)
        
        # MA ratio - above 1 is bullish
        ma_score = 0.8 if metrics['ma_ratio'] > 1.05 else (0.2 if metrics['ma_ratio'] < 0.95 else 0.5)
        
        # Momentum - higher is better
        mom_score = min(max(metrics['momentum'] / 20 + 0.5, 0), 1)
        
        # Technical score
        technical_score = (
            rsi_score * 0.2 +
            sharpe_score * 0.2 +
            drawdown_score * 0.2 +
            ma_score * 0.2 +
            mom_score * 0.2
        ) * weights['technical']
        
        # Fundamental components
        pe_score = 0
        if metrics['pe'] > 0:  # Valid P/E
            # Lower P/E is generally better, but too low might be a red flag
            # Scoring curve: highest around 10-15, lower for very low or high P/E
            if metrics['pe'] < 5:
                pe_score = 0.5  # Potentially undervalued or troubled
            elif metrics['pe'] < 15:
                pe_score = 0.9  # Good value zone
            elif metrics['pe'] < 25:
                pe_score = 0.7  # Reasonably valued
            elif metrics['pe'] < 50:
                pe_score = 0.4  # Getting expensive
            else:
                pe_score = 0.2  # Very expensive
        else:
            pe_score = 0.3  # No P/E (could be no earnings)
        
        # D/E ratio - lower is generally better for stability
        de_score = 0
        if metrics['de'] > 0:  # Valid D/E
            if metrics['de'] < 0.5:
                de_score = 0.9  # Very low debt
            elif metrics['de'] < 1:
                de_score = 0.8  # Low debt
            elif metrics['de'] < 2:
                de_score = 0.6  # Moderate debt
            else:
                de_score = 0.3  # High debt
        else:
            de_score = 0.5  # No D/E data
            
        # Fundamental score
        fundamental_score = (pe_score * 0.6 + de_score * 0.4) * weights['fundamental']
        
        # Sentiment components - from AI models
        openai_score = 1 if 'buy' in metrics['openai'].lower() else (0 if 'sell' in metrics['openai'].lower() else 0.5)
        gemini_score = 1 if 'buy' in metrics['gemini'].lower() else (0 if 'sell' in metrics['gemini'].lower() else 0.5)
        sentiment_score = (openai_score * 0.5 + gemini_score * 0.5) * weights['sentiment']
        
        # Calculate final score
        total_score = technical_score + fundamental_score + sentiment_score
        return round(total_score, 3)
    except Exception as e:
        logging.error(f"Score calculation error: {str(e)}")
        return 0

# --- Main update ---
def update():
    try:
        main, tmp = init_sheets()
        rows = main.get_all_values()[1:]
        total = len(rows)
        logging.info(f"Starting update of {total} symbols at {datetime.now():%H:%M:%S}")
        print(f"🚀 Starting update of {total} symbols at {datetime.now():%H:%M:%S}")

        # clear old
        to_clear = [C_PRICE,C_TODAY,C_SUPPORT,C_RESIST,C_SCORE,C_EXP,C_OPENAI,C_GEMINI]
        main.batch_clear([f"{c}2:{c}{total+1}" for c in to_clear])

        # reset temp
        tmp.clear()
        time.sleep(1)
        batch_write(tmp, [{'range':'A1:P1','values':[HEADER]}])

        temp_buf, main_buf = [], []
        batch_size = 5

        for idx, row in enumerate(rows, start=2):
            # Initialize ALL values with defaults to avoid blanks
            sym = row[0].strip() if len(row) > 0 else "UNKNOWN"
            date_str = row[2].strip() if len(row) > 2 else ""
            
            # Default values for everything
            price_on_c = 'N/A'
            support = 0
            resistance = 0
            rsi = 50  # default to neutral
            macd_val = 0
            signal_val = 0
            bb_middle, bb_upper, bb_lower = 0, 0, 0
            sharpe = 0
            squeeze = 'No'
            dd = 0  # drawdown
            beta = 0
            mar = 1.0  # MA50/200 ratio - default to neutral
            mom = 0  # momentum
            pe_val = 0
            de_val = 0
            score = 0
            exp = 'Hold'  # default to Hold rather than N/A
            ai_o = 'Hold'
            ai_g = 'Hold'

            print(f"[{idx-1}/{total}] {sym} — fetching price data...", end="\r")
            logging.info(f"Processing {sym} (row {idx})")
            
            # price on date
            try:
                dt = datetime.strptime(date_str, '%d/%m/%Y')
                d1 = yf.Ticker(sym).history(start=dt, end=dt+timedelta(days=1))
                if not d1.empty:
                    price_on_c = round(d1['Close'].iloc[0], 2)
            except Exception as e:
                logging.error(f"Error fetching date price for {sym}: {str(e)}")
                price_on_c = 'N/A'  # Use default if error

            # Get historical data with better error handling
            hist = []
            try:
                ticker_data = yf.Ticker(sym).history(period='1y')
                if not ticker_data.empty:
                    hist = ticker_data['Close'].tolist()
            except Exception as e:
                logging.error(f"Error fetching history for {sym}: {str(e)}")
                hist = []

            # Process metrics only if we have sufficient data
            if len(hist) >= 30:
                try:
                    support = round(min(hist[-20:]), 2)
                    resistance = round(max(hist[-20:]), 2)
                except Exception as e:
                    logging.error(f"Support/resistance calculation error for {sym}: {str(e)}")
                    support, resistance = 0, 0
                    
                try:
                    rsi = round(calculate_rsi(hist), 2)
                except Exception as e:
                    logging.error(f"RSI calculation error for {sym}: {str(e)}")
                    rsi = 50
                
                try:
                    macd_val, signal_val = calculate_macd(hist)
                    macd_val = round(macd_val, 3)
                    signal_val = round(signal_val, 3)
                except Exception as e:
                    logging.error(f"MACD calculation error for {sym}: {str(e)}")
                    macd_val, signal_val = 0, 0
                
                try:
                    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(hist)
                    bb_middle = round(bb_middle, 2)
                    bb_upper = round(bb_upper, 2)
                    bb_lower = round(bb_lower, 2)
                except Exception as e:
                    logging.error(f"Bollinger bands calculation error for {sym}: {str(e)}")
                    bb_middle, bb_upper, bb_lower = 0, 0, 0
                    
                try:
                    sharpe = round(calculate_sharpe(pd.Series(hist).pct_change().dropna()), 3)
                except Exception as e:
                    logging.error(f"Sharpe calculation error for {sym}: {str(e)}")
                    sharpe = 0
                    
                try:
                    squeeze = 'Yes' if detect_squeeze(hist) else 'No'
                except Exception as e:
                    logging.error(f"Squeeze detection error for {sym}: {str(e)}")
                    squeeze = 'No'
                    
                try:
                    dd = round(max_drawdown(hist), 2)
                except Exception as e:
                    logging.error(f"Drawdown calculation error for {sym}: {str(e)}")
                    dd = 0
                    
                try:
                    mar = round(calc_moving_avg_ratio(hist), 3)
                except Exception as e:
                    logging.error(f"MA ratio calculation error for {sym}: {str(e)}")
                    mar = 1.0
                    
                try:
                    mom = round(momentum(hist), 3)
                except Exception as e:
                    logging.error(f"Momentum calculation error for {sym}: {str(e)}")
                    mom = 0

                # Get beta - isolate in try/except to prevent failing other calculations
                try:
                    prof = fetch_profile(sym)
                    if isinstance(prof, dict):
                        try:
                            beta = round(float(prof.get('beta', 0) or 0), 3)
                        except:
                            beta = 0
                except Exception as e:
                    logging.error(f"Beta fetch error for {sym}: {str(e)}")
                    beta = 0

                # Get P/E - isolate each API call
                try:
                    raw_pe = fetch_pe(sym)
                    pe_val = round(float(raw_pe or 0), 2)
                except Exception as e:
                    logging.error(f"P/E fetch error for {sym}: {str(e)}")
                    pe_val = 0

                # Get D/E - isolate each API call
                try:
                    raw_de = fetch_de(sym)
                    de_val = round(float(raw_de or 0), 2)
                except Exception as e:
                    logging.error(f"D/E fetch error for {sym}: {str(e)}")
                    de_val = 0

                # Get AI recommendations - isolate API calls
                try:
                    ai_o = fetch_openai(sym)
                    if not ai_o or ai_o in ['RateLimit', 'Error', 'N/A']:
                        ai_o = 'Hold'  # Default to Hold on errors
                except Exception as e:
                    logging.error(f"OpenAI fetch error for {sym}: {str(e)}")
                    ai_o = 'Hold'
                    
                try:
                    ai_g = fetch_gemini(sym)
                    if not ai_g or ai_g in ['RateLimit', 'Error', 'N/A']:
                        ai_g = 'Hold'  # Default to Hold on errors
                except Exception as e:
                    logging.error(f"Gemini fetch error for {sym}: {str(e)}")
                    ai_g = 'Hold'

                # Calculate comprehensive score with all metrics
                metrics = {
                    'rsi': rsi,
                    'sharpe': sharpe,
                    'drawdown': dd,
                    'ma_ratio': mar,
                    'momentum': mom,
                    'pe': pe_val,
                    'de': de_val,
                    'openai': ai_o,
                    'gemini': ai_g
                }
                
                try:
                    score = calculate_comprehensive_score(metrics)
                except Exception as e:
                    logging.error(f"Score calculation error for {sym}: {str(e)}")
                    score = 0
                    
                # Set verdict based on score
                if score > 0.7:
                    exp = 'Strong Buy'
                elif score > 0.5:
                    exp = 'Buy'
                elif score > 0.3:
                    exp = 'Hold'
                else:
                    exp = 'Sell'

            print(f"[{idx-1}/{total}] {sym} | price={price_on_c} sup={support} res={resistance} score={score}      ")

            # Ensure all values are defined before adding to buffers
            temp_buf.append({
                'range': f'A{idx}:P{idx}',
                'values': [[
                    sym, 
                    support if support != 0 else 0,  # Force zeros instead of blanks
                    resistance if resistance != 0 else 0,
                    rsi if rsi != 0 else 0,
                    round(sharpe, 3) if sharpe != 0 else 0,
                    squeeze,
                    round(dd, 2) if dd != 0 else 0,
                    round(beta, 3) if beta != 0 else 0,
                    round(mar, 3) if mar != 0 else 0,
                    round(mom, 3) if mom != 0 else 0,
                    round(pe_val, 2) if pe_val != 0 else 0,
                    round(de_val, 2) if de_val != 0 else 0,
                    score if score != 0 else 0,
                    exp, 
                    ai_o, 
                    ai_g
                ]]
            })

            # Ensure main buffer has all values
            today = datetime.now().strftime('%Y-%m-%d')
            main_buf += [
                {'range':f'{C_TODAY}{idx}','values':[[today]]},
                {'range':f'{C_PRICE}{idx}','values':[[price_on_c]]},
                {'range':f'{C_SUPPORT}{idx}','values':[[support if support != 0 else 0]]},
                {'range':f'{C_RESIST}{idx}','values':[[resistance if resistance != 0 else 0]]},
                {'range':f'{C_SCORE}{idx}','values':[[score if score != 0 else 0]]},
                {'range':f'{C_EXP}{idx}','values':[[exp]]},
                {'range':f'{C_OPENAI}{idx}','values':[[ai_o]]},
                {'range':f'{C_GEMINI}{idx}','values':[[ai_g]]},
                {'range':f'{C_RSI}{idx}','values':[[rsi if rsi != 0 else 0]]},
                {'range':f'{C_SHARPE}{idx}','values':[[round(sharpe, 3) if sharpe != 0 else 0]]},
                {'range':f'{C_SQUEEZE}{idx}','values':[[squeeze]]},
                {'range':f'{C_DRAWDOWN}{idx}','values':[[round(dd, 2) if dd != 0 else 0]]},
                {'range':f'{C_BETA}{idx}','values':[[round(beta, 3) if beta != 0 else 0]]},
                {'range':f'{C_MA_RATIO}{idx}','values':[[round(mar, 3) if mar != 0 else 0]]},
                {'range':f'{C_MOMENTUM}{idx}','values':[[round(mom, 3) if mom != 0 else 0]]},
                {'range':f'{C_PE}{idx}','values':[[round(pe_val, 2) if pe_val != 0 else 0]]},
                {'range':f'{C_DE}{idx}','values':[[round(de_val, 2) if de_val != 0 else 0]]}
            ]

            # flush chunks
            if len(temp_buf) >= batch_size:
                batch_write(tmp, temp_buf)
                print(f"Flushed temp up to row {idx}")
                temp_buf = []
            if len(main_buf) >= batch_size:
                batch_write(main, main_buf)
                print(f"Flushed main up to row {idx}")
                main_buf = []

        # final flush
        if temp_buf:
            batch_write(tmp, temp_buf)
            print("Final temp flush")
        if main_buf:
            batch_write(main, main_buf)
            print("Final main flush")

        logging.info(f"Update loop completed at {datetime.now():%H:%M:%S}")
        print(f"🔔 Update loop done at {datetime.now():%H:%M:%S}'")
        return True
    except Exception as e:
        logging.critical(f"Critical error in update function: {str(e)}")
        print(f"❌ Critical error: {str(e)}")
        return False

if __name__ == '__main__':
    try:
        logging.info("Stock update script started")
        success = update()
        if success:
            logging.info("Update completed successfully")
            print("🎉 Update completed SUCCESSFULLY.")
        else:
            logging.error("Update finished but reported failure")
            print("❌ Update finished but reported failure.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {str(e)}")
        print(f"❌ Update FAILED with error: {e}")