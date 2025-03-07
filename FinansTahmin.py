# 0. Gerekli kütüphaneleri import edelim
import requests
import pandas as pd 
import plotly.graph_objs as go 
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Kullanıcıdan coin sembolünü alalım
coin_symbol = input("Lütfen analiz yapmak istediğiniz coin sembolünü girin (örneğin: aca): ").lower()

# Tüm coinlerin listesini alalım
url = "https://api.coingecko.com/api/v3/coins/list"
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"API isteği başarısız oldu, durum kodu: {response.status_code}")

coin_list = response.json()

# Kullanıcının girdiği coin sembolüne göre coin kimliğini bulalım
selected_coin = next((coin for coin in coin_list if coin['symbol'] == coin_symbol), None)

if selected_coin is None:
    raise Exception(f"{coin_symbol.upper()} Coin bulunamadı.")

print(f"{coin_symbol.upper()} Coin ID: {selected_coin['id']}")

# 1. CoinGecko API'den son 6 aylık seçilen coin fiyatlarını TL cinsinden çekelim
url = f"https://api.coingecko.com/api/v3/coins/{selected_coin['id']}/market_chart"
params = {
    "vs_currency": "try",
    "days" : "180",
    "interval": "daily"
}

response = requests.get(url, params=params)

# API isteğinin başarılı olup olmadığını kontrol edin
if response.status_code != 200:
    raise Exception(f"API isteği başarısız oldu, durum kodu: {response.status_code}")

veri = response.json()

# API yanıtını yazdır
print(veri)

if "prices" not in veri:
    raise KeyError("API yanıtında 'prices' anahtarı bulunamadı.")

# 2. Veriyi Pandas DataFrame'e çevirelim
fiyatlar = veri["prices"]
df = pd.DataFrame(fiyatlar, columns=["zaman_damgasi", "fiyat"])

# Zaman damgasını datetime formatına dönüştürün ve olası hataları ele alın
try:
    df["zaman_damgasi"] = pd.to_datetime(df["zaman_damgasi"], unit="ms")
except Exception as e:
    raise ValueError(f"Zaman damgası dönüştürme hatası: {e}")

df.set_index("zaman_damgasi", inplace=True)

# 3. Fiyat sütununu sayısal formata dönüştürün
df["fiyat"] = pd.to_numeric(df["fiyat"], errors="coerce")

# 4. Veriyi kontrol et
print(f"Nan veya hatalı değer sayısı: {df.isnull().sum()}")

# NaN değerleri içeren satırları düşürün
df.dropna(inplace=True)

# 5. Günlük frekans belirleyin
df = df.asfreq("D")

# 6. SARIMA modelini oluşturun
model = SARIMAX(df["fiyat"], order=(2,1,2), seasonal_order=(1,1,1,30))
model_fit = model.fit(disp=False)

# 7. Önümüzdeki 30 gün için tahmin yapalım
tahmin = model_fit.forecast(steps=30)
tahmin_indeksi = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
tahmin_serisi = pd.Series(tahmin, index=tahmin_indeksi)

# 8. Tahmini grafiğe ekleyelim
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = df.index,
    y = df["fiyat"],
    mode = "lines",
    name = f"{coin_symbol.upper()} Coin TL verileri",
    line = dict(color="blue"),
))
fig.add_trace(go.Scatter(
    x = tahmin_serisi.index,
    y = tahmin_serisi,
    mode = "lines",
    name = f"30 Günlük {coin_symbol.upper()} Coin Tahmini",
    line = dict(color="red", dash="dash")
))

# 9. Grafik gösterimi
fig.update_layout(
    title= f"{coin_symbol.upper()} Coin Fiyatları - son 6 aylık veri",
    xaxis_title = "Tarih",
    yaxis_title = "Fiyat TL",
    hovermode = "x",
    height = 600,
    template = "plotly_dark"
)

fig.show()
