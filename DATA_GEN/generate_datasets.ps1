# PowerShell script to generate various Bitcoin synthetic datasets
# Usage: .\generate_datasets.ps1

Write-Host "Generating Bitcoin Synthetic Datasets..." -ForegroundColor Green
Write-Host ""

# Create data directory if it doesn't exist
if (!(Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

Write-Host "Generating UPTREND datasets..." -ForegroundColor Yellow
python src/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-01-15 --interval 1m --market-type UPTREND --initial-price 45000 --output data/uptrend_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-02-01 --end-date 2024-02-15 --interval 5m --market-type UPTREND --initial-price 48000 --output data/uptrend_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-03-01 --end-date 2024-03-31 --interval 15m --market-type UPTREND --initial-price 50000 --output data/uptrend_15m_1month.csv

Write-Host ""
Write-Host "Generating DOWNTREND datasets..." -ForegroundColor Yellow
python src/btc_data_generator.py --start-date 2024-04-01 --end-date 2024-04-15 --interval 1m --market-type DOWNTREND --initial-price 65000 --output data/downtrend_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-05-01 --end-date 2024-05-15 --interval 5m --market-type DOWNTREND --initial-price 62000 --output data/downtrend_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-06-01 --end-date 2024-06-30 --interval 15m --market-type DOWNTREND --initial-price 58000 --output data/downtrend_15m_1month.csv

Write-Host ""
Write-Host "Generating SWING datasets..." -ForegroundColor Yellow
python src/btc_data_generator.py --start-date 2024-07-01 --end-date 2024-07-15 --interval 1m --market-type SWING --initial-price 55000 --output data/swing_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-08-01 --end-date 2024-08-15 --interval 5m --market-type SWING --initial-price 54000 --output data/swing_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-09-01 --end-date 2024-09-30 --interval 15m --market-type SWING --initial-price 53000 --output data/swing_15m_1month.csv

Write-Host ""
Write-Host "Generating MIXED datasets..." -ForegroundColor Yellow
python src/btc_data_generator.py --start-date 2024-10-01 --end-date 2024-10-15 --interval 1m --market-type MIXED --initial-price 60000 --output data/mixed_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-11-01 --end-date 2024-11-15 --interval 5m --market-type MIXED --initial-price 58000 --output data/mixed_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-12-01 --end-date 2024-12-31 --interval 15m --market-type MIXED --initial-price 56000 --output data/mixed_15m_1month.csv

Write-Host ""
Write-Host "All datasets generated successfully!" -ForegroundColor Green
Write-Host "Check the 'data' folder for the generated CSV files." -ForegroundColor Cyan
