@echo off
REM Batch script to generate various Bitcoin synthetic datasets
REM Usage: generate_datasets.bat

echo Generating Bitcoin Synthetic Datasets...
echo.

REM Create data directory if it doesn't exist
if not exist "data" mkdir data

echo Generating UPTREND datasets...
python src/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-01-15 --interval 1m --market-type UPTREND --initial-price 45000 --output data/uptrend_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-02-01 --end-date 2024-02-15 --interval 5m --market-type UPTREND --initial-price 48000 --output data/uptrend_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-03-01 --end-date 2024-03-31 --interval 15m --market-type UPTREND --initial-price 50000 --output data/uptrend_15m_1month.csv

echo.
echo Generating DOWNTREND datasets...
python src/btc_data_generator.py --start-date 2024-04-01 --end-date 2024-04-15 --interval 1m --market-type DOWNTREND --initial-price 65000 --output data/downtrend_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-05-01 --end-date 2024-05-15 --interval 5m --market-type DOWNTREND --initial-price 62000 --output data/downtrend_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-06-01 --end-date 2024-06-30 --interval 15m --market-type DOWNTREND --initial-price 58000 --output data/downtrend_15m_1month.csv

echo.
echo Generating SWING datasets...
python src/btc_data_generator.py --start-date 2024-07-01 --end-date 2024-07-15 --interval 1m --market-type SWING --initial-price 55000 --output data/swing_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-08-01 --end-date 2024-08-15 --interval 5m --market-type SWING --initial-price 54000 --output data/swing_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-09-01 --end-date 2024-09-30 --interval 15m --market-type SWING --initial-price 53000 --output data/swing_15m_1month.csv

echo.
echo Generating MIXED datasets...
python src/btc_data_generator.py --start-date 2024-10-01 --end-date 2024-10-15 --interval 1m --market-type MIXED --initial-price 60000 --output data/mixed_1m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-11-01 --end-date 2024-11-15 --interval 5m --market-type MIXED --initial-price 58000 --output data/mixed_5m_2weeks.csv
python src/btc_data_generator.py --start-date 2024-12-01 --end-date 2024-12-31 --interval 15m --market-type MIXED --initial-price 56000 --output data/mixed_15m_1month.csv

echo.
echo All datasets generated successfully!
echo Check the 'data' folder for the generated CSV files.
pause
