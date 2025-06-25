"""
Data Management Module for RL Trading Bot
Handles data downloading, processing, and validation
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv

from ..data.download_coinapi import CoinAPIDataDownloader
from ..data.setup_data import DataProcessor

class DataManager:
    """
    Manages data downloading, processing, and validation
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize data manager
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_processor = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download and prepare training data using CoinAPI only
        
        Returns:
            Raw OHLCV DataFrame
        """
        self.logger.info("Starting data download...")
        
        # Check CoinAPI key
        coinapi_key = os.getenv('COINAPI_API_KEY')
        if not coinapi_key:
            raise ValueError("COINAPI_API_KEY is required. Please set it in your .env file")
        
        symbol = self.config.get('symbol', 'BINANCEFTS_PERP_BTC_USDT')
        interval = self.config.get('interval', '15m')
        
        self.logger.info(f"CoinAPI key available: Yes")
        self.logger.info(f"CoinAPI key starts with: {coinapi_key[:8]}...")
        
        # Use CoinAPI with daily batch download
        try:
            self.logger.info("Using CoinAPI as data source")
            coinapi_downloader = CoinAPIDataDownloader()
            
            # Test connection
            if not coinapi_downloader.test_api_connection():
                raise ValueError("CoinAPI connection failed")
            
            df = self._download_from_coinapi_with_daily_batches(coinapi_downloader, symbol, interval)
            
            if df is None or len(df) == 0:
                raise ValueError("CoinAPI returned no data")
                
        except Exception as e:
            self.logger.error(f"CoinAPI download failed: {e}")
            raise ValueError(f"Failed to download data from CoinAPI: {e}")
        
        # Final validation
        if df is None or len(df) == 0:
            self.logger.error("No data could be downloaded")
            raise ValueError(f"No data received for {symbol}. Check symbol name, API key, and network connection.")
        
        # Validate data
        if hasattr(coinapi_downloader, 'validate_data'):
            is_valid, issues = coinapi_downloader.validate_data(df)
            if not is_valid:
                self.logger.warning(f"Data validation issues: {issues}")
        
        self.logger.info(f"Downloaded {len(df)} data points for {symbol}")
        return df
        
    def _download_from_coinapi_with_daily_batches(self, downloader, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Download data from CoinAPI using daily batch method with memory limits"""
        try:
            use_date_range = self.config.get('use_date_range', False)
            
            if use_date_range:
                start_date = self.config.get('start_date')
                end_date = self.config.get('end_date')
                
                # Validate and adjust dates
                start_date, end_date = self._validate_date_range(start_date, end_date)
                
                # Use daily batch download
                success = downloader.download_daily_data_batch(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    exchange='BINANCE'
                )
                
                if success:
                    print(f"✅ Daily batch download completed successfully")
                    df = self._load_data_with_memory_checks(downloader, symbol, interval, start_date, end_date)
                    return df
                else:
                    print(f"❌ Daily batch download failed")
                    return None
            else:
                # Use recent data
                days = min(self.config.get('data_days', 7), 14)
                success = downloader.download_recent_batch(
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    exchange='BINANCE'                )
                
                if success:
                    end_date = pd.Timestamp.now() - pd.Timedelta(days=1)
                    start_date = end_date - pd.Timedelta(days=days)
                    
                    df = self._load_data_with_memory_checks(
                        downloader, symbol, interval, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    return df
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"CoinAPI download failed: {e}")
            return None

    def _validate_date_range(self, start_date: str, end_date: str) -> tuple:
        """Validate date range - removed artificial memory limits, now supports multi-year training"""
        current_date = pd.Timestamp.now()
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate total days for information
        total_days = (end_dt - start_dt).days + 1
        self.logger.info(f"Requested date range: {total_days} days ({start_date} to {end_date})")
        
        # REMOVED: Artificial memory limit - now supports multi-year training
        # Old code limited to 30 days, new system uses chunked loading for any range
        
        # Ensure dates are historical (keep some buffer for data availability)
        max_end_date = current_date - pd.Timedelta(days=2)
        if end_dt >= max_end_date:
            end_dt = max_end_date
            end_date = end_dt.strftime('%Y-%m-%d')
            self.logger.warning(f"Adjusted end date to be historical: {end_date}")
        
        if start_dt >= end_dt:
            end_dt = max_end_date
            start_dt = end_dt - pd.Timedelta(days=7)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            self.logger.warning(f"Adjusted date range to: {start_date} to {end_date}")
          # Log final range
        final_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        self.logger.info(f"Final date range: {final_days} days - ready for chunked loading if needed")
        
        return start_date, end_date

    def _load_data_with_memory_checks(self, downloader, symbol: str, interval: str, 
                                    start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data with intelligent chunking for large date ranges"""
        try:
            # Calculate date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            total_days = (end_dt - start_dt).days + 1
            
            # Configure chunking parameters
            max_chunk_days = self.config.get('max_chunk_days', 90)  # 90 days per chunk (reasonable memory usage)
            max_total_rows = self.config.get('max_total_rows', 200000)  # Maximum total rows to prevent excessive memory
            
            self.logger.info(f"Loading {total_days} days of data with chunking if needed")
            
            # If date range is small, load normally
            if total_days <= max_chunk_days:
                return self._load_single_chunk(downloader, symbol, interval, start_date, end_date, max_total_rows)
            
            # For large date ranges, use chunked loading
            return self._load_chunked_data(downloader, symbol, interval, start_date, end_date, 
                                         max_chunk_days, max_total_rows)
            
        except Exception as e:
            self.logger.error(f"Error in chunked data loading: {e}")
            return None

    def _load_single_chunk(self, downloader, symbol: str, interval: str, 
                          start_date: str, end_date: str, max_rows: int) -> Optional[pd.DataFrame]:
        """Load a single chunk of data"""
        try:
            # Try to load from saved files first
            df = self._load_from_daily_files(downloader, symbol, interval, start_date, end_date)
            
            if df is None or len(df) == 0:
                self.logger.error("Single chunk data loading returned no data")
                return None
            
            # Fix timestamp issues
            df = self._fix_timestamp_issues(df)
            
            # Apply memory limits if needed
            if len(df) > max_rows:
                self.logger.warning(f"Chunk has {len(df)} rows, sampling to {max_rows} for memory efficiency")
                df = df.sample(n=max_rows, random_state=42).sort_index()
            
            self.logger.info(f"Loaded single chunk: {len(df)} records from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading single chunk: {e}")
            return None

    def _load_chunked_data(self, downloader, symbol: str, interval: str, 
                          start_date: str, end_date: str, chunk_days: int, max_total_rows: int) -> Optional[pd.DataFrame]:
        """Load large date ranges using intelligent chunking"""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            total_days = (end_dt - start_dt).days + 1
            
            self.logger.info(f"🔄 Starting chunked loading for {total_days} days in {chunk_days}-day chunks")
            
            chunks = []
            current_start = start_dt
            chunk_count = 0
            
            while current_start < end_dt:
                chunk_count += 1
                current_end = min(current_start + pd.Timedelta(days=chunk_days), end_dt)
                
                chunk_start_str = current_start.strftime('%Y-%m-%d')
                chunk_end_str = current_end.strftime('%Y-%m-%d')
                
                self.logger.info(f"📦 Loading chunk {chunk_count}: {chunk_start_str} to {chunk_end_str}")
                
                # Load chunk
                chunk_df = self._load_single_chunk(downloader, symbol, interval, 
                                                 chunk_start_str, chunk_end_str, max_total_rows // 4)
                
                if chunk_df is not None and len(chunk_df) > 0:
                    chunks.append(chunk_df)
                    self.logger.info(f"✅ Chunk {chunk_count}: {len(chunk_df)} records")
                else:
                    self.logger.warning(f"⚠️ Chunk {chunk_count}: No data loaded")
                
                current_start = current_end + pd.Timedelta(days=1)
            
            if not chunks:
                self.logger.error("No chunks loaded successfully")
                return None
            
            # Combine all chunks
            self.logger.info(f"🔗 Combining {len(chunks)} chunks...")
            combined_df = pd.concat(chunks, ignore_index=False)
            combined_df = combined_df.sort_index().drop_duplicates()
            
            # Apply final memory limits
            if len(combined_df) > max_total_rows:
                self.logger.warning(f"Combined data has {len(combined_df)} rows, sampling to {max_total_rows}")
                combined_df = combined_df.sample(n=max_total_rows, random_state=42).sort_index()
            
            self.logger.info(f"🎉 Chunked loading complete: {len(combined_df)} total records")
            self.logger.info(f"📅 Final range: {combined_df.index.min()} to {combined_df.index.max()}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error in chunked data loading: {e}")
            return None

    def _load_from_daily_files(self, downloader, symbol: str, interval: str, 
                              start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from daily files directly"""
        try:
            # Find data directory
            data_dir = self._find_data_directory(downloader)
            if data_dir is None:
                return None
            
            # Get CSV files
            csv_files = list(data_dir.glob("*.csv"))
            if not csv_files:
                self.logger.warning(f"No CSV files found in {data_dir}")
                return None
            
            # Filter relevant files
            relevant_files = self._filter_relevant_files(csv_files, symbol, interval, start_date, end_date)
            
            if not relevant_files:
                self.logger.warning("No relevant files found")
                return None
            
            # Load and combine files
            return self._load_and_combine_files(relevant_files, start_date, interval)
            
        except Exception as e:
            self.logger.error(f"Failed to load from daily files: {e}")
            return None

    def _find_data_directory(self, downloader) -> Optional[Path]:
        """Find directory containing CSV data files"""
        possible_dirs = [
            Path(getattr(downloader, 'data_dir', 'data/coinapi')),
            Path("data/coinapi"),
            Path("data"),
            Path("data/raw"),
            Path("."),
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists() and list(dir_path.glob("*.csv")):
                self.logger.info(f"Found data directory: {dir_path}")
                return dir_path
        
        # Search entire project as fallback
        cwd = Path.cwd()
        all_csv_files = list(cwd.rglob("*.csv"))
        if all_csv_files:
            data_dir = all_csv_files[0].parent
            self.logger.info(f"Using directory of first CSV file found: {data_dir}")
            return data_dir
        
        return None

    def _filter_relevant_files(self, csv_files: list, symbol: str, interval: str, 
                              start_date: str, end_date: str) -> list:
        """Filter CSV files by symbol, interval, and date range"""
        # Filter by symbol/interval
        clean_symbol = symbol.replace('_', '').replace('-', '').upper()
        relevant_files = []
        
        for csv_file in csv_files:
            filename_upper = csv_file.name.upper()
            if (clean_symbol in filename_upper or 
                symbol.upper() in filename_upper or 
                interval.upper() in filename_upper):
                relevant_files.append(csv_file)
        
        if not relevant_files:
            self.logger.warning(f"No files matching {symbol}/{interval}, using all CSV files")
            relevant_files = csv_files[:50]  # Limit to prevent memory issues
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        date_filtered_files = []
        for file_path in relevant_files:
            file_date = self._extract_date_from_filename(file_path.stem)
            if file_date and start_dt <= file_date <= end_dt:
                date_filtered_files.append(file_path)
            elif file_date is None:
                date_filtered_files.append(file_path)  # Include if can't parse date
        
        return date_filtered_files[:50]  # Limit files

    def _extract_date_from_filename(self, filename: str) -> Optional[pd.Timestamp]:
        """Extract date from filename using regex patterns"""
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{8})',              # YYYYMMDD
            r'(\d{4}_\d{2}_\d{2})'   # YYYY_MM_DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1).replace('_', '-')
                if len(date_str) == 8:  # YYYYMMDD format
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                try:
                    return pd.to_datetime(date_str)
                except:
                    continue
        return None

    def _load_and_combine_files(self, files: list, start_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Load and combine multiple CSV files"""
        dfs = []
        for file_path in sorted(files):
            try:
                df_day = pd.read_csv(file_path)
                if len(df_day) > 0:
                    dfs.append(df_day)
                    self.logger.debug(f"Loaded {len(df_day)} records from {file_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path.name}: {e}")
        
        if not dfs:
            return None
        
        # Combine dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Set timestamp as index
        timestamp_cols = ['timestamp', 'time', 'datetime', 'date', 'time_period_start', 'time_period_end']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in combined_df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            combined_df[timestamp_col] = pd.to_datetime(combined_df[timestamp_col])
            combined_df.set_index(timestamp_col, inplace=True)
        else:
            # Create artificial timestamp index
            start_time = pd.to_datetime(start_date)
            freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1H', '4h': '4H', '1d': '1D'}
            freq = freq_map.get(interval, '15min')
            time_index = pd.date_range(start=start_time, periods=len(combined_df), freq=freq)
            combined_df.index = time_index
        
        # Remove duplicates and sort
        if hasattr(combined_df.index, 'duplicated') and combined_df.index.duplicated().any():
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        combined_df.sort_index(inplace=True)
        
        self.logger.info(f"Successfully combined {len(combined_df)} records from {len(dfs)} files")
        return combined_df

    def _fix_timestamp_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix timestamp and duplicate issues in the dataframe"""
        try:
            # Check if index is datetime, if not convert it
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.index = pd.to_datetime(df['timestamp'])
                    df = df.drop('timestamp', axis=1)
                elif 'time' in df.columns:
                    df.index = pd.to_datetime(df['time'])
                    df = df.drop('time', axis=1)
                else:
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        df.reset_index(drop=True, inplace=True)
                        return df
            
            # Remove duplicates and sort
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            # Validate timestamp range
            if isinstance(df.index, pd.DatetimeIndex):
                min_date = df.index.min()
                if min_date.year < 2010:  # Fix epoch issues
                    start_dt = pd.to_datetime(self.config.get('start_date', '2025-01-01'))
                    freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1H', '4h': '4H', '1d': '1D'}
                    freq = freq_map.get(self.config.get('interval', '15m'), '15min')
                    new_index = pd.date_range(start=start_dt, periods=len(df), freq=freq)
                    df.index = new_index
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fixing timestamp issues: {e}")
            return df

    def process_data(self, raw_data: pd.DataFrame) -> tuple:
        """
        Process raw data and create train/test splits
        
        Args:
            raw_data: Raw OHLCV data
            
        Returns:
            Tuple of (train_data, test_data)
        """
        self.logger.info("Starting data processing...")
        
        # Fix timestamp issues
        raw_data = self._fix_timestamp_issues(raw_data)
        
        # Validate raw data
        self.logger.info(f"Raw data shape: {raw_data.shape}")
        self.logger.info(f"Raw data columns: {list(raw_data.columns)}")
        
        # Clean data
        raw_data = self._clean_data(raw_data)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            lookback_window=self.config.get('lookback_window', 20)
        )
        
        # Process data
        try:
            train_data, test_data = self.data_processor.prepare_data(
                raw_data,
                funding_rates=None,
                train_ratio=self.config.get('train_ratio', 0.8)
            )
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
        
        # Validate processed data
        self._validate_processed_data(train_data, test_data)
        
        # Save processed data
        self._save_processed_data(train_data, test_data)
        
        self.logger.info(f"Data processing completed. Train: {train_data.shape}, Test: {test_data.shape}")
        return train_data, test_data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling NaN and infinite values"""
        # Check for NaN values
        nan_counts = data.isnull().sum()
        if nan_counts.any():
            self.logger.warning(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        if inf_counts.any():
            self.logger.warning(f"Infinite values found: {inf_counts[inf_counts > 0].to_dict()}")
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data

    def _validate_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Validate processed data for training"""
        for name, data in [("Training", train_data), ("Test", test_data)]:
            # Check for NaN values
            nan_counts = data.isnull().sum()
            if nan_counts.any():
                raise ValueError(f"{name} data contains NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            
            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            inf_mask = np.isinf(data[numeric_cols]).any(axis=1)
            if inf_mask.any():
                raise ValueError(f"{name} data contains infinite values in {inf_mask.sum()} rows")

    def _save_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Save processed data and scalers"""
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_data.to_csv(data_dir / f"train_data_{timestamp}.csv")
        test_data.to_csv(data_dir / f"test_data_{timestamp}.csv")
        
        if self.data_processor:
            scaler_path = data_dir / f"scalers_{timestamp}.pkl"
            self.data_processor.save_scalers(str(scaler_path))
