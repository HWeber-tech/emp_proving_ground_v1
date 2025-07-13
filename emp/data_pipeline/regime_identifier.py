"""
MarketRegimeIdentifier: Scans historical data, classifies regimes, and outputs regimes.json.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MarketRegimeIdentifier:
    """
    Scans the full historical dataset, classifies regimes, and outputs regimes.json.
    """
    
    def __init__(self, data_dir: str, symbol: str):
        self.data_dir = data_dir
        self.symbol = symbol
        
        # Regime classification parameters
        self.rolling_window = 60  # 60-day rolling window
        self.min_regime_days = 30  # Minimum days for a regime period
        
        # Regime thresholds
        self.regime_thresholds = {
            'volatility': {
                'low': 0.10,    # 10% annualized volatility
                'high': 0.25    # 25% annualized volatility
            },
            'hurst': {
                'trending': 0.55,    # > 0.55 indicates trending
                'ranging': 0.45      # < 0.45 indicates mean-reversion
            },
            'kurtosis': {
                'normal': 3.0,       # Normal distribution kurtosis
                'crisis': 5.0        # Crisis level kurtosis
            }
        }
    
    def identify_regimes(self, start_year: int, end_year: int, output_path: str = "regimes.json") -> Dict:
        """
        Identify market regimes and output regimes.json.
        
        Args:
            start_year: Start year for analysis
            end_year: End year for analysis
            output_path: Path to output regimes.json file
            
        Returns:
            Dictionary with regime information
        """
        logger.info(f"Starting regime identification for {self.symbol} from {start_year} to {end_year}")
        
        try:
            # Load historical data
            df = self._load_historical_data(start_year, end_year)
            if df.empty:
                raise ValueError(f"No data found for {self.symbol} in specified period")
            
            # Resample to daily OHLCV
            daily_df = self._resample_to_daily(df)
            
            # Calculate regime indicators
            regime_df = self._calculate_regime_indicators(daily_df)
            
            # Classify regimes
            classified_df = self._classify_regimes(regime_df)
            
            # Identify regime periods
            regime_periods = self._identify_regime_periods(classified_df)
            
            # Create regimes.json output
            regimes_data = self._create_regimes_output(regime_periods, start_year, end_year)
            
            # Save to file
            self._save_regimes_json(regimes_data, output_path)
            
            logger.info(f"Regime identification complete. Found {len(regime_periods)} regime periods")
            return regimes_data
            
        except Exception as e:
            logger.error(f"Regime identification failed: {e}")
            raise
    
    def _load_historical_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Load historical tick data for the specified period.
        
        Args:
            start_year: Start year
            end_year: End year
            
        Returns:
            DataFrame with tick data
        """
        from .storage import TickDataStorage
        
        storage = TickDataStorage(self.data_dir)
        
        # Check if data is available
        data_range = storage.get_data_range(self.symbol)
        if data_range is None:
            raise ValueError(f"No data available for {self.symbol}")
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31, 23, 59, 59)
        
        # Load tick data
        df = storage.load_tick_data(self.symbol, start_date, end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol} in {start_year}-{end_year}")
        
        logger.info(f"Loaded {len(df)} ticks from {start_date} to {end_date}")
        return df
    
    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample tick data to daily OHLCV bars.
        
        Args:
            df: Tick data DataFrame
            
        Returns:
            Daily OHLCV DataFrame
        """
        # Ensure timestamp is the index
        df = df.set_index('timestamp').sort_index()
        
        # Resample to daily OHLCV
        daily_df = df.resample('D').agg({
            'bid': 'ohlc',
            'ask': 'ohlc',
            'bid_volume': 'sum',
            'ask_volume': 'sum'
        })
        
        # Flatten column names
        daily_df.columns = [
            'bid_open', 'bid_high', 'bid_low', 'bid_close',
            'ask_open', 'ask_high', 'ask_low', 'ask_close',
            'bid_volume', 'ask_volume'
        ]
        
        # Calculate mid-price OHLCV
        daily_df['open'] = (daily_df['bid_open'] + daily_df['ask_open']) / 2
        daily_df['high'] = (daily_df['bid_high'] + daily_df['ask_high']) / 2
        daily_df['low'] = (daily_df['bid_low'] + daily_df['ask_low']) / 2
        daily_df['close'] = (daily_df['bid_close'] + daily_df['ask_close']) / 2
        daily_df['volume'] = daily_df['bid_volume'] + daily_df['ask_volume']
        
        # Calculate daily returns
        daily_df['returns'] = daily_df['close'].pct_change()
        
        # Remove rows with missing data
        daily_df = daily_df.dropna()
        
        logger.info(f"Resampled to {len(daily_df)} daily bars")
        return daily_df
    
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime indicators on rolling windows.
        
        Args:
            df: Daily OHLCV DataFrame
            
        Returns:
            DataFrame with regime indicators
        """
        result_df = df.copy()
        
        # Calculate realized volatility (annualized)
        result_df['volatility'] = (
            result_df['returns'].rolling(window=self.rolling_window)
            .std() * np.sqrt(252)
        )
        
        # Calculate Hurst exponent
        result_df['hurst'] = result_df['close'].rolling(
            window=self.rolling_window
        ).apply(self._calculate_hurst_exponent)
        
        # Calculate kurtosis
        result_df['kurtosis'] = result_df['returns'].rolling(
            window=self.rolling_window
        ).kurt()
        
        # Calculate additional indicators
        result_df['atr'] = self._calculate_atr(result_df, window=self.rolling_window)
        result_df['trend_strength'] = self._calculate_trend_strength(result_df, window=self.rolling_window)
        
        # Remove NaN values from rolling calculations
        result_df = result_df.dropna()
        
        logger.info(f"Calculated regime indicators for {len(result_df)} days")
        return result_df
    
    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """
        Calculate Hurst exponent for a price series.
        
        Args:
            prices: Price series
            
        Returns:
            Hurst exponent value
        """
        if len(prices) < 10:
            return 0.5  # Neutral value for insufficient data
        
        try:
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(log_returns) < 10:
                return 0.5
            
            # Calculate cumulative sum
            cumsum = np.cumsum(log_returns)
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(len(log_returns) // 2, 20))
            rs_values = []
            
            for lag in lags:
                # Split data into chunks
                chunks = len(log_returns) // lag
                if chunks < 2:
                    continue
                
                rs_chunk_values = []
                for i in range(chunks):
                    chunk = cumsum[i * lag:(i + 1) * lag]
                    if len(chunk) < 2:
                        continue
                    
                    # Calculate R (range)
                    R = np.max(chunk) - np.min(chunk)
                    
                    # Calculate S (standard deviation)
                    S = np.std(chunk)
                    
                    if S > 0:
                        rs_chunk_values.append(R / S)
                
                if rs_chunk_values:
                    rs_values.append(np.mean(rs_chunk_values))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Fit linear regression to log(R/S) vs log(lag)
            lags_used = lags[:len(rs_values)]
            log_lags = np.log(lags_used)
            log_rs = np.log(rs_values)
            
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            
            return slope
            
        except Exception as e:
            logger.warning(f"Hurst calculation failed: {e}")
            return 0.5
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: OHLCV DataFrame
            window: Rolling window size
            
        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_trend_strength(self, df: pd.DataFrame, window: int = 60) -> pd.Series:
        """
        Calculate trend strength using linear regression R-squared.
        
        Args:
            df: OHLCV DataFrame
            window: Rolling window size
            
        Returns:
            Trend strength series
        """
        def r_squared(x):
            if len(x) < 2:
                return 0.0
            try:
                y = np.arange(len(x))
                slope, intercept, r_value, p_value, std_err = stats.linregress(y, x)
                return r_value ** 2
            except:
                return 0.0
        
        trend_strength = df['close'].rolling(window=window).apply(r_squared)
        return trend_strength
    
    def _classify_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each day into a market regime.
        
        Args:
            df: DataFrame with regime indicators
            
        Returns:
            DataFrame with regime classifications
        """
        result_df = df.copy()
        
        # Initialize regime columns
        result_df['volatility_regime'] = 'normal'
        result_df['trend_regime'] = 'neutral'
        result_df['crisis_regime'] = 'normal'
        result_df['composite_regime'] = 'ranging'
        
        # Classify volatility regime
        low_vol = result_df['volatility'] < self.regime_thresholds['volatility']['low']
        high_vol = result_df['volatility'] > self.regime_thresholds['volatility']['high']
        
        result_df.loc[low_vol, 'volatility_regime'] = 'low'
        result_df.loc[high_vol, 'volatility_regime'] = 'high'
        
        # Classify trend regime
        trending = result_df['hurst'] > self.regime_thresholds['hurst']['trending']
        ranging = result_df['hurst'] < self.regime_thresholds['hurst']['ranging']
        
        result_df.loc[trending, 'trend_regime'] = 'trending'
        result_df.loc[ranging, 'trend_regime'] = 'ranging'
        
        # Classify crisis regime
        crisis = result_df['kurtosis'] > self.regime_thresholds['kurtosis']['crisis']
        result_df.loc[crisis, 'crisis_regime'] = 'crisis'
        
        # Create composite regime classification
        for idx in result_df.index:
            vol_regime = result_df.loc[idx, 'volatility_regime']
            trend_regime = result_df.loc[idx, 'trend_regime']
            crisis_regime = result_df.loc[idx, 'crisis_regime']
            
            # Determine composite regime
            if crisis_regime == 'crisis':
                composite = 'volatile'
            elif vol_regime == 'high':
                if trend_regime == 'trending':
                    composite = 'trending'
                else:
                    composite = 'volatile'
            elif trend_regime == 'trending':
                composite = 'trending'
            else:
                composite = 'ranging'
            
            result_df.loc[idx, 'composite_regime'] = composite
        
        logger.info(f"Classified {len(result_df)} days into regimes")
        return result_df
    
    def _identify_regime_periods(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify contiguous regime periods.
        
        Args:
            df: DataFrame with regime classifications
            
        Returns:
            List of regime period dictionaries
        """
        regime_periods = []
        
        # Group by regime changes
        df['regime_change'] = df['composite_regime'] != df['composite_regime'].shift(1)
        df['regime_group'] = df['regime_change'].cumsum()
        
        for group_id, group_df in df.groupby('regime_group'):
            if len(group_df) < self.min_regime_days:
                continue  # Skip periods that are too short
            
            regime_type = group_df['composite_regime'].iloc[0]
            start_date = group_df.index[0]
            end_date = group_df.index[-1]
            
            # Calculate regime statistics
            avg_volatility = group_df['volatility'].mean()
            avg_hurst = group_df['hurst'].mean()
            avg_kurtosis = group_df['kurtosis'].mean()
            
            period = {
                'regime_type': regime_type,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration_days': len(group_df),
                'avg_volatility': float(avg_volatility),
                'avg_hurst': float(avg_hurst),
                'avg_kurtosis': float(avg_kurtosis),
                'start_timestamp': start_date.isoformat(),
                'end_timestamp': end_date.isoformat()
            }
            
            regime_periods.append(period)
        
        # Sort by start date
        regime_periods.sort(key=lambda x: x['start_date'])
        
        logger.info(f"Identified {len(regime_periods)} regime periods")
        return regime_periods
    
    def _create_regimes_output(self, regime_periods: List[Dict], start_year: int, end_year: int) -> Dict:
        """
        Create the final regimes.json output structure.
        
        Args:
            regime_periods: List of regime periods
            start_year: Analysis start year
            end_year: Analysis end year
            
        Returns:
            Dictionary for regimes.json
        """
        # Group periods by regime type
        regime_groups = {}
        for period in regime_periods:
            regime_type = period['regime_type']
            if regime_type not in regime_groups:
                regime_groups[regime_type] = []
            regime_groups[regime_type].append(period)
        
        # Find the longest period for each regime type
        triathlon_datasets = {}
        for regime_type, periods in regime_groups.items():
            if periods:
                # Sort by duration and take the longest
                longest_period = max(periods, key=lambda x: x['duration_days'])
                triathlon_datasets[regime_type] = longest_period
        
        output = {
            'metadata': {
                'symbol': self.symbol,
                'analysis_start_year': start_year,
                'analysis_end_year': end_year,
                'analysis_date': datetime.now().isoformat(),
                'rolling_window_days': self.rolling_window,
                'min_regime_days': self.min_regime_days,
                'regime_thresholds': self.regime_thresholds
            },
            'regime_periods': regime_periods,
            'triathlon_datasets': triathlon_datasets,
            'regime_statistics': self._calculate_regime_statistics(regime_periods)
        }
        
        return output
    
    def _calculate_regime_statistics(self, regime_periods: List[Dict]) -> Dict:
        """
        Calculate statistics for each regime type.
        
        Args:
            regime_periods: List of regime periods
            
        Returns:
            Dictionary with regime statistics
        """
        stats = {}
        
        for period in regime_periods:
            regime_type = period['regime_type']
            if regime_type not in stats:
                stats[regime_type] = {
                    'count': 0,
                    'total_days': 0,
                    'avg_duration': 0,
                    'avg_volatility': 0,
                    'avg_hurst': 0,
                    'avg_kurtosis': 0
                }
            
            stats[regime_type]['count'] += 1
            stats[regime_type]['total_days'] += period['duration_days']
            stats[regime_type]['avg_volatility'] += period['avg_volatility']
            stats[regime_type]['avg_hurst'] += period['avg_hurst']
            stats[regime_type]['avg_kurtosis'] += period['avg_kurtosis']
        
        # Calculate averages
        for regime_type in stats:
            count = stats[regime_type]['count']
            if count > 0:
                stats[regime_type]['avg_duration'] = stats[regime_type]['total_days'] / count
                stats[regime_type]['avg_volatility'] /= count
                stats[regime_type]['avg_hurst'] /= count
                stats[regime_type]['avg_kurtosis'] /= count
        
        return stats
    
    def _save_regimes_json(self, regimes_data: Dict, output_path: str):
        """
        Save regimes data to JSON file.
        
        Args:
            regimes_data: Regimes data dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(regimes_data, f, indent=2, default=str)
            
            logger.info(f"Saved regimes data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save regimes.json: {e}")
            raise 