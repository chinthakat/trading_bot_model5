#!/usr/bin/env python3
"""
Trade Tracer Module
Creates detailed trace files for each completed trade with comprehensive trade analysis and reward breakdown
"""

import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

class TradeTracer:
    """
    Creates detailed trace files for each completed trade
    Each trade close generates a separate JSON file with:
    - Trade execution details (amounts, prices, fees)
    - Complete reward system breakdown
    - Market context and timing analysis    - Performance metrics and ratios
    """
    
    def __init__(self, trace_dir: str = "logs/trade_traces", session_name: Optional[str] = None,
                 enable_tracing: bool = True, episode_num: int = 0, batch_num: int = 0):
        """
        Initialize trade tracer
        
        Args:
            trace_dir: Directory to store trade trace files
            session_name: Optional session name for organization
            enable_tracing: Whether to enable trace file creation
            episode_num: Episode number for logging
            batch_num: Batch number for logging
        """
        self.enable_tracing = enable_tracing
        self.episode_num = episode_num
        self.batch_num = batch_num
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        if self.enable_tracing:
            self.trace_dir = Path(trace_dir)
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a consistent filename for all trades
            trace_filename = "trade_traces.jsonl"
            self.trace_file = self.trace_dir / trace_filename
            
            self.session_name = session_name or f"ep{episode_num:03d}_batch{batch_num:03d}"

            # Setup a dedicated text log file for the tracer
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            tracer_log_file = log_dir / "trade_tracer.log"

            # Avoid adding handlers repeatedly
            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(tracer_log_file) for h in self.logger.handlers):
                file_handler = logging.FileHandler(tracer_log_file, mode='a')
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            self.logger.info(f"Initialized TradeTracer. JSONL traces in: {self.trace_file}. Text logs in: {tracer_log_file}")
        else:
            self.trace_dir = None
            self.session_name = session_name or "disabled_session"
            self.trace_file = None
            self.logger.info("TradeTracer initialized with tracing disabled")
        
        # Trade tracking
        self.trade_counter = 0
        self.open_positions = {}  # Track open positions for trace completion

    def log_trade_open(self, 
                      trade_id: str,
                      action: str,
                      symbol: str,
                      entry_price: float,
                      position_size: float,
                      leverage: float,
                      commission: float,
                      balance_before: float,
                      balance_after: float,
                      market_timestamp: Optional[pd.Timestamp] = None,
                      market_data: Optional[Dict] = None,
                      observation_space: Optional[Dict] = None,
                      confidence: float = 1.0,
                      trade_reason: str = "",
                      episode_step: int = 0,
                      reward_info: Optional[Dict] = None) -> None:
        """
        Log trade opening with initial details
        
        Args:
            trade_id: Unique identifier for this trade
            action: BUY or SELL (opening action)
            symbol: Trading symbol
            entry_price: Trade entry price
            position_size: Position size (positive for long, negative for short)
            leverage: Leverage used
            commission: Commission paid for entry
            balance_before: Account balance before trade
            balance_after: Account balance after trade
            market_timestamp: Market timestamp for this trade
            market_data: Market indicators at trade time
            observation_space: The agent's observation space at the time of the trade
            confidence: AI confidence in this trade
            trade_reason: Reason for opening trade
            episode_step: RL episode step
            reward_info: Reward breakdown for opening action        """
        try:
            if not self.enable_tracing:
                return  # Skip tracing if disabled
                
            # Clean and validate inputs
            entry_price = float(np.nan_to_num(entry_price, nan=0.0))
            position_size = float(np.nan_to_num(position_size, nan=0.0))
            leverage = float(np.nan_to_num(leverage, nan=1.0))
            commission = float(np.nan_to_num(commission, nan=0.0))
            balance_before = float(np.nan_to_num(balance_before, nan=0.0))
            balance_after = float(np.nan_to_num(balance_after, nan=0.0))
            confidence = float(np.nan_to_num(confidence, nan=1.0))
            
            # Get timestamp
            if market_timestamp is None:
                timestamp = datetime.now()
            else:
                if isinstance(market_timestamp, pd.Timestamp):
                    timestamp = market_timestamp.to_pydatetime()
                else:
                    timestamp = pd.to_datetime(market_timestamp).to_pydatetime()
            
            # Determine position side
            side = "LONG" if position_size > 0 else "SHORT"
            
            # Create trade record
            trade_record = {
                'trade_id': trade_id,
                'status': 'OPEN',
                'symbol': symbol,
                'side': side,
                
                # Entry details
                'entry_action': action,
                'entry_price': round(entry_price, 8),
                'entry_timestamp': timestamp.isoformat(),
                'entry_datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'position_size': round(abs(position_size), 8),
                'leverage': round(leverage, 2),
                'entry_commission': round(commission, 6),
                
                # Account state at entry
                'balance_before_entry': round(balance_before, 2),
                'balance_after_entry': round(balance_after, 2),
                'position_value_at_entry': round(abs(position_size) * entry_price, 2),
                
                # Market context at entry
                'market_data_at_entry': market_data or {},
                'observation_at_entry': observation_space or {},
                
                # Trade metadata
                'confidence': round(confidence, 2),
                'trade_reason': trade_reason,
                'episode_step_at_entry': episode_step,
                
                # Reward information at entry
                'entry_reward_breakdown': reward_info or {},
                
                # Exit details (to be filled later)
                'exit_action': None,
                'exit_price': None,
                'exit_timestamp': None,
                'exit_datetime': None,
                'exit_commission': None,
                'exit_reward_breakdown': {},
                
                # Trade results (to be calculated at exit)
                'duration_seconds': None,
                'duration_steps': None,
                'gross_pnl': None,
                'net_pnl': None,
                'total_commission': round(commission, 6),
                'pnl_percentage': None,
                'win_loss': None,
                
                # Final account state (to be filled at exit)
                'balance_before_exit': None,
                'balance_after_exit': None,
                'equity_change': None,
                
                # Performance metrics (to be calculated)
                'risk_reward_ratio': None,
                'return_on_risk': None,
                'holding_period_return': None
            }
              # Store open position
            self.open_positions[trade_id] = trade_record
            
            # Immediately write the trade opening to trace file
            self._write_trace_event("TRADE_OPENED", trade_record)
            
            self.logger.info(f"Trade opened: {trade_id} - {side} {abs(position_size):.6f} {symbol} @ {entry_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade open: {e}")

    def log_trade_close(self,
                       trade_id: str,
                       exit_action: str,
                       exit_price: float,
                       commission: float,
                       balance_before: float,
                       balance_after: float,
                       market_timestamp: Optional[pd.Timestamp] = None,
                       market_data: Optional[Dict] = None,
                       observation_space: Optional[Dict] = None,
                       episode_step: int = 0,
                       reward_info: Optional[Dict] = None) -> None:
        """
        Log trade closing and create detailed trace file
        
        Args:
            trade_id: Unique identifier for this trade
            exit_action: CLOSE action
            exit_price: Trade exit price
            commission: Commission paid for exit
            balance_before: Account balance before close
            balance_after: Account balance after close
            market_timestamp: Market timestamp for close
            market_data: Market indicators at close time
            observation_space: The agent's observation space at the time of the trade
            episode_step: RL episode step at close
            reward_info: Reward breakdown for closing action        """
        try:
            if not self.enable_tracing:
                return  # Skip tracing if disabled
                
            if trade_id not in self.open_positions:
                self.logger.warning(f"Trade {trade_id} not found in open positions")
                return
            
            # Get open trade record
            trade_record = self.open_positions[trade_id].copy()
            
            # Clean and validate inputs
            exit_price = float(np.nan_to_num(exit_price, nan=0.0))
            commission = float(np.nan_to_num(commission, nan=0.0))
            balance_before = float(np.nan_to_num(balance_before, nan=0.0))
            balance_after = float(np.nan_to_num(balance_after, nan=0.0))
            
            # Get exit timestamp
            if market_timestamp is None:
                exit_timestamp = datetime.now()
            else:
                if isinstance(market_timestamp, pd.Timestamp):
                    exit_timestamp = market_timestamp.to_pydatetime()
                else:
                    exit_timestamp = pd.to_datetime(market_timestamp).to_pydatetime()
            
            # Calculate trade metrics
            entry_price = trade_record['entry_price']
            position_size = trade_record['position_size']
            side = trade_record['side']
            entry_timestamp = pd.to_datetime(trade_record['entry_timestamp']).to_pydatetime()
            
            # Calculate P&L
            if side == "LONG":
                gross_pnl = position_size * (exit_price - entry_price)
            else:  # SHORT
                gross_pnl = position_size * (entry_price - exit_price)
            
            total_commission = trade_record['entry_commission'] + commission
            net_pnl = gross_pnl - total_commission
              # Calculate percentages and ratios
            position_value = position_size * entry_price
            initial_balance = trade_record.get('balance_before_entry', 10000.0)
            pnl_percentage = (net_pnl / initial_balance) if initial_balance > 0 else 0.0  # As decimal (e.g., -0.0536 for -5.36%)
            
            # Calculate duration
            duration_seconds = (exit_timestamp - entry_timestamp).total_seconds()
            duration_steps = episode_step - trade_record['episode_step_at_entry']
            
            # Determine win/loss
            win_loss = "WIN" if net_pnl > 0 else ("LOSS" if net_pnl < 0 else "BREAKEVEN")
            
            # Calculate performance metrics
            risk_amount = position_value * 0.02  # Assume 2% risk
            risk_reward_ratio = abs(net_pnl / risk_amount) if risk_amount > 0 else 0.0
            
            equity_change = balance_after - trade_record['balance_before_entry']
            holding_period_return = (equity_change / trade_record['balance_before_entry'] * 100) if trade_record['balance_before_entry'] > 0 else 0.0
            
            # Update trade record with exit details
            trade_record.update({
                'status': 'CLOSED',
                
                # Exit details
                'exit_action': exit_action,
                'exit_price': round(exit_price, 8),
                'exit_timestamp': exit_timestamp.isoformat(),
                'exit_datetime': exit_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_commission': round(commission, 6),
                'exit_reward_breakdown': reward_info or {},
                
                # Market context at exit
                'market_data_at_exit': market_data or {},
                'observation_at_exit': observation_space or {},
                'episode_step_at_exit': episode_step,
                
                # Trade results
                'duration_seconds': int(duration_seconds),
                'duration_steps': duration_steps,
                'gross_pnl': round(gross_pnl, 6),
                'net_pnl': round(net_pnl, 6),
                'total_commission': round(total_commission, 6),
                'pnl_percentage': round(pnl_percentage, 4),
                'win_loss': win_loss,
                
                # Final account state
                'balance_before_exit': round(balance_before, 2),
                'balance_after_exit': round(balance_after, 2),
                'equity_change': round(equity_change, 2),
                
                # Performance metrics
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'return_on_risk': round(net_pnl / risk_amount * 100, 2) if risk_amount > 0 else 0.0,
                'holding_period_return': round(holding_period_return, 4)
            })
            
            # Generate comprehensive trade analysis
            trade_analysis = self._generate_trade_analysis(trade_record)
            trade_record['trade_analysis'] = trade_analysis
            
            # Save detailed trace file
            self._write_trace_event("TRADE_CLOSED", trade_record)
            
            # Remove from open positions
            del self.open_positions[trade_id]
            
            self.logger.info(f"Trade closed: {trade_id} - {win_loss} P&L: ${net_pnl:.2f} ({pnl_percentage:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error logging trade close: {e}")

    def _generate_trade_analysis(self, trade_record: Dict) -> Dict:
        """
        Generate comprehensive trade analysis
        
        Args:
            trade_record: Complete trade record
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {}
        
        try:
            # Basic trade classification
            analysis['trade_type'] = f"{trade_record['side']} position"
            analysis['outcome'] = trade_record['win_loss']
            analysis['profitability'] = "Profitable" if trade_record['net_pnl'] > 0 else "Unprofitable"
            
            # Performance metrics
            analysis['performance_grade'] = self._grade_trade_performance(trade_record)
            analysis['risk_assessment'] = self._assess_trade_risk(trade_record)
            
            # Timing analysis
            analysis['timing_analysis'] = self._analyze_trade_timing(trade_record)
            
            # Reward system analysis
            analysis['reward_analysis'] = self._analyze_reward_breakdown(trade_record)
            
            # Market context analysis
            analysis['market_analysis'] = self._analyze_market_context(trade_record)
            
            # Improvement suggestions
            analysis['suggestions'] = self._generate_improvement_suggestions(trade_record)
            
        except Exception as e:
            self.logger.warning(f"Error generating trade analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis

    def _grade_trade_performance(self, trade_record: Dict) -> str:
        """Grade trade performance from A+ to F"""
        net_pnl = trade_record['net_pnl']
        pnl_pct = trade_record['pnl_percentage']
        risk_reward = trade_record['risk_reward_ratio']
        
        if net_pnl > 0 and pnl_pct > 5 and risk_reward > 2:
            return "A+"
        elif net_pnl > 0 and pnl_pct > 2 and risk_reward > 1.5:
            return "A"
        elif net_pnl > 0 and pnl_pct > 1:
            return "B+"
        elif net_pnl > 0:
            return "B"
        elif pnl_pct > -1:
            return "C"
        elif pnl_pct > -2:
            return "D"
        else:
            return "F"

    def _assess_trade_risk(self, trade_record: Dict) -> Dict:
        """Assess risk characteristics of the trade"""
        leverage = trade_record['leverage']
        position_value = trade_record['position_value_at_entry']
        balance = trade_record['balance_before_entry']
        
        risk_pct = (position_value / balance * 100) if balance > 0 else 100
        
        risk_level = "LOW"
        if risk_pct > 50:
            risk_level = "VERY HIGH"
        elif risk_pct > 30:
            risk_level = "HIGH"
        elif risk_pct > 20:
            risk_level = "MEDIUM"
        
        return {
            'risk_level': risk_level,
            'position_risk_percentage': round(risk_pct, 2),
            'leverage_used': leverage,
            'risk_grade': "A" if risk_level == "LOW" else ("B" if risk_level == "MEDIUM" else "F")
        }

    def _analyze_trade_timing(self, trade_record: Dict) -> Dict:
        """Analyze trade timing characteristics"""
        duration_seconds = trade_record['duration_seconds']
        duration_minutes = duration_seconds / 60
        duration_hours = duration_minutes / 60
        
        # Classify holding period
        if duration_minutes < 5:
            timing_type = "SCALP"
        elif duration_minutes < 60:
            timing_type = "SHORT_TERM"
        elif duration_hours < 24:
            timing_type = "INTRADAY"
        else:            timing_type = "SWING"
        
        return {
            'timing_type': timing_type,
            'duration_minutes': round(duration_minutes, 1),
            'duration_hours': round(duration_hours, 1),
            'entry_time': trade_record['entry_datetime'],
            'exit_time': trade_record['exit_datetime']
        }

    def _analyze_reward_breakdown(self, trade_record: Dict) -> Dict:
        """Analyze reward system breakdown for insights"""
        entry_rewards = trade_record.get('entry_reward_breakdown', {})
        exit_rewards = trade_record.get('exit_reward_breakdown', {})
        
        # Combine rewards
        total_rewards = {}
        all_keys = set(entry_rewards.keys()) | set(exit_rewards.keys())
        
        for key in all_keys:
            entry_val = entry_rewards.get(key, 0.0)
            exit_val = exit_rewards.get(key, 0.0)
            
            # Ensure both values are numeric
            try:
                entry_val = float(entry_val) if entry_val is not None else 0.0
                exit_val = float(exit_val) if exit_val is not None else 0.0
                total_rewards[key] = entry_val + exit_val
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue
        
        # Find dominant reward components
        sorted_rewards = sorted(total_rewards.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'total_reward_points': sum(total_rewards.values()),
            'entry_reward_points': sum(v for v in entry_rewards.values() if isinstance(v, (int, float))),
            'exit_reward_points': sum(v for v in exit_rewards.values() if isinstance(v, (int, float))),
            'dominant_factors': sorted_rewards[:3],
            'reward_efficiency': trade_record['net_pnl'] / max(abs(sum(total_rewards.values())), 1.0)
        }

    def _analyze_market_context(self, trade_record: Dict) -> Dict:
        """Analyze market context during trade"""
        entry_market = trade_record.get('market_data_at_entry', {})
        exit_market = trade_record.get('market_data_at_exit', {})
        
        analysis = {
            'entry_context': entry_market,
            'exit_context': exit_market
        }
        
        # Analyze trend if data available
        if 'sma_short' in entry_market and 'sma_long' in entry_market:
            entry_price = trade_record['entry_price']
            sma_short = entry_market['sma_short']
            sma_long = entry_market['sma_long']
            
            if entry_price > sma_short > sma_long:
                analysis['entry_trend'] = "UPTREND"
            elif entry_price < sma_short < sma_long:
                analysis['entry_trend'] = "DOWNTREND"
            else:
                analysis['entry_trend'] = "SIDEWAYS"
        
        return analysis

    def _generate_improvement_suggestions(self, trade_record: Dict) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []
          # Performance-based suggestions
        if trade_record['win_loss'] == "LOSS":
            if trade_record['pnl_percentage'] < -5:
                suggestions.append("Consider tighter stop-loss to limit large losses")
            if trade_record['leverage'] > 5:
                suggestions.append("Reduce leverage to decrease risk exposure")
        
        # Risk management suggestions
        balance = trade_record['balance_before_entry']
        position_value = trade_record['position_value_at_entry']
        
        if balance > 0:
            risk_pct = (position_value / balance * 100)
            if risk_pct > 30:
                suggestions.append("Reduce position size relative to account balance")
        
        # Timing suggestions
        if trade_record['duration_seconds'] < 300:  # Less than 5 minutes
            suggestions.append("Consider longer holding periods to reduce commission impact")
        
        # Reward system suggestions
        reward_analysis = trade_record.get('trade_analysis', {}).get('reward_analysis', {})
        if reward_analysis.get('total_reward_points', 0) < 0:
            suggestions.append("Focus on trades with positive reward signals")
        
        return suggestions

    def _write_trace_event(self, event_type: str, event_data: Dict) -> None:
        """
        Writes a trace event to the JSONL file.

        Args:
            event_type: The type of event (e.g., "TRADE_OPENED", "TRADE_CLOSED").
            event_data: The dictionary containing the event's data.
        """
        if not self.enable_tracing or self.trace_file is None:
            return

        try:
            self.trade_counter += 1
            
            trace_record = {
                'trace_metadata': {
                    'trace_version': '1.1',
                    'generated_at': datetime.now().isoformat(),
                    'session_name': self.session_name,
                    'trace_id': f"{self.session_name}-{self.trade_counter}",
                    'event_type': event_type,
                    'episode': self.episode_num,
                    'batch': self.batch_num
                },
                'event_data': event_data
            }

            with open(self.trace_file, 'a', encoding='utf-8') as f:
                json.dump(trace_record, f, ensure_ascii=False)
                f.write('\n')
            
            self.logger.debug(f"Trace event '{event_type}' written for trade {event_data.get('trade_id', 'N/A')}")

        except Exception as e:
            self.logger.error(f"Error writing trace event: {e}")

    def get_session_summary(self) -> Dict:
        """Get summary of current session from JSONL file"""
        if not self.enable_tracing or not self.trace_file or not self.trace_file.exists():
            return {
                'total_trades': 0,
                'trace_file': str(self.trace_file) if self.trace_file else 'None',
                'open_positions': len(self.open_positions)
            }
        
        # Analyze completed trades from JSONL file
        wins = 0
        losses = 0
        total_pnl = 0.0
        total_commission = 0.0
        total_trades = 0
        
        try:
            with open(self.trace_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            trace_record = json.loads(line)
                            if trace_record.get('trace_metadata', {}).get('event_type') == 'TRADE_CLOSED':
                                total_trades += 1
                                trade_details = trace_record['event_data']
                                
                                if trade_details.get('win_loss') == 'WIN':
                                    wins += 1
                                elif trade_details.get('win_loss') == 'LOSS':
                                    losses += 1
                                
                                total_pnl += trade_details.get('net_pnl', 0.0)
                                total_commission += trade_details.get('total_commission', 0.0)
                            
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse trace line: {e}")
                            continue
        except Exception as e:
            self.logger.warning(f"Error reading trace file: {e}")
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            'session_name': self.session_name,
            'trace_file': str(self.trace_file),
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'total_commission': round(total_commission, 6),
            'net_profit': round(total_pnl - total_commission, 2),
            'open_positions': len(self.open_positions),
            'episode': self.episode_num,
            'batch': self.batch_num
        }

    def log_invalid_action(self, 
                          action_type: int,
                          action_str: str,
                          reason: str,
                          size_pct: float,
                          current_price: float,
                          balance: float,
                          position_size: float,
                          market_timestamp: Optional[pd.Timestamp] = None,
                          market_data: Optional[Dict] = None,
                          observation_space: Optional[Dict] = None,
                          confidence: float = 1.0,
                          episode_step: int = 0,
                          reward_info: Optional[Dict] = None) -> None:
        """
        Log invalid action with reward details to discourage such actions
        
        Args:
            action_type: The attempted action type (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)
            action_str: String representation of the action
            reason: Reason why the action was invalid
            size_pct: The size percentage that was attempted
            current_price: Current market price
            balance: Current account balance
            position_size: Current position size
            market_timestamp: Market timestamp for this action
            market_data: Market indicators at action time
            observation_space: The agent's observation space at the time of the action
            confidence: AI confidence in this action
            episode_step: RL episode step
            reward_info: Reward breakdown for this invalid action
        """
        
        try:
            if not self.enable_tracing or self.trace_file is None:
                return  # Skip tracing if disabled
            
            # Get timestamp
            if market_timestamp is None:
                timestamp = datetime.now()
            else:
                if isinstance(market_timestamp, pd.Timestamp):
                    timestamp = market_timestamp.to_pydatetime()
                else:
                    timestamp = pd.to_datetime(market_timestamp).to_pydatetime()
            
            # Clean and validate inputs
            current_price = float(np.nan_to_num(current_price, nan=0.0))
            balance = float(np.nan_to_num(balance, nan=0.0))
            position_size = float(np.nan_to_num(position_size, nan=0.0))
            confidence = float(np.nan_to_num(confidence, nan=1.0))
            size_pct = float(np.nan_to_num(size_pct, nan=0.0))
            
            # Create invalid action record with episode/batch info
            event_data = {
                "action_details": {
                    "action_type": action_type,
                    "action_str": action_str,
                    "reason": reason,
                    "attempted_size_pct": round(size_pct, 6),
                    "timestamp": timestamp.isoformat(),
                    "datetime": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "episode_step": episode_step,
                    "confidence": round(confidence, 4)
                },
                "market_context": {
                    "current_price": round(current_price, 8),
                    "account_balance": round(balance, 2),
                    "position_size": round(position_size, 8),
                    "position_side": "LONG" if position_size > 0 else ("SHORT" if position_size < 0 else "FLAT"),
                    "market_data": market_data or {},
                    "observation": observation_space or {}
                },
                "reward_analysis": {
                    "reward_breakdown": reward_info or {},
                    "punishment_applied": True,
                    "learning_signal": "This action was invalid and should be avoided",
                    "recommendations": [
                        f"Avoid {action_str} actions with invalid parameters",
                        "Check position state before attempting to close positions",
                        "Ensure size_pct is positive for buy/sell actions",
                        "Use valid action types (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)"
                    ]
                },
                "analysis": {
                    "invalid_action_type": self._classify_invalid_action(action_type, action_str, reason),
                    "severity": self._assess_invalid_action_severity(action_str, reason),
                    "common_causes": self._get_common_causes(action_str),
                    "prevention_tips": self._get_prevention_tips(action_str)
                }
            }
            
            self._write_trace_event("INVALID_ACTION", event_data)
            
            self.logger.debug(f"Invalid action logged: Episode {self.episode_num}, Batch {self.batch_num}, Action: {action_str} - {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to log invalid action: {e}")
    
    def _classify_invalid_action(self, action_type: int, action_str: str, reason: str) -> str:
        """Classify the type of invalid action"""
        if "invalid size_pct" in reason.lower():
            return "INVALID_SIZE_PARAMETER"
        elif "no open position" in reason.lower():
            return "NO_POSITION_TO_CLOSE"
        elif "invalid action type" in reason.lower():
            return "INVALID_ACTION_TYPE"
        else:
            return "UNKNOWN_INVALID_ACTION"
    
    def _assess_invalid_action_severity(self, action_str: str, reason: str) -> str:
        """Assess the severity of the invalid action"""
        if "INVALID_ACTION" in action_str:
            return "HIGH"
        elif "CLOSE_INVALID" in action_str:
            return "MEDIUM"
        elif "invalid size_pct" in reason.lower():
            return "LOW"
        else:
            return "MEDIUM"
    
    def _get_common_causes(self, action_str: str) -> List[str]:
        """Get common causes for this type of invalid action"""
        if "BUY_INVALID" in action_str or "SELL_INVALID" in action_str:
            return [
                "Size percentage is zero or negative",
                "Model output is not properly normalized",
                "Action space bounds are not respected"
            ]
        elif "CLOSE_INVALID" in action_str:
            return [
                "Trying to close position when no position is open",
                "Position tracking is out of sync",
                "Multiple close attempts on same position"
            ]
        elif "INVALID_ACTION" in action_str:
            return [
                "Action type is outside valid range [0,3]",
                "Model output is corrupted or malformed",
                "Action space discretization error"
            ]
        else:
            return ["Unknown cause"]
    
    def _get_prevention_tips(self, action_str: str) -> List[str]:
        """Get prevention tips for this type of invalid action"""
        if "BUY_INVALID" in action_str or "SELL_INVALID" in action_str:
            return [
                "Ensure size_pct is positive before executing trades",
                "Add validation in action preprocessing",
                "Check model output bounds and normalization",
                "Implement action masking for invalid size values"
            ]
        elif "CLOSE_INVALID" in action_str:
            return [
                "Check position_size != 0 before allowing close actions",
                "Implement position state validation",
                "Use action masking to prevent close when no position exists",
                "Add position tracking consistency checks"
            ]
        elif "INVALID_ACTION" in action_str:
            return [
                "Constrain action space to valid range [0,3]",
                "Add action type validation in environment",
                "Implement action clipping/normalization",
                "Check model architecture for action head bounds"
            ]
        else:
            return ["Review action validation logic"]

    def close_all_open_positions(self, reason: str = "Session ended", current_episode_step: int = None):
        """Close all open positions (for session cleanup)"""
        for trade_id in list(self.open_positions.keys()):
            trade_record = self.open_positions[trade_id]
            
            # Use provided episode step or fallback to entry step + 1 to ensure duration > 0
            exit_episode_step = current_episode_step if current_episode_step is not None else (trade_record['episode_step_at_entry'] + 1)
            
            # Create a basic close record
            self.log_trade_close(
                trade_id=trade_id,
                exit_action="FORCE_CLOSE",                exit_price=trade_record['entry_price'],  # Use entry price as fallback
                commission=0.0,
                balance_before=trade_record['balance_after_entry'],
                balance_after=trade_record['balance_after_entry'],
                market_timestamp=datetime.now(),
                episode_step=exit_episode_step,
                reward_info={'forced_close': True, 'reason': reason}
            )

    def __del__(self):
        """Destructor to ensure cleanup"""
        # This method is intentionally left empty to prevent errors during
        # Python's shutdown sequence, where modules like 'logging' or 'numpy'
        # might already be unloaded, causing unpredictable errors.
        pass


def main():
    """Example usage of TradeTracer"""
    # Initialize tracer
    tracer = TradeTracer(session_name="test_trace_session")
      # Simulate opening a trade
    tracer.log_trade_open(
        trade_id="TRADE_001",
        action="BUY",
        symbol="BINANCEFTS_PERP_BTC_USDT",
        entry_price=45000.0,
        position_size=0.1,
        leverage=2.0,
        commission=2.25,
        balance_before=10000.0,
        balance_after=9997.75,
        market_data={'sma_short': 44800, 'sma_long': 44000, 'rsi': 65},
        observation_space={'price_feature': [1.0, 0.9, 1.1], 'volume_feature': [0.5, 0.6, 0.4]},
        confidence=0.85,
        trade_reason="Strong bullish breakout above resistance",
        episode_step=100,
        reward_info={
            'profit_loss_points': 0.0,
            'position_management_points': -1.0,
            'risk_management_points': 5.0,
            'market_timing_points': 3.0,
            'consistency_points': 0.0,
            'commission_points': -0.45,
            'total_points': 6.55
        }
    )
    
    # Simulate closing the trade
    tracer.log_trade_close(
        trade_id="TRADE_001",
        exit_action="CLOSE",
        exit_price=46000.0,
        commission=2.30,
        balance_before=10100.0,
        balance_after=10095.45,
        market_data={'sma_short': 45900, 'sma_long': 44200, 'rsi': 72},
        observation_space={'price_feature': [1.2, 1.1, 1.3], 'volume_feature': [0.7, 0.8, 0.6]},
        episode_step=150,
        reward_info={
            'profit_loss_points': 22.2,
            'position_management_points': -0.5,
            'risk_management_points': 5.0,
            'market_timing_points': 2.0,
            'consistency_points': 2.0,
            'commission_points': -0.46,
            'total_points': 30.24
        }
    )
    
    # Print session summary
    summary = tracer.get_session_summary()
    print("\nTradeTracer Session Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
