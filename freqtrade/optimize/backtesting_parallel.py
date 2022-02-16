# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from joblib import Parallel, delayed, dump, load
from typing import Any, Dict, List, Optional, Tuple

from pandas import DataFrame
import time

from freqtrade import constants
from freqtrade.configuration import TimeRange, validate_config_consistency
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data import history
from freqtrade.data.btanalysis import find_existing_backtest_stats, trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframe, trim_dataframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import BacktestState, SellType, RunMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.misc import get_strategy_run_id
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.bt_progress import BTProgress
from freqtrade.optimize.optimize_reports import (generate_backtest_stats, show_backtest_results,
                                                 store_backtest_stats)
from freqtrade.persistence import LocalTrade, Order, PairLocks, Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy.interface import IStrategy, SellCheckTuple
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)

# Indexes for backtest tuples
DATE_IDX = 0
BUY_IDX = 1
OPEN_IDX = 2
CLOSE_IDX = 3
SELL_IDX = 4
LOW_IDX = 5
HIGH_IDX = 6
BUY_TAG_IDX = 7
EXIT_TAG_IDX = 8


class BacktestingParallel:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        LoggingMixin.show_output = False
        self.config = config
        self.results: Dict[str, Any] = {}

        config['dry_run'] = True
        self.run_ids: Dict[str, str] = {}
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}

        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
        self.dataprovider = DataProvider(self.config, self.exchange)

        if self.config.get('strategy_list', None):
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver.load_strategy(stratconf))
                validate_config_consistency(stratconf)

        else:
            # No strategy list specified, only one strategy
            self.strategylist.append(StrategyResolver.load_strategy(self.config))
            validate_config_consistency(self.config)

        if "timeframe" not in self.config:
            raise OperationalException("Timeframe (ticker interval) needs to be set in either "
                                       "configuration or as cli argument `--timeframe 5m`")
        self.timeframe = str(self.config.get('timeframe'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)
        self.init_backtest_detail()
        self.pairlists = PairListManager(self.exchange, self.config)
        if 'VolumePairList' in self.pairlists.name_list:
            raise OperationalException("VolumePairList not allowed for backtesting. "
                                       "Please use StaticPairlist instead.")
        if 'PerformanceFilter' in self.pairlists.name_list:
            raise OperationalException("PerformanceFilter not allowed for backtesting.")

        if len(self.strategylist) > 1 and 'PrecisionFilter' in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if config.get('fee', None) is not None:
            self.fee = config['fee']
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])

        self.timerange = TimeRange.parse_timerange(
            None if self.config.get('timerange') is None else str(self.config.get('timerange')))

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        # Add maximum startup candle count to configuration for informative pairs support
        self.config['startup_candle_count'] = self.required_startup
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)
        self.init_backtest()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        LoggingMixin.show_output = True
        PairLocks.use_db = True
        Trade.use_db = True

    def init_backtest_detail(self):
        # Load detail timeframe if specified
        self.timeframe_detail = str(self.config.get('timeframe_detail', ''))
        if self.timeframe_detail:
            self.timeframe_detail_min = timeframe_to_minutes(self.timeframe_detail)
            if self.timeframe_min <= self.timeframe_detail_min:
                raise OperationalException(
                    "Detail timeframe must be smaller than strategy timeframe.")

        else:
            self.timeframe_detail_min = 0
        self.detail_data: Dict[str, DataFrame] = {}

    def init_backtest(self):

        self.prepare_backtest(False)

        self.wallets = Wallets(self.config, self.exchange, log=False)

        self.progress = BTProgress()
        self.abort = False

    def _set_strategy(self, strategy: IStrategy):
        """
        Load strategy into backtesting
        """
        self.strategy: IStrategy = strategy
        strategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        strategy.wallets = self.wallets
        # Set stoploss_on_exchange to false for backtesting,
        # since a "perfect" stoploss-sell is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types['stoploss_on_exchange'] = False

    def _load_protections(self, strategy: IStrategy):
        if self.config.get('enable_protections', False):
            conf = self.config
            if hasattr(strategy, 'protections'):
                conf = deepcopy(conf)
                conf['protections'] = strategy.protections
            self.protections = ProtectionManager(self.config, strategy.protections)

    def load_bt_data(self) -> Tuple[Dict[str, DataFrame], TimeRange]:
        """
        Loads backtest data and returns the data combined with the timerange
        as tuple.
        """
        self.progress.init_step(BacktestState.DATALOAD, 1)

        file_id = str(self.config.get('timerange'))
        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
        cache_file = None
        if self.config.get('backtest_cache', True):
            cache_file = (self.config['user_data_dir'] /
                          'backtest_load_cache' / f'data_{file_id}_{self.exchange.name}.pkl')

        data = history.load_data(
            datadir=self.config['datadir'],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=self.timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
            cache_file=cache_file,
        )

        min_date, max_date = history.get_timerange(data)

        logger.info(f'Loading data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days).')

        # Adjust startts forward if not enough data is available
        self.timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe),
                                                 self.required_startup, min_date)

        self.progress.set_new_value(1)
        return data, self.timerange

    def load_bt_data_detail(self) -> None:
        """
        Loads backtest detail data (smaller timeframe) if necessary.
        """
        if self.timeframe_detail:
            file_id = str(self.config.get('timerange'))
            self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
            cache_file = None
            if self.config.get('backtest_cache', True):
                cache_file = (self.config['user_data_dir'] /
                              'backtest_load_cache' / f'data_details_{file_id}_{self.exchange.name}.pkl')

            self.detail_data = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=self.timeframe_detail,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config.get('dataformat_ohlcv', 'json'),
                cache_file=cache_file,
            )
        else:
            self.detail_data = {}

    def prepare_backtest(self, enable_protections):
        """
        Backtesting setup method - called once for every call to "backtest()".
        """
        PairLocks.use_db = False
        PairLocks.timeframe = self.config['timeframe']
        Trade.use_db = False
        PairLocks.reset_locks()
        Trade.reset_trades()
        self.rejected_trades = 0
        self.dataprovider.clear_cache()
        if enable_protections:
            self._load_protections(self.strategy)

    def check_abort(self):
        """
        Check if abort was requested, raise DependencyException if that's the case
        Only applies to Interactive backtest mode (webserver mode)
        """
        if self.abort:
            self.abort = False
            raise DependencyException("Stop requested")

    def _get_ohlcv_as_lists_for_one_pair(self, pair_data: DataFrame, pair: str) -> Tuple:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        """
        tik = time.perf_counter()
        # Every change to this headers list must evaluate further usages of the resulting tuple
        # and eventually change the constants for indexes at the top
        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high', 'buy_tag', 'exit_tag']
        # data: Dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(pair_data))

        # Create dict with data

        self.check_abort()
        self.progress.increment()
        if not pair_data.empty:
            pair_data.loc[:, 'buy'] = 0  # cleanup if buy_signal is exist
            pair_data.loc[:, 'sell'] = 0  # cleanup if sell_signal is exist
            pair_data.loc[:, 'buy_tag'] = None  # cleanup if buy_tag is exist
            pair_data.loc[:, 'exit_tag'] = None  # cleanup if exit_tag is exist

        df_analyzed = self.strategy.advise_sell(
            self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair}).copy()
        # Trim startup period from analyzed dataframe
        df_analyzed = pair_data = trim_dataframe(
            df_analyzed, self.timerange, startup_candles=self.required_startup)
        # To avoid using data from future, we use buy/sell signals shifted
        # from the previous candle
        df_analyzed.loc[:, 'buy'] = df_analyzed.loc[:, 'buy'].shift(1)
        df_analyzed.loc[:, 'sell'] = df_analyzed.loc[:, 'sell'].shift(1)
        df_analyzed.loc[:, 'buy_tag'] = df_analyzed.loc[:, 'buy_tag'].shift(1)
        df_analyzed.loc[:, 'exit_tag'] = df_analyzed.loc[:, 'exit_tag'].shift(1)

        # Update dataprovider cache
        self.dataprovider._set_cached_df(pair, self.timeframe, df_analyzed)

        df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)

        # Convert from Pandas to list for performance reasons
        # (Looping Pandas is slow.)
        data = df_analyzed[headers].values.tolist()
        tok = time.perf_counter()
        logger.info(f"ohlcv_as_lists method took {tok - tik:0.4f} seconds, for {len(data)} pair(s) ")
        return data

    def _get_ohlcv_as_lists(self, processed: Dict[str, DataFrame]) -> Dict[str, Tuple]:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        """
        tik = time.perf_counter()
        # Every change to this headers list must evaluate further usages of the resulting tuple
        # and eventually change the constants for indexes at the top
        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high', 'buy_tag', 'exit_tag']
        data: Dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))

        # Create dict with data
        for pair in processed.keys():
            pair_data = processed[pair]
            self.check_abort()
            self.progress.increment()
            if not pair_data.empty:
                pair_data.loc[:, 'buy'] = 0  # cleanup if buy_signal is exist
                pair_data.loc[:, 'sell'] = 0  # cleanup if sell_signal is exist
                pair_data.loc[:, 'buy_tag'] = None  # cleanup if buy_tag is exist
                pair_data.loc[:, 'exit_tag'] = None  # cleanup if exit_tag is exist

            df_analyzed = self.strategy.advise_sell(
                self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair}).copy()
            # Trim startup period from analyzed dataframe
            df_analyzed = processed[pair] = pair_data = trim_dataframe(
                df_analyzed, self.timerange, startup_candles=self.required_startup)
            # To avoid using data from future, we use buy/sell signals shifted
            # from the previous candle
            df_analyzed.loc[:, 'buy'] = df_analyzed.loc[:, 'buy'].shift(1)
            df_analyzed.loc[:, 'sell'] = df_analyzed.loc[:, 'sell'].shift(1)
            df_analyzed.loc[:, 'buy_tag'] = df_analyzed.loc[:, 'buy_tag'].shift(1)
            df_analyzed.loc[:, 'exit_tag'] = df_analyzed.loc[:, 'exit_tag'].shift(1)

            # Update dataprovider cache
            self.dataprovider._set_cached_df(pair, self.timeframe, df_analyzed)

            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            data[pair] = df_analyzed[headers].values.tolist()
        tok = time.perf_counter()
        logger.info(f"ohlcv_as_lists method took {tok - tik:0.4f} seconds, for {len(data)} pair(s) ")
        return data

    def _get_close_rate(self, sell_row: Tuple, trade: LocalTrade, sell: SellCheckTuple,
                        trade_dur: int) -> float:
        """
        Get close rate for backtesting result
        """
        # Special handling if high or low hit STOP_LOSS or ROI
        if sell.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            if trade.stop_loss > sell_row[HIGH_IDX]:
                # our stoploss was already higher than candle high,
                # possibly due to a cancelled trade exit.
                # sell at open price.
                return sell_row[OPEN_IDX]

            # Special case: trailing triggers within same candle as trade opened. Assume most
            # pessimistic price movement, which is moving just enough to arm stoploss and
            # immediately going down to stop price.
            if sell.sell_type == SellType.TRAILING_STOP_LOSS and trade_dur == 0:
                if (
                    not self.strategy.use_custom_stoploss and self.strategy.trailing_stop
                    and self.strategy.trailing_only_offset_is_reached
                    and self.strategy.trailing_stop_positive_offset is not None
                    and self.strategy.trailing_stop_positive
                ):
                    # Worst case: price reaches stop_positive_offset and dives down.
                    stop_rate = (sell_row[OPEN_IDX] *
                                 (1 + abs(self.strategy.trailing_stop_positive_offset) -
                                  abs(self.strategy.trailing_stop_positive)))
                else:
                    # Worst case: price ticks tiny bit above open and dives down.
                    stop_rate = sell_row[OPEN_IDX] * (1 - abs(trade.stop_loss_pct))
                    assert stop_rate < sell_row[HIGH_IDX]
                # Limit lower-end to candle low to avoid sells below the low.
                # This still remains "worst case" - but "worst realistic case".
                return max(sell_row[LOW_IDX], stop_rate)

            # Set close_rate to stoploss
            return trade.stop_loss
        elif sell.sell_type == (SellType.ROI):
            roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
            if roi is not None and roi_entry is not None:
                if roi == -1 and roi_entry % self.timeframe_min == 0:
                    # When forceselling with ROI=-1, the roi time will always be equal to trade_dur.
                    # If that entry is a multiple of the timeframe (so on candle open)
                    # - we'll use open instead of close
                    return sell_row[OPEN_IDX]

                # - (Expected abs profit + open_rate + open_fee) / (fee_close -1)
                close_rate = - (trade.open_rate * roi + trade.open_rate *
                                (1 + trade.fee_open)) / (trade.fee_close - 1)

                if (trade_dur > 0 and trade_dur == roi_entry
                        and roi_entry % self.timeframe_min == 0
                        and sell_row[OPEN_IDX] > close_rate):
                    # new ROI entry came into effect.
                    # use Open rate if open_rate > calculated sell rate
                    return sell_row[OPEN_IDX]

                return close_rate

            else:
                # This should not be reached...
                return sell_row[OPEN_IDX]
        else:
            return sell_row[OPEN_IDX]

    def _get_adjust_trade_entry_for_candle(self, trade: LocalTrade, row: Tuple
                                           ) -> LocalTrade:

        current_profit = trade.calc_profit_ratio(row[OPEN_IDX])
        min_stake = self.exchange.get_min_pair_stake_amount(trade.pair, row[OPEN_IDX], -0.1)
        max_stake = self.wallets.get_available_stake_amount()
        stake_amount = strategy_safe_wrapper(self.strategy.adjust_trade_position,
                                             default_retval=None)(
            trade=trade, current_time=row[DATE_IDX].to_pydatetime(), current_rate=row[OPEN_IDX],
            current_profit=current_profit, min_stake=min_stake, max_stake=max_stake)

        # Check if we should increase our position
        if stake_amount is not None and stake_amount > 0.0:
            pos_trade = self._enter_trade(trade.pair, row, stake_amount, trade)
            if pos_trade is not None:
                return pos_trade

        return trade

    def _get_sell_trade_entry_for_candle(self, trade: LocalTrade,
                                         sell_row: Tuple) -> Optional[LocalTrade]:

        # Check if we need to adjust our current positions
        if self.strategy.position_adjustment_enable:
            check_adjust_buy = True
            if self.strategy.max_entry_position_adjustment > -1:
                count_of_buys = trade.nr_of_successful_buys
                check_adjust_buy = (count_of_buys <= self.strategy.max_entry_position_adjustment)
            if check_adjust_buy:
                trade = self._get_adjust_trade_entry_for_candle(trade, sell_row)

        sell_candle_time = sell_row[DATE_IDX].to_pydatetime()
        sell = self.strategy.should_sell(trade, sell_row[OPEN_IDX],  # type: ignore
                                         sell_candle_time, sell_row[BUY_IDX],
                                         sell_row[SELL_IDX],
                                         low=sell_row[LOW_IDX], high=sell_row[HIGH_IDX])

        if sell.sell_flag:
            trade.close_date = sell_candle_time

            trade_dur = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            closerate = self._get_close_rate(sell_row, trade, sell, trade_dur)
            # call the custom exit price,with default value as previous closerate
            current_profit = trade.calc_profit_ratio(closerate)
            if sell.sell_type in (SellType.SELL_SIGNAL, SellType.CUSTOM_SELL):
                # Custom exit pricing only for sell-signals
                closerate = strategy_safe_wrapper(self.strategy.custom_exit_price,
                                                  default_retval=closerate)(
                    pair=trade.pair, trade=trade,
                    current_time=sell_row[DATE_IDX],
                    proposed_rate=closerate, current_profit=current_profit)
            # Use the maximum between close_rate and low as we cannot sell outside of a candle.
            closerate = min(max(closerate, sell_row[LOW_IDX]), sell_row[HIGH_IDX])

            # Confirm trade exit:
            time_in_force = self.strategy.order_time_in_force['sell']
            if not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                    pair=trade.pair, trade=trade, order_type='limit', amount=trade.amount,
                    rate=closerate,
                    time_in_force=time_in_force,
                    sell_reason=sell.sell_reason,
                    current_time=sell_candle_time):
                return None

            trade.sell_reason = sell.sell_reason

            # Checks and adds an exit tag, after checking that the length of the
            # sell_row has the length for an exit tag column
            if(
                len(sell_row) > EXIT_TAG_IDX
                and sell_row[EXIT_TAG_IDX] is not None
                and len(sell_row[EXIT_TAG_IDX]) > 0
            ):
                trade.sell_reason = sell_row[EXIT_TAG_IDX]

            trade.close(closerate, show_msg=False)
            return trade

        return None

    def _get_sell_trade_entry(self, trade: LocalTrade, sell_row: Tuple) -> Optional[LocalTrade]:
        if self.timeframe_detail and trade.pair in self.detail_data:
            sell_candle_time = sell_row[DATE_IDX].to_pydatetime()
            sell_candle_end = sell_candle_time + timedelta(minutes=self.timeframe_min)

            detail_data = self.detail_data[trade.pair]
            detail_data = detail_data.loc[
                (detail_data['date'] >= sell_candle_time) &
                (detail_data['date'] < sell_candle_end)
            ].copy()
            if len(detail_data) == 0:
                # Fall back to "regular" data if no detail data was found for this candle
                return self._get_sell_trade_entry_for_candle(trade, sell_row)
            detail_data.loc[:, 'buy'] = sell_row[BUY_IDX]
            detail_data.loc[:, 'sell'] = sell_row[SELL_IDX]
            detail_data.loc[:, 'buy_tag'] = sell_row[BUY_TAG_IDX]
            detail_data.loc[:, 'exit_tag'] = sell_row[EXIT_TAG_IDX]
            headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high', 'buy_tag', 'exit_tag']
            for det_row in detail_data[headers].values.tolist():
                res = self._get_sell_trade_entry_for_candle(trade, det_row)
                if res:
                    return res

            return None

        else:
            return self._get_sell_trade_entry_for_candle(trade, sell_row)

    def _get_entry_rate(self, pair: str, buy_row) -> float:
        rate = buy_row[OPEN_IDX]
        # print(buy_row)
        if self.timeframe_detail and pair in self.detail_data:
            buy_candle_time = buy_row[DATE_IDX].to_pydatetime()
            buy_candle_end = buy_candle_time + timedelta(minutes=self.timeframe_min)

            detail_data = self.detail_data[pair]
            detail_data = detail_data.loc[
                (detail_data['date'] >= buy_candle_time) &
                (detail_data['date'] < buy_candle_end)
            ].copy()
            if len(detail_data) > 0:
                buy_candle = detail_data.iloc[0]
                # Worst case
                rate = buy_candle['high']
            return rate

        else:
            return rate

    def _enter_trade(self, pair: str, row: Tuple, stake_amount: Optional[float] = None,
                     trade: Optional[LocalTrade] = None) -> Optional[LocalTrade]:

        current_time = row[DATE_IDX].to_pydatetime()
        entry_tag = row[BUY_TAG_IDX] if len(row) >= BUY_TAG_IDX + 1 else None
        # let's call the custom entry price, using the open price as default price
        propose_rate = strategy_safe_wrapper(self.strategy.custom_entry_price,
                                             default_retval=row[OPEN_IDX])(
            pair=pair, current_time=current_time,
            proposed_rate=row[OPEN_IDX], entry_tag=entry_tag)  # default value is the open rate

        # Move rate to within the candle's low/high rate
        propose_rate = min(max(propose_rate, row[LOW_IDX]), row[HIGH_IDX])

        min_stake_amount = self.exchange.get_min_pair_stake_amount(pair, propose_rate, -0.05) or 0
        max_stake_amount = self.wallets.get_available_stake_amount()

        pos_adjust = trade is not None
        if not pos_adjust:
            try:
                stake_amount = self.wallets.get_trade_stake_amount(pair, None)
            except DependencyException:
                return trade

            stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount,
                                                 default_retval=stake_amount)(
                pair=pair, current_time=current_time, current_rate=propose_rate,
                proposed_stake=stake_amount, min_stake=min_stake_amount, max_stake=max_stake_amount,
                entry_tag=entry_tag)

        stake_amount = self.wallets.validate_stake_amount(pair, stake_amount, min_stake_amount)

        if not stake_amount:
            # In case of pos adjust, still return the original trade
            # If not pos adjust, trade is None
            return trade

        order_type = self.strategy.order_types['buy']
        time_in_force = self.strategy.order_time_in_force['sell']
        # Confirm trade entry:
        if not pos_adjust:
            if not strategy_safe_wrapper(self.strategy.confirm_trade_entry, default_retval=True)(
                    pair=pair, order_type=order_type, amount=stake_amount, rate=propose_rate,
                    time_in_force=time_in_force, current_time=current_time,
                    entry_tag=entry_tag):
                return None

        if stake_amount and (not min_stake_amount or stake_amount > min_stake_amount):
            amount = round(stake_amount / propose_rate, 8)
            if trade is None:
                # Enter trade
                trade = LocalTrade(
                    pair=pair,
                    open_rate=propose_rate,
                    open_date=current_time,
                    stake_amount=stake_amount,
                    amount=amount,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                    buy_tag=entry_tag,
                    exchange='backtesting',
                    orders=[]
                )
            trade.adjust_stop_loss(trade.open_rate, self.strategy.stoploss, initial=True)

            order = Order(
                ft_is_open=False,
                ft_pair=trade.pair,
                symbol=trade.pair,
                ft_order_side="buy",
                side="buy",
                order_type="market",
                status="closed",
                order_date=current_time,
                order_filled_date=current_time,
                order_update_date=current_time,
                price=propose_rate,
                average=propose_rate,
                amount=amount,
                filled=amount,
                cost=stake_amount + trade.fee_open
            )
            trade.orders.append(order)
            if pos_adjust:
                trade.recalc_trade_from_orders()

        return trade

    def handle_left_open(self, processed: Dict[str, DataFrame]) -> List[LocalTrade]:
        """
        Handling of left open trades at the end of backtesting
        """
        logger.info('Handle left open trades')
        data: Dict[str, Tuple] = {}
        trades = []
        for pair in processed.keys():
            for trade in LocalTrade.get_trades_proxy(pair=pair, is_open=True):

                if not data.get(pair):
                    data[pair] = self._get_ohlcv_as_lists_for_one_pair(processed[pair], pair)
                sell_row = data[pair][-1]

                trade.close_date = sell_row[DATE_IDX].to_pydatetime()
                trade.sell_reason = SellType.FORCE_SELL.value
                trade.close(sell_row[OPEN_IDX], show_msg=False)
                LocalTrade.close_bt_trade(trade)
                # Deepcopy object to have wallets update correctly
                trade1 = deepcopy(trade)
                trade1.is_open = True
                trades.append(trade1)
        return trades

    def trade_slot_available(self, max_open_trades: int, open_trade_count: int) -> bool:
        # Always allow trades when max_open_trades is enabled.
        if max_open_trades <= 0 or open_trade_count < max_open_trades:
            return True
        # Rejected trade
        self.rejected_trades += 1
        return False

    @delayed
    def backtest_one_pair(self, pair: str, processed: DataFrame, tmp, position_stacking, end_date, enable_protections):
        tik = time.perf_counter()
        self.prepare_backtest(enable_protections)

        data: Tuple = self._get_ohlcv_as_lists_for_one_pair(processed, pair)

        row_index = 0
        while tmp <= end_date:
            self.check_abort()

            try:
                # Row is treated as "current incomplete candle".
                # Buy / sell signals are shifted by 1 to compensate for this.
                row = data[row_index]
            except IndexError:
                # missing Data for one pair at the end.
                # Warnings for this are shown during data loading
                tmp += timedelta(minutes=self.timeframe_min)
                continue

            # Waits until the time-counter reaches the start of the data for this pair.
            if row[DATE_IDX] > tmp:
                tmp += timedelta(minutes=self.timeframe_min)
                continue

            row_index += 1
            self.dataprovider._set_dataframe_max_index(row_index)

            # without positionstacking, we can only have one open trade per pair.
            # max_open_trades must be respected
            # don't open on the last row
            if (
                    (position_stacking or len(LocalTrade.trades_open) == 0)
                    and tmp != end_date
                    and row[BUY_IDX] == 1
                    and row[SELL_IDX] != 1
                    and not PairLocks.is_pair_locked(pair, tmp)
            ):
                trade = self._enter_trade(pair, row)
                if trade:
                    # TODO: hacky workaround to avoid opening > max_open_trades
                    # This emulates previous behaviour - not sure if this is correct
                    # Prevents buying if the trade-slot was freed in this candle
                    LocalTrade.add_bt_trade(trade)

            for trade in list(LocalTrade.trades_open):
                # also check the buying candle for sell conditions.
                trade_entry = self._get_sell_trade_entry(trade, row)
                # Sell occurred
                if trade_entry:
                    LocalTrade.close_bt_trade(trade)
                    if enable_protections:
                        self.protections.stop_per_pair(pair, row[DATE_IDX])
                        self.protections.global_stop(tmp)

            tmp += timedelta(minutes=self.timeframe_min)
        tok = time.perf_counter()
        logger.info(f"{pair} in {tok - tik:0.4f} s")
        return pair, row_index, LocalTrade.trades_open, LocalTrade.trades

    def backtest(self, processed:  Dict[str, DataFrame],
                 start_date: datetime, end_date: datetime,
                 max_open_trades: int = 0, position_stacking: bool = False,
                 enable_protections: bool = False) -> Dict[str, Any]:
        """
        Implement backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid extensive logging in this method and functions it calls.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        :param start_date: backtesting timerange start datetime
        :param end_date: backtesting timerange end datetime
        :param max_open_trades: maximum number of concurrent trades, <= 0 means unlimited
        :param position_stacking: do we allow position stacking?
        :param enable_protections: Should protections be enabled?
        :return: DataFrame with trades (results of backtesting)
        """
        tik = time.perf_counter()
        self.prepare_backtest(enable_protections)
        tmp = start_date + timedelta(minutes=self.timeframe_min)

        self.progress.init_step(BacktestState.BACKTEST, int(
            (end_date - start_date) / timedelta(minutes=self.timeframe_min)))

        if max_open_trades == float('inf'):
            with Parallel(n_jobs=-1) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f'Effective number of parallel workers used: {jobs}')
                res = parallel(
                    self.backtest_one_pair
                    (pair, processed[pair], tmp, position_stacking, end_date, enable_protections)
                    for pair in list(processed.keys())
                    )
            # for pair in list(processed.keys()):
            #     res = self.backtest_one_pair(pair, processed[pair], tmp, position_stacking, end_date, enable_protections)
            for pair, row_index, trades_open, trades in res:
                LocalTrade.trades_open += trades_open
                LocalTrade.trades += trades
        else:
            raise OperationalException("Only max_open_trades=-1 implemented")
        self.handle_left_open(processed=processed)
        self.wallets.update()

        results = trade_list_to_dataframe(LocalTrade.get_trades_proxy())
        tok = time.perf_counter()
        logger.info(f"backtest method took {tok - tik:0.4f} seconds, for {len(processed)} pair(s) ")
        return {
            'results': results,
            'config': self.strategy.config,
            'locks': PairLocks.get_all_locks(),
            'rejected_signals': self.rejected_trades,
            'final_balance': self.wallets.get_total(self.strategy.config['stake_currency']),
        }

    def backtest_one_strategy(self, strat: IStrategy, data: Dict[str, DataFrame], timerange: TimeRange):

        self.progress.init_step(BacktestState.ANALYZE, 0)

        logger.info("Running backtesting for Strategy %s", strat.get_strategy_name())
        backtest_start_time = datetime.now(timezone.utc)
        self._set_strategy(strat)

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)()

        # Use max_open_trades in backtesting, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            # Must come from strategy config, as the strategy may modify this setting.
            max_open_trades = self.strategy.config['max_open_trades']
        else:
            logger.info(
                'Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            max_open_trades = 0

        file_id = str(self.config.get('timerange'))
        data_pickle_file = (self.config['user_data_dir'] /
                                 'backtest_load_cache' / f'preprocessed_{type(strat).__name__}_{file_id}_{self.exchange.name}.pkl')
        custom_trade_info_pickle_file = (self.config['user_data_dir'] /
                                         'backtest_load_cache' / f'custom_trade_info_{type(strat).__name__}_{file_id}_{self.exchange.name}.pkl')
        p = Path(data_pickle_file)
        if self.config.get('backtest_cache', True) and p.is_file():
            with data_pickle_file.open('rb') as f:
                preprocessed = load(f, mmap_mode='r').copy()
            ct = Path(custom_trade_info_pickle_file)
            if ct.is_file():
                with custom_trade_info_pickle_file.open('rb') as f:
                    self.strategy.custom_trade_info = load(f, mmap_mode='c')
            logger.info("Data loaded from cache.")
        else:
            # need to reprocess data every time to populate signals
            preprocessed = self.strategy.advise_all_indicators(data)

            # Trim startup period from analyzed dataframe
            preprocessed = trim_dataframes(preprocessed, timerange, self.required_startup)

            dump(preprocessed, data_pickle_file)
            if hasattr(self.strategy, 'custom_trade_info'):
                dump(self.strategy.custom_trade_info, custom_trade_info_pickle_file)

            logger.info("Dataload complete. Calculating indicators")

        if not preprocessed:
            raise OperationalException(
                "No data left after adjusting for startup candles.")

        # Use preprocessed_tmp for date generation (the trimmed dataframe).
        # Backtesting will re-trim the dataframes after buy/sell signal generation.
        min_date, max_date = history.get_timerange(preprocessed)
        logger.info(f'Backtesting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days).')
        # Execute backtest and store results
        results = self.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
            max_open_trades=max_open_trades,
            position_stacking=self.config.get('position_stacking', False),
            enable_protections=self.config.get('enable_protections', False),
        )
        backtest_end_time = datetime.now(timezone.utc)
        results.update({
            'run_id': self.run_ids.get(strat.get_strategy_name(), ''),
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp()),
        })
        self.all_results[self.strategy.get_strategy_name()] = results

        # return min_date, max_date, self.strategy.get_strategy_name(), results
        return min_date, max_date

    def _get_min_cached_backtest_date(self):
        min_backtest_date = None
        backtest_cache_age = self.config.get('backtest_cache', constants.BACKTEST_CACHE_DEFAULT)
        if self.timerange.stopts == 0 or datetime.fromtimestamp(
           self.timerange.stopts, tz=timezone.utc) > datetime.now(tz=timezone.utc):
            logger.warning('Backtest result caching disabled due to use of open-ended timerange.')
        elif backtest_cache_age == 'day':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(days=1)
        elif backtest_cache_age == 'week':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(weeks=1)
        elif backtest_cache_age == 'month':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(weeks=4)
        return min_backtest_date

    def load_prior_backtest(self):
        self.run_ids = {
            strategy.get_strategy_name(): get_strategy_run_id(strategy)
            for strategy in self.strategylist
        }

        # Load previous result that will be updated incrementally.
        # This can be circumvented in certain instances in combination with downloading more data
        min_backtest_date = self._get_min_cached_backtest_date()
        if min_backtest_date is not None:
            self.results = find_existing_backtest_stats(
                self.config['user_data_dir'] / 'backtest_results', self.run_ids, min_backtest_date)

    def start(self) -> None:
        """
        Run backtesting end-to-end
        :return: None
        """
        data: Dict[str, Any] = {}

        data, timerange = self.load_bt_data()
        self.load_bt_data_detail()
        logger.info("Dataload complete. Calculating indicators")

        self.load_prior_backtest()

        for strat in self.strategylist:
            if self.results and strat.get_strategy_name() in self.results['strategy']:
                # When previous result hash matches - reuse that result and skip backtesting.
                logger.info(f'Reusing result of previous backtest for {strat.get_strategy_name()}')
                continue

            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)
        # ray.init()
        # futures = [self.backtest_one_strategy.remote(self, strat, data, timerange) for strat in self.strategylist]
        #
        # min_date, max_date = 0, 0
        # for min_date, max_date, strategy_name, results in ray.get(futures):
        #     self.all_results[strategy_name] = results

        # with Parallel(n_jobs=-1) as parallel:
        #     res = parallel(
        #         self.backtest_one_strategy
        #         (strat, data, timerange)
        #         for strat in self.strategylist
        #         )
        # for min_date, max_date, strategy_name, results in res:
        #     self.all_results[strategy_name] = results

        # Update old results with new ones.
        if len(self.all_results) > 0:
            results = generate_backtest_stats(
                data, self.all_results, min_date=min_date, max_date=max_date)
            if self.results:
                self.results['metadata'].update(results['metadata'])
                self.results['strategy'].update(results['strategy'])
                self.results['strategy_comparison'].extend(results['strategy_comparison'])
            else:
                self.results = results

            if self.config.get('export', 'none') == 'trades':
                store_backtest_stats(self.config['exportfilename'], self.results)

        # Results may be mixed up now. Sort them so they follow --strategy-list order.
        if 'strategy_list' in self.config and len(self.results) > 0:
            self.results['strategy_comparison'] = sorted(
                self.results['strategy_comparison'],
                key=lambda c: self.config['strategy_list'].index(c['key']))
            self.results['strategy'] = dict(
                sorted(self.results['strategy'].items(),
                       key=lambda kv: self.config['strategy_list'].index(kv[0])))

        if len(self.strategylist) > 0:
            # Show backtest results
            show_backtest_results(self.config, self.results)
