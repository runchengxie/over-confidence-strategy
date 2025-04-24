import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 数据加载占位符函数 (需要您根据数据源自行实现) ---

def get_csi500_constituents_and_weights(date_str: str) -> pd.DataFrame:
    """
    获取指定日期的中证500成分股列表及其在该指数中的官方权重。

    Args:
        date_str (str): 日期字符串 (例如 'YYYY-MM-DD')

    Returns:
        pd.DataFrame: 包含至少 ['ticker', 'weight'] 列的DataFrame。
                      'ticker': 股票代码 (例如 '000001.SZ')
                      'weight': 股票在指数中的权重 (例如 0.005 表示 0.5%)
                      确保权重之和接近 1。
    """
    # --- 实现细节 ---
    # 连接您的数据源 (Wind, Tushare, JoinQuant, etc.)
    # 查询指定日期的中证500成分股及权重
    # 返回格式化的DataFrame
    # 示例返回 (需要替换为真实数据):
    print(f"模拟: 正在获取 {date_str} 的中证500成分股和权重...")
    # 实际应返回包含500只股票的DataFrame
    data = {
        'ticker': [f'{i:06d}.SZ' for i in range(1, 501)],
        # 权重需要从数据源获取，这里用均值代替
        'weight': [1/500] * 500
    }
    df = pd.DataFrame(data)
    # 确保权重列是数值类型
    df['weight'] = pd.to_numeric(df['weight'])
    # 确保权重和为1 (或接近1)
    df['weight'] = df['weight'] / df['weight'].sum()
    return df

def get_stock_data(tickers: list, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    获取一批股票在指定时间段内的历史日行情数据和股本数据。

    Args:
        tickers (list): 股票代码列表。
        start_date_str (str): 开始日期 'YYYY-MM-DD'。
        end_date_str (str): 结束日期 'YYYY-MM-DD'。

    Returns:
        pd.DataFrame: 包含 ['date', 'ticker', 'volume', 'float_shares'] 列的DataFrame。
                      'date': 日期
                      'ticker': 股票代码
                      'volume': 成交量
                      'float_shares': 流通股本 (用于计算换手率)
                      可能还需要收盘价 'close' 用于其他分析或回测。
    """
    # --- 实现细节 ---
    # 连接您的数据源
    # 查询指定股票列表和时间段的日行情数据（成交量）和流通股本数据
    # 返回包含所需列的DataFrame
    # 示例返回 (需要替换为真实数据):
    print(f"模拟: 正在获取 {len(tickers)} 只股票从 {start_date_str} 到 {end_date_str} 的数据...")
    all_data = []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    current_date = start_date
    while current_date <= end_date:
        for ticker in tickers:
            # 模拟数据，实际需要从数据源获取
            volume = np.random.randint(100000, 10000000)
            float_shares = np.random.randint(1e8, 1e10)
            all_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'volume': volume,
                'float_shares': float_shares
            })
        current_date += timedelta(days=1)

    if not all_data:
        return pd.DataFrame(columns=['date', 'ticker', 'volume', 'float_shares'])

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['float_shares'] = pd.to_numeric(df['float_shares'])
    return df

# --- 策略核心逻辑 ---

def calculate_monthly_turnover(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算每只股票上个月的平均日换手率。

    Args:
        stock_data (pd.DataFrame): 包含 ['date', 'ticker', 'volume', 'float_shares'] 的数据。
                                   应包含需要计算换手率的那个月的所有交易日数据。

    Returns:
        pd.DataFrame: 包含 ['ticker', 'avg_daily_turnover'] 的DataFrame。
    """
    if stock_data.empty or 'float_shares' not in stock_data.columns or stock_data['float_shares'].isnull().all():
         print("警告: 无法计算换手率，股本数据缺失或无效。")
         return pd.DataFrame(columns=['ticker', 'avg_daily_turnover'])

    # 确保 float_shares 不为 0，避免除零错误
    stock_data = stock_data[stock_data['float_shares'] > 0].copy()
    if stock_data.empty:
         print("警告: 过滤掉股本为0的数据后，没有有效数据计算换手率。")
         return pd.DataFrame(columns=['ticker', 'avg_daily_turnover'])

    stock_data['daily_turnover'] = stock_data['volume'] / stock_data['float_shares']
    # 按股票分组，计算上月日均换手率
    avg_turnover = stock_data.groupby('ticker')['daily_turnover'].mean().reset_index()
    avg_turnover.rename(columns={'daily_turnover': 'avg_daily_turnover'}, inplace=True)
    return avg_turnover

def construct_portfolio(date_str: str, n_group: int = 100) -> pd.DataFrame:
    """
    根据过度自信策略构建目标投资组合。

    Args:
        date_str (str): 调仓日期 'YYYY-MM-DD'。
        n_group (int): 高/低换手率分组的股票数量。

    Returns:
        pd.DataFrame: 包含 ['ticker', 'target_weight'] 的目标投资组合DataFrame。
                      目标权重之和应为 1。
    """
    print(f"\n--- 开始构建 {date_str} 的投资组合 ---")

    # 1. 获取当期中证500成分股及权重
    constituents = get_csi500_constituents_and_weights(date_str)
    if constituents.empty:
        print("错误: 无法获取成分股信息，无法构建组合。")
        return pd.DataFrame(columns=['ticker', 'target_weight'])
    print(f"获取到 {len(constituents)} 只成分股。")

    # 2. 获取上一个月的起止日期，用于计算换手率
    current_date = datetime.strptime(date_str, '%Y-%m-%d')
    end_date_prev_month = current_date - timedelta(days=1)
    start_date_prev_month = (end_date_prev_month.replace(day=1))
    start_date_str = start_date_prev_month.strftime('%Y-%m-%d')
    end_date_str = end_date_prev_month.strftime('%Y-%m-%d')

    # 3. 获取上个月的股票数据 (成交量, 流通股本)
    tickers = constituents['ticker'].tolist()
    stock_data_prev_month = get_stock_data(tickers, start_date_str, end_date_str)

    # 4. 计算上个月的平均日换手率
    turnover_df = calculate_monthly_turnover(stock_data_prev_month)
    if turnover_df.empty:
        print("警告: 未能计算换手率，将使用等权重（或原始权重）作为备选。")
        # 在此可以添加备选逻辑，例如返回原始权重
        constituents.rename(columns={'weight': 'target_weight'}, inplace=True)
        return constituents[['ticker', 'target_weight']]

    print(f"计算得到 {len(turnover_df)} 只股票的换手率。")

    # 5. 合并换手率到成分股数据中
    portfolio_data = pd.merge(constituents, turnover_df, on='ticker', how='left')
    # 对于没有换手率数据的股票（例如当月新上市或停牌），给予一个中性值（例如NaN或中位数）
    # 这里我们暂时填充NaN，排序时它们会被放在一边
    # portfolio_data['avg_daily_turnover'].fillna(portfolio_data['avg_daily_turnover'].median(), inplace=True)

    # 6. 根据换手率排序，识别高、低换手组
    # 注意处理NaN值，pandas的sort_values默认会将NaN放在最后
    portfolio_data = portfolio_data.sort_values(by='avg_daily_turnover', ascending=True, na_position='last')

    # 处理成分股数量不足的情况
    num_constituents = len(portfolio_data)
    actual_n_group = min(n_group, num_constituents // 3) # 确保分组合理
    if actual_n_group < n_group:
        print(f"警告: 成分股数量 ({num_constituents}) 不足，实际分组数量调整为 {actual_n_group}")
    if actual_n_group == 0 and num_constituents > 0:
         print(f"警告: 成分股数量 ({num_constituents}) 过少，无法有效分组，返回原始权重。")
         portfolio_data.rename(columns={'weight': 'target_weight'}, inplace=True)
         return portfolio_data[['ticker', 'target_weight']]
    elif num_constituents == 0:
         return pd.DataFrame(columns=['ticker', 'target_weight'])


    low_turnover_group = portfolio_data.head(actual_n_group)
    high_turnover_group = portfolio_data.tail(actual_n_group)
    # 使用 .iloc 进行索引，避免标签索引问题
    middle_group = portfolio_data.iloc[actual_n_group:-actual_n_group]

    print(f"低换手组: {len(low_turnover_group)} 只, 高换手组: {len(high_turnover_group)} 只, 中间组: {len(middle_group)} 只")

    # 7. 计算权重调整
    W_H_total = high_turnover_group['weight'].sum() # 高换手组总权重
    W_L_total = low_turnover_group['weight'].sum() # 低换手组总权重

    print(f"高换手组总权重: {W_H_total:.4f}, 低换手组总权重: {W_L_total:.4f}")

    # 创建目标权重列
    portfolio_data['target_weight'] = 0.0

    # 更新低换手组权重
    if W_L_total > 1e-8: # 避免除零
        low_indices = low_turnover_group.index
        portfolio_data.loc[low_indices, 'target_weight'] = portfolio_data.loc[low_indices, 'weight'] + \
                                                          (portfolio_data.loc[low_indices, 'weight'] / W_L_total) * W_H_total
    else:
        print("警告: 低换手组总权重过小，无法按比例分配权重。低换手组权重将保持不变。")
        low_indices = low_turnover_group.index
        portfolio_data.loc[low_indices, 'target_weight'] = portfolio_data.loc[low_indices, 'weight']


    # 更新中间组权重 (保持不变)
    middle_indices = middle_group.index
    portfolio_data.loc[middle_indices, 'target_weight'] = portfolio_data.loc[middle_indices, 'weight']

    # 高换手组权重为 0 (默认已经是0)

    # 8. 验证权重和并返回
    final_portfolio = portfolio_data[['ticker', 'target_weight']].copy()
    # 过滤掉权重为0的股票（高换手组）
    final_portfolio = final_portfolio[final_portfolio['target_weight'] > 1e-9] # 使用小的阈值避免浮点误差

    # 最终归一化，确保权重和为1
    total_target_weight = final_portfolio['target_weight'].sum()
    if total_target_weight > 1e-8:
        final_portfolio['target_weight'] = final_portfolio['target_weight'] / total_target_weight
    else:
        print("错误: 最终目标权重和为零，无法归一化。")
        return pd.DataFrame(columns=['ticker', 'target_weight'])


    print(f"构建完成，最终组合包含 {len(final_portfolio)} 只股票。")
    print(f"目标权重之和: {final_portfolio['target_weight'].sum():.4f}")

    return final_portfolio[['ticker', 'target_weight']]

# --- 主程序入口 (示例) ---
if __name__ == "__main__":
    # 假设今天是 2019-06-01，我们要构建这个月的组合
    rebalance_date = '2019-06-01' # 研报回测期的一个时点

    # 构建投资组合
    target_portfolio = construct_portfolio(rebalance_date)

    if not target_portfolio.empty:
        print("\n目标投资组合:")
        # 打印权重最高的几只股票作为示例
        print(target_portfolio.sort_values(by='target_weight', ascending=False).head())

        # --- 后续步骤 ---
        # 1. 计算与当前持仓的差异，生成交易订单
        # 2. 执行交易
        # 3. 进行业绩归因、风险分析等
        # 4. (用于回测) 记录当期组合，滚动到下一个调仓日
