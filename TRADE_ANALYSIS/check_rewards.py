import pandas as pd

# Read the minimal file
df = pd.read_csv('trade_analysis_detailed_minimal.csv')

print("REWARD ANALYSIS:")
print(f"Total trades: {len(df)}")
print(f"Total non-zero rewards: {(df['close_reward'] != 0).sum()}")
print(f"Positive rewards: {(df['close_reward'] > 0).sum()}")
print(f"Negative rewards: {(df['close_reward'] < 0).sum()}")
print(f"Reward range: {df['close_reward'].min():.4f} to {df['close_reward'].max():.4f}")

print("\nSample rewards:")
print(df[['trade_id', 'net_pnl', 'close_reward', 'win_loss']].head(10))

print("\nReward vs PnL comparison:")
print(f"Rewards match PnL: {(df['close_reward'] == df['net_pnl']).sum()} trades")
print(f"Percentage match: {(df['close_reward'] == df['net_pnl']).sum() / len(df) * 100:.1f}%")
