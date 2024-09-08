def exit_trade(signal):
    signal_strength = signal["strength"]
    if signal_strength >= 0.3 and signal_strength <= 0.4:
        return True  # Strong sell signal detected
    elif 0.6 < signal_strength <= 0.7:
        return True  # Weak signal detected


def exit_on_support_resistance(df, pivot_point_data):
    # Assuming 'resistance1' and 'support1' come from pivot_point_data
    current_price = df['close'].iloc[-1]
    
    if current_price >= pivot_point_data['resistance1']:
        return True
    elif current_price <= pivot_point_data['support1']:
        return True