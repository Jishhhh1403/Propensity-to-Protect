SELECT
  customer_id,
  @as_of_date AS as_of_date,
  date AS feature_date,
  DATE_DIFF(date, @as_of_date, DAY) AS day_offset,
  daily_income,
  daily_spend,
  tx_count,
  cat_flag_health,
  cat_flag_retail,
  channel_digital_pct,
  is_weekend,
  net_flow,
  daily_balance,
  rel_spend,
  bal_trajectory,
  velocity_ratio,
  spend_z_score,
  day_sin,
  day_cos,
  age,
  income_band,
  credit_score,
  current_income,
  city
FROM `{{PROJECT}}.{{DATASET}}.{{DAILY_FEATURES_TABLE}}`
WHERE customer_id = @customer_id
  AND date BETWEEN DATE_SUB(@as_of_date, INTERVAL 30 DAY)
               AND DATE_SUB(@as_of_date, INTERVAL 1 DAY)
ORDER BY date;
