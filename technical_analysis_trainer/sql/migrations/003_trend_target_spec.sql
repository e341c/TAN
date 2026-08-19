-- Record what each trend prediction actually predicted.
--
-- `predicted_trend` is "UP"/"DOWN" under both targets, but next_bar means "the
-- next bar closes beyond +/-threshold" while triple_barrier means "price touches
-- +k*sigma before -k*sigma within N bars" -- a different threshold over a
-- different horizon. Without this column, rows written by different models are
-- indistinguishable and the accuracy panel scores them all against whichever
-- target happens to be configured today.
--
-- NULL means "written before the spec existed" and is interpreted as next_bar,
-- which is what every run prior to 2026-08-16 was trained on.

ALTER TABLE trend_predictions
    ADD COLUMN IF NOT EXISTS target_spec TEXT;

COMMENT ON COLUMN trend_predictions.target_spec IS
    'JSON target definition (mode, barrier multipliers, horizon). NULL => next_bar.';
