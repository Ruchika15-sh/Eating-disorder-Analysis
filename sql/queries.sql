-- queries.sql
-- UK Eating Disorder Analytics — analytical queries
-- Run against the uk_ed_analytics MySQL database (after schema.sql + load_to_mysql.py)

USE uk_ed_analytics;

-- ============================================================
-- 1. Subtype breakdown, 2024-25 (excludes the F50 grand total row)
--    -> "Which eating disorder subtype accounts for the most hospital
--        episodes, and what share is that of the total?"
-- ============================================================
SELECT
    diagnosis_label,
    fce_sum AS episodes,
    ROUND(fce_sum * 100.0 / (
        SELECT fce_sum FROM nhs_ed_subtypes
        WHERE financial_year = '2024-25' AND diag_code = 'F50'
    ), 1) AS pct_of_total,
    ROUND(fce_female_sum * 100.0 / fce_sum, 1) AS pct_female
FROM nhs_ed_subtypes
WHERE financial_year = '2024-25' AND diag_code != 'F50'
ORDER BY episodes DESC;


-- ============================================================
-- 2. Year-over-year change in total admissions
--    -> "How has the overall admission volume trended since 2021-22?"
-- ============================================================
SELECT
    financial_year,
    admissions,
    admissions - LAG(admissions) OVER (ORDER BY financial_year) AS change_from_prior_year,
    ROUND(
        (admissions - LAG(admissions) OVER (ORDER BY financial_year)) * 100.0
        / LAG(admissions) OVER (ORDER BY financial_year), 1
    ) AS pct_change
FROM ed_yearly_admissions
ORDER BY financial_year;


-- ============================================================
-- 3. Which subtype grew or shrank the most between 2021-22 and 2024-25?
--    -> self-join to compare the same diagnosis code across two years
-- ============================================================
SELECT
    a.diagnosis_label,
    a.fce_sum AS episodes_2021_22,
    b.fce_sum AS episodes_2024_25,
    b.fce_sum - a.fce_sum AS episode_change,
    ROUND((b.fce_sum - a.fce_sum) * 100.0 / a.fce_sum, 1) AS pct_change
FROM nhs_ed_subtypes a
JOIN nhs_ed_subtypes b
    ON a.diag_code = b.diag_code
WHERE a.financial_year = '2021-22'
    AND b.financial_year = '2024-25'
    AND a.diag_code != 'F50'
ORDER BY pct_change DESC;


-- ============================================================
-- 4. Age band with the highest episode count, per subtype
--    -> window function to rank age bands within each diagnosis
-- ============================================================
SELECT diagnosis_label, age_band, episode_count
FROM (
    SELECT
        diagnosis_label,
        age_band,
        episode_count,
        RANK() OVER (PARTITION BY diag_code ORDER BY episode_count DESC) AS rnk
    FROM nhs_ed_age_breakdown
) ranked
WHERE rnk = 1
ORDER BY episode_count DESC;


-- ============================================================
-- 5. EAT-26 synthetic data: average subscale scores per risk pattern
--    -> "What does the model actually learn to separate on?"
-- ============================================================
SELECT
    risk_label,
    COUNT(*) AS n,
    ROUND(AVG(dieting_score), 1) AS avg_dieting,
    ROUND(AVG(bulimia_food_preoccupation_score), 1) AS avg_bulimia_preoccupation,
    ROUND(AVG(oral_control_score), 1) AS avg_oral_control,
    ROUND(AVG(total_eat26_score), 1) AS avg_total_score
FROM eat26_synthetic
GROUP BY risk_label
ORDER BY avg_total_score DESC;


-- ============================================================
-- 6. EAT-26: risk pattern distribution by BMI category
--    -> cross-tab style query using conditional aggregation
-- ============================================================
SELECT
    bmi_category,
    SUM(risk_label = 'Low Risk') AS low_risk,
    SUM(risk_label = 'Restrictive (AN-type)') AS restrictive_an,
    SUM(risk_label = 'Bulimic (BN-type)') AS bulimic_bn,
    SUM(risk_label = 'Binge (BED-type)') AS binge_bed,
    COUNT(*) AS total
FROM eat26_synthetic
GROUP BY bmi_category
ORDER BY total DESC;


-- ============================================================
-- 7. Regional mortality rate, ranked, with England & Wales average
--    for comparison (subquery in SELECT)
-- ============================================================
SELECT
    region,
    ed_deaths,
    all_cause_deaths,
    ed_death_rate_per_100k_allcause,
    ROUND(
        (SELECT AVG(ed_death_rate_per_100k_allcause) FROM ons_ed_mortality), 2
    ) AS national_avg_rate
FROM ons_ed_mortality
ORDER BY ed_death_rate_per_100k_allcause DESC;


-- ============================================================
-- 8. Emergency vs planned admission share, 2024-25 subtypes
--    -> requires fae_sum + would need fae_emergency_sum if loaded;
--       shown here as the FCE/FAE ratio as a simple proxy metric
-- ============================================================
SELECT
    diagnosis_label,
    fce_sum,
    fae_sum,
    ROUND(fce_sum * 1.0 / NULLIF(fae_sum, 0), 2) AS episodes_per_admission
FROM nhs_ed_subtypes
WHERE financial_year = '2024-25' AND diag_code != 'F50'
ORDER BY episodes_per_admission DESC;
