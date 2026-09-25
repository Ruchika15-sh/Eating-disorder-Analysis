-- schema.sql
-- UK Eating Disorder Analytics — MySQL schema
-- Run this first to create the database and tables, then load the CSVs
-- via src/load_to_mysql.py

CREATE DATABASE IF NOT EXISTS uk_ed_analytics
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE uk_ed_analytics;

-- ============================================================
-- Module 1: synthetic EAT-26 screening data (classifier training set)
-- ============================================================
DROP TABLE IF EXISTS eat26_synthetic;
CREATE TABLE eat26_synthetic (
    id                                INT AUTO_INCREMENT PRIMARY KEY,
    age                               INT NOT NULL,
    gender                            VARCHAR(10) NOT NULL,
    bmi_category                      VARCHAR(20) NOT NULL,
    dieting_score                     INT NOT NULL,
    bulimia_food_preoccupation_score  INT NOT NULL,
    oral_control_score                INT NOT NULL,
    total_eat26_score                 INT NOT NULL,
    risk_label                        VARCHAR(50) NOT NULL,
    INDEX idx_risk_label (risk_label)
) ENGINE=InnoDB;

-- ============================================================
-- Module 2/3: real NHS Digital admissions by subtype, all years
-- ============================================================
DROP TABLE IF EXISTS nhs_ed_subtypes;
CREATE TABLE nhs_ed_subtypes (
    financial_year     VARCHAR(7) NOT NULL,
    diag_code          VARCHAR(10) NOT NULL,
    diagnosis_label    VARCHAR(100) NOT NULL,
    fce_sum            INT,
    fae_sum            INT,
    fce_male_sum       INT,
    fce_female_sum     INT,
    PRIMARY KEY (financial_year, diag_code)
) ENGINE=InnoDB;

-- ============================================================
-- Module 2: age breakdown by subtype, 2024-25
-- ============================================================
DROP TABLE IF EXISTS nhs_ed_age_breakdown;
CREATE TABLE nhs_ed_age_breakdown (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    diag_code        VARCHAR(10) NOT NULL,
    diagnosis_label  VARCHAR(100) NOT NULL,
    financial_year   VARCHAR(7) NOT NULL,
    age_band         VARCHAR(20) NOT NULL,
    episode_count    INT,
    INDEX idx_diag_code (diag_code),
    INDEX idx_age_band (age_band)
) ENGINE=InnoDB;

-- ============================================================
-- Module 3: F50 total, one row per year (forecast input series)
-- ============================================================
DROP TABLE IF EXISTS ed_yearly_admissions;
CREATE TABLE ed_yearly_admissions (
    financial_year  VARCHAR(7) PRIMARY KEY,
    year_end        INT NOT NULL,
    admissions      INT NOT NULL
) ENGINE=InnoDB;

-- ============================================================
-- Module 4: ONS mortality by region, 2024
-- ============================================================
DROP TABLE IF EXISTS ons_ed_mortality;
CREATE TABLE ons_ed_mortality (
    region                          VARCHAR(50) PRIMARY KEY,
    all_cause_deaths                INT NOT NULL,
    ed_deaths                       INT NOT NULL,
    ed_death_rate_per_100k_allcause DECIMAL(10,4)
) ENGINE=InnoDB;
