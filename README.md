# Persona-Driven Creative GTM System

A lightweight project that connects **customer behavior** to **marketing creative generation**.

Instead of writing one generic advertisement, this project:
1) segments customers into **life-stage cohorts** and **behavioral personas** (spend × engagement), then  
2) turns a selected segment into a structured **creative brief**, and  
3) uses an LLM (Gemini) to generate persona-specific marketing copy variants.

## Why this exists
My initial idea was to generate copy based on “generations” and cultural language differences, but the dataset did not support that storyline cleanly. EDA showed that **age-level averages looked similar**, while meaningful differences appeared **within** age groups. That motivated the persona-based approach.

---

## Project structure

- `creative_gtm.py`  
  CLI tool that loads data, builds personas, creates a segment brief, and calls Gemini to generate copy.

- `EDA.ipynb`  
  Notebook used to explore the dataset and justify segmentation choices (plots + talking points for the demo).

- `marketing_campaign.csv`  
  Dataset file.

---

## Personas (2×2 framework)

### 1) Life-stage cohorts (age-based context)

Customers are first grouped into broad life-stage cohorts based on birth year. These cohorts provide contextual information but are not used alone to drive creative decisions.

- **Young Professionals**: born 1988 and later  
- **Mid-Career**: born 1976–1987  
- **Established**: born 1965–1975  
- **Legacy**: born before 1965  

EDA showed that aggregate response rates across these cohorts were nearly identical, indicating that age alone does not explain how customers respond to marketing.

### 2) Behavioral personas (defined within each cohort)

Within each life-stage cohort, customers are further segmented using **behavioral signals**.

Personas are defined **within each cohort** using cohort-specific medians:

- **Champions**: high spend, high engagement  
- **Premium-but-Quiet**: high spend, low engagement  
- **Deal-Responsive**: low spend, high engagement  
- **Hard-to-Convert**: low spend, low engagement  

Signals used:
- Spend proxy: `total_spend` (sum of product category spend)
- Engagement proxy: `past_accept_count` (sum of AcceptedCmp1–5)

---

## How to run

### Run personas only (no LLM)
python creative_gtm.py --csv marketing_campaign.csv --no_llm

### Run full generation (Gemini)
python creative_gtm.py \
  --csv marketing_campaign.csv \
  --cohort "Mid-Career" \
  --persona "Champions" \
  --channel email \
  --style playful

