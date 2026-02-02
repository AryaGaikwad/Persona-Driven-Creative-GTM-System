# Persona-Driven Creative GTM System

A lightweight project that connects **customer behavior** to **marketing creative generation**.

Instead of writing one generic prompt, this project:
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
  Dataset file (semicolon-delimited).

---

## Personas (2×2 framework)

Personas are defined **within each cohort** using cohort-specific medians:

- **Champions**: High spend, high engagement  
- **Premium-but-Quiet**: High spend, low engagement  
- **Deal-Responsive**: Low spend, high engagement  
- **Hard-to-Convert**: Low spend, low engagement

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

