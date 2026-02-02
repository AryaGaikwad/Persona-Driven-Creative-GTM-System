# creative_gtm.py
# STEP 1: Read semicolon-delimited data + light EDA + cohort prep
#
# Run:
#   python creative_gtm.py --csv marketing_campaign.csv

import argparse
import os
import sys
import pandas as pd
import json
import textwrap
from typing import Dict, Any, List
from google import genai
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Cohort assignment (life-stage, not fake generations)
# ----------------------------
def assign_cohort(year_birth):
    if pd.isna(year_birth):
        return "Unknown"
    y = int(year_birth)

    if y >= 1988:
        return "Young Professionals"
    elif 1976 <= y <= 1987:
        return "Mid-Career"
    elif 1965 <= y <= 1975:
        return "Established"
    else:
        return "Legacy"

# ----------------------------
# Load + prepare data
# ----------------------------
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # IMPORTANT: dataset is semicolon-separated
    df = pd.read_csv(csv_path, sep=";")

    accepted_cols = [
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5"
    ]

    required = [
        "Year_Birth", "Income", "Education", "Marital_Status",
        "Kidhome", "Teenhome",
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
        *accepted_cols,
        "Response"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ---- Basic cleaning ----
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")

    # Cohort (life stage)
    df["cohort"] = df["Year_Birth"].apply(assign_cohort)

    # Campaign engagement history (context, not prediction)
    df["past_accept_count"] = df[accepted_cols].sum(axis=1).astype(int)
    df["ever_accepted"] = (df["past_accept_count"] > 0).astype(int)

    # GTM outcome references (NOT used to judge AI yet)
    df["responded_last_campaign"] = df["Response"].astype(int)
    df["responded_any_campaign"] = (
        (df["ever_accepted"] == 1) | (df["Response"] == 1)
    ).astype(int)

    # Spending as preference/taste proxy
    spend_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df["total_spend"] = df[spend_cols].sum(axis=1)

    return df

# ----------------------------
# Light EDA (talking points)
# ----------------------------
# def run_eda(df: pd.DataFrame):
#     print("\n=== BASIC DATA OVERVIEW ===")
#     print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

#     print("\n=== Cohort distribution ===")
#     print(df["cohort"].value_counts())

#     print("\n=== Missing values (top 8) ===")
#     print(df.isna().sum().sort_values(ascending=False).head(8))

#     print("\n=== Campaign response rates by cohort ===")
#     print(
#         df.groupby("cohort")["responded_last_campaign"]
#         .mean()
#         .sort_values(ascending=False)
#         .round(3)
#     )

#     print("\n=== Avg total spend by cohort ===")
#     print(
#         df.groupby("cohort")["total_spend"]
#         .mean()
#         .sort_values(ascending=False)
#         .round(1)
#     )

#     print("\n=== Avg past campaign accept count by cohort ===")
#     print(
#         df.groupby("cohort")["past_accept_count"]
#         .mean()
#         .sort_values(ascending=False)
#         .round(2)
#     )

def assign_persona_2x2(row: pd.Series, spend_med: float, engage_med: float) -> str:
    """
    2x2 persona matrix:
      - Spend: High/Low based on cohort median total_spend
      - Engagement: High/Low based on cohort median past_accept_count

    Personas:
      - Champions: High spend, High engagement
      - Premium-but-Quiet: High spend, Low engagement
      - Deal-Responsive: Low spend, High engagement
      - Hard-to-Convert: Low spend, Low engagement
    """
    high_spend = row["total_spend"] >= spend_med
    high_engage = row["past_accept_count"] >= engage_med

    if high_spend and high_engage:
        return "Champions"
    if high_spend and not high_engage:
        return "Premium-but-Quiet"
    if (not high_spend) and high_engage:
        return "Deal-Responsive"
    return "Hard-to-Convert"

def build_personas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 2x2 personas per cohort using medians:
      - Spend median: total_spend
      - Engagement median: past_accept_count

    Returns one row per (cohort, persona) with summary stats.
    """
    persona_rows = []

    for cohort, g in df.groupby("cohort"):
        g = g.copy()

        spend_med = g["total_spend"].median()
        engage_med = g["past_accept_count"].median()

        # Edge case: if a cohort has median engagement 0, ">= 0" makes everyone high_engage.
        # Fix: if engage_med == 0, use responded_any_campaign as engagement proxy split.
        use_response_proxy = (engage_med == 0)

        if use_response_proxy:
            # Split engagement by responded_any_campaign (0/1)
            g["persona"] = g.apply(
                lambda r: assign_persona_2x2(
                    r,
                    spend_med=spend_med,
                    engage_med=1  # treat "engaged" as responded_any_campaign==1
                ) if r["responded_any_campaign"] in [0, 1] else "Hard-to-Convert",
                axis=1
            )
            # But we need high_engage based on responded_any_campaign:
            # We'll implement directly to avoid confusion:
            def _persona_proxy(r):
                high_spend = r["total_spend"] >= spend_med
                high_engage = r["responded_any_campaign"] == 1
                if high_spend and high_engage:
                    return "Champions"
                if high_spend and not high_engage:
                    return "Premium-but-Quiet"
                if (not high_spend) and high_engage:
                    return "Deal-Responsive"
                return "Hard-to-Convert"
            g["persona"] = g.apply(_persona_proxy, axis=1)
        else:
            g["persona"] = g.apply(assign_persona_2x2, axis=1, spend_med=spend_med, engage_med=engage_med)

        # Aggregate stats for each persona in cohort
        for persona, p in g.groupby("persona"):
            persona_rows.append({
                "cohort": cohort,
                "persona": persona,
                "count": len(p),
                "avg_income": round(p["Income"].mean(), 1),
                "avg_total_spend": round(p["total_spend"].mean(), 1),
                "response_rate_last_campaign": round(p["responded_last_campaign"].mean(), 3),
                "response_rate_any_campaign": round(p["responded_any_campaign"].mean(), 3),
                "avg_past_accept_count": round(p["past_accept_count"].mean(), 2),
                "kids_ratio": round((p["Kidhome"] + p["Teenhome"]).mean(), 2),
                "web_purchase_rate": round(p["NumWebPurchases"].mean(), 2),
                "store_purchase_rate": round(p["NumStorePurchases"].mean(), 2),
            })

    return pd.DataFrame(persona_rows)

def persona_messaging_focus(persona: str) -> str:
    if persona == "Champions":
        return "lean into confident, premium positioning and a strong hook; assume high intent"
    if persona == "Premium-but-Quiet":
        return "use trust-building and relevance first; avoid loud hype; emphasize quality"
    if persona == "Deal-Responsive":
        return "highlight concrete value, savings, bundles, and urgency without sounding spammy"
    return "keep it simple and low-pressure; focus on clarity, social proof, and reducing friction"

def persona_to_insight(row) -> str:
    messaging_focus = persona_messaging_focus(row["persona"])

    tone = (
        "more open to frequent campaigns"
        if row["response_rate_last_campaign"] > 0.15
        else "selective and campaign-fatigued"
    )

    channel_hint = (
        "digital-first"
        if row["web_purchase_rate"] > row["store_purchase_rate"]
        else "offline-inclined"
    )


    return (
        f"{row['persona']}s in the {row['cohort']} cohort tend to be {tone}. "
        f"They have an average spend of {row['avg_total_spend']} and show "
        f"{channel_hint} behavior. Messaging should {messaging_focus}."
    )

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY env var. Set it before running.")
    return genai.Client(api_key=api_key)

def gemini_text(client, prompt: str, model: str = "gemini-2.5-flash-lite") -> str:
    resp = client.models.generate_content(model=model, contents=prompt)

    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return getattr(resp, "text", "") or str(resp)

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # Remove first line ```json or ```
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()

def strategist_agent_prompt(brief: Dict[str, Any], product: str, channel: str, style: str) -> str:
    return f"""
You are a GTM strategist. You will produce a concise creative strategy for a marketing message.

PRODUCT:
{product}

CHANNEL:
{channel}

STYLE:
{style}

PERSONA BRIEF (data-backed):
{json.dumps(brief, indent=2)}

TASK:
Return a JSON object with keys:
- "positioning": 1 sentence positioning for this persona
- "hooks": list of 3 hook ideas
- "tone_rules": list of 4 tone rules
- "reference_vibes": list of 3 culturally-neutral reference vibes (NO copyrighted quotes, no brand names, no movie dialogue)
- "do_not": list of 4 things to avoid (compliance/brand safety)

Only output valid JSON.
""".strip()

def copywriter_agent_prompt(strategy: Dict[str, Any], product: str, channel: str) -> str:
    return f"""
You are a creative copywriter.

PRODUCT:
{product}

CHANNEL:
{channel}

STRATEGY (follow strictly):
{json.dumps(strategy, indent=2)}

TASK:
Generate 3 variants of marketing copy tailored to the persona.
Requirements:
- Each variant must be different (different hook angle).
- No copyrighted quotes, no movie dialogue, no brand names.
- Keep it short and punchy.
- End with a clear CTA.

Return ONLY JSON:
{{
  "variants": [
    {{"id": "A", "copy": "..."}},
    {{"id": "B", "copy": "..."}},
    {{"id": "C", "copy": "..."}}
  ]
}}
""".strip()

def critic_agent_prompt(brief: Dict[str, Any], strategy: Dict[str, Any], variants: List[Dict[str, str]]) -> str:
    return f"""
You are a growth marketer acting as a critic and evaluator.

PERSONA BRIEF:
{json.dumps(brief, indent=2)}

STRATEGY:
{json.dumps(strategy, indent=2)}

VARIANTS:
{json.dumps(variants, indent=2)}

TASK:
Score each variant from 1-10 on:
- persona_fit
- clarity
- hook_strength
- brand_safety

Then pick a winner and explain why in 2-3 sentences.

Return ONLY JSON:
{{
  "scores": [
    {{"id":"A","persona_fit":0,"clarity":0,"hook_strength":0,"brand_safety":0,"notes":"..."}},
    ...
  ],
  "winner_id": "...",
  "winner_reason": "..."
}}
""".strip()

def persona_row_to_brief(row) -> Dict[str, Any]:
    channel_pref = "digital" if row["web_purchase_rate"] > row["store_purchase_rate"] else "offline"
    return {
        "cohort": row["cohort"],
        "persona": row["persona"],
        "count": int(row["count"]),
        "avg_income": row["avg_income"],
        "avg_total_spend": row["avg_total_spend"],
        "response_rate_last_campaign": row["response_rate_last_campaign"],
        "avg_past_accept_count": row["avg_past_accept_count"],
        "kids_ratio": row["kids_ratio"],
        "channel_preference": channel_pref,
        "creative_direction": (
        "aspirational, premium, confident"
        if row["persona"] in ["Champions", "Premium-but-Quiet"]
        else "clear, trustworthy, value-focused"
        ),
    }

def extract_json(text: str) -> Any:
    text = strip_code_fences(text)

    # quick path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model output.")

    return json.loads(text[start:end+1])










def main():
    parser = argparse.ArgumentParser(
        description="Creative GTM: data-driven personas + AI-generated marketing copy"
    )

    # ---------- Core ----------
    parser.add_argument("--csv", required=True, help="Path to marketing_campaign.csv")

    # ---------- Persona selection ----------
    parser.add_argument("--cohort", default=None, help="Pick a cohort (e.g., 'Young Professionals')")
    parser.add_argument("--persona", default=None, help="Pick a persona ('Engaged Spender' or 'Value-Oriented')")

    # ---------- Creative inputs ----------
    parser.add_argument(
        "--product",
        default="A smarter way to run targeted campaigns using customer signals.",
        help="Product / value proposition"
    )
    parser.add_argument(
        "--channel",
        default="email",
        choices=["email", "linkedin", "ads"],
        help="Channel for the copy"
    )
    parser.add_argument(
        "--style",
        default="playful",
        choices=["playful", "straightforward", "meme-lite"],
        help="Creative style"
    )

    # ---------- Control ----------
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip Gemini calls and only print GTM personas"
    )

    args = parser.parse_args()

    df = load_and_prepare(args.csv)
    # run_eda(df)

    print("\n=== BUILDING GTM PERSONAS ===")
    personas = build_personas(df)
    personas["gtm_insight"] = personas.apply(persona_to_insight, axis=1)

    for _, row in personas.iterrows():
        print("\n---")
        print(f"Cohort: {row['cohort']}")
        print(f"Persona: {row['persona']}")
        print(f"Count: {row['count']}")
        print(f"GTM Insight: {row['gtm_insight']}")

    # ---------- Stop here if LLM disabled ----------
    if args.no_llm:
        print("\n(LLM disabled) Use --cohort and --persona to generate copy with Gemini.")
        return

    # ---------- Select persona ----------
    if args.cohort is None:
        args.cohort = personas["cohort"].iloc[0]
    if args.persona is None:
        args.persona = "Champions"

    selection = personas[
        (personas["cohort"] == args.cohort) &
        (personas["persona"] == args.persona)
    ]

    if selection.empty:
        raise ValueError(
            f"No match for cohort='{args.cohort}' persona='{args.persona}'. "
            f"Available cohorts: {sorted(personas['cohort'].unique().tolist())}"
        )

    row = selection.iloc[0]
    brief = persona_row_to_brief(row)

    print("\n=== SELECTED PERSONA (for copy generation) ===")
    print(json.dumps(brief, indent=2))

    # ---------- Gemini agents ----------
    client = get_gemini_client()

    # Strategist
    strat_prompt = strategist_agent_prompt(brief, args.product, args.channel, args.style)
    strategy = extract_json(gemini_text(client, strat_prompt))

    # Copywriter
    copy_prompt = copywriter_agent_prompt(strategy, args.product, args.channel)
    copy_obj = extract_json(gemini_text(client, copy_prompt))
    variants = copy_obj["variants"]

    # Critic
    critic_prompt = critic_agent_prompt(brief, strategy, variants)
    critique = extract_json(gemini_text(client, critic_prompt))

    

    # ---------- Gemini agents (RAW OUTPUT MODE) ----------
    # client = get_gemini_client()

    # print("\n================ RAW STRATEGIST OUTPUT ================\n")
    # strat_prompt = strategist_agent_prompt(brief, args.product, args.channel, args.style)
    # strat_text = gemini_text(client, strat_prompt)
    # print(strat_text)

    # print("\n================ RAW COPYWRITER OUTPUT ================\n")
    # copy_prompt = copywriter_agent_prompt(
    #     strategy={},  # dummy, not used when just inspecting
    #     product=args.product,
    #     channel=args.channel
    # )
    # copy_text = gemini_text(client, copy_prompt)
    # print(copy_text)

    # print("\n================ RAW CRITIC OUTPUT ================\n")
    # critic_prompt = critic_agent_prompt(
    #     brief=brief,
    #     strategy={},
    #     variants=[]
    # )
    # critic_text = gemini_text(client, critic_prompt)
    # print(critic_text)

    # print("\n================ END RAW OUTPUT ================\n")
    # return


    # ---------- Output ----------
    print("\n=== STRATEGY ===")
    
    print(json.dumps(strategy, indent=2))
    strat_text = gemini_text(client, strat_prompt)

    

    print("\n=== VARIANTS ===")
    for v in variants:
        print("\n--- Variant", v["id"], "---")
        print(textwrap.fill(v["copy"], width=90))

    print("\n=== SCORECARD ===")
    for s in critique["scores"]:
        print(
            f"{s['id']}: fit={s['persona_fit']}/10 "
            f"clarity={s['clarity']}/10 "
            f"hook={s['hook_strength']}/10 "
            f"safety={s['brand_safety']}/10 | {s.get('notes','')}"
        )

    print("\n=== WINNER ===")
    print("Winner:", critique["winner_id"])
    print(textwrap.fill(critique["winner_reason"], width=90))



# ----------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
