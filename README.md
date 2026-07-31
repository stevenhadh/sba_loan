# Small Business Loan Risk Prediction

A model that estimates default probability for SBA 7(a) loan applicants using only information available at the time of application.

## The Dataset

The U.S. Small Business Administration (SBA) releases loan performance data every quarter. There are 1.9 million rows of loan information with fields
covering four categories of information:

- **Loan terms**: approved amount, SBA-guaranteed portion, initial
  interest rate, fixed vs. variable, term in months, subprogram, and
  processing method
- **Borrower identity**: name, address, ZIP code, NAICS industry
  description, franchise code, and jobs supported
- **Lender identity**: bank name, FDIC/NCUA number, and address
- **Geography**: borrower state, project county, SBA district office,
  and congressional district

We will be predicting two outcomes: `CHGOFF` or charge-off (for defaulted loans), and `PIF` or paid in full (for non-defaulted loans). Loans that are still active are excluded.

There is a lack of information such as borrower credit score, business financials, and any lender underwriting data. Every field in the data is structural or administrative, describing the loan and the parties, not the borrower's capacity to repay it. This is a core limitation.

## The Problem

**Things that look almost right are scarier than things that are obviously wrong.**

Before building anything, the features need to pass one test: would this information exist on a loan application? If a field is only known after the loan is issued or after it defaults, it can't be a model input.

### `terminmonths` leakage

This field looks like contracted loan term. This is true for most loans. 

<img width="735" height="315" alt="image" src="https://github.com/user-attachments/assets/a1296912-aa26-4154-aab4-87e7d581930d" />

But after examining the distribution of `terminmonths` values separately for defaulted and non-defaulted loans, the charged-off loans showed a cluster of values inconsistent with SBA's standard loan terms. These non-standard values encoded something else: the actual time elapsed before the borrower defaulted.

A model trained on this feature learns to recognize default after it already happened. A leaky model produces results that look great in validation but it will fall apart in deployment.

### `businessage` documentation change

The SBA changed how it recorded business age in 2018.

<img width="735" height="350" alt="image" src="https://github.com/user-attachments/assets/0b9d313f-7a6c-41cf-9c6b-d598cc332b77" />

| Old Categories |
|---|
| New — less than 1 year|
| Existing — 2 to 3 years |
| Existing — 3 to 4 years |
| Existing — 5 or more years |

| New Categories |
|---|
| Under 2 years |
| 2 years or more |

Training on this feature means that the model learns a documentation artifact, not borrower risk. It was dropped and replaced with a binary `is_startup` flag that is consistently derivable from the application at any point in time.

## Model Results

The model is an XGBoost classifier trained on a temporal split: trained on loans before 2018, tested on loans after 2018. This prevents future loans from leaking into training through a random shuffle.

**What is PR-AUC?** Accuracy and ROC-AUC are misleading on imbalanced datasets - when defaults are rare, a model that predicts "no default" for every loan scores well on both. Precision-Recall AUC (PR-AUC) measures performance specifically on the minority class (defaults), making it the appropriate metric here. A random classifier scores equal to the base default rate (0.08); A higher PR-AUC is better.

| | PR-AUC | Notes |
|---|---|---|
| Leaky model (with `terminmonths`) | ~0.72 | Learns time-to-default, not risk |
| Clean model | ~0.22 | ~2.75x lift over base rate |

The clean model's PR-AUC of 0.22 represents a real but weak signal: industry, geography, and loan size contain some information about default risk, but not enough to make loan-level predictions and underwrite individual loans from public data alone.
