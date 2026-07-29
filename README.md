# Small Business Loan Risk Prediction

A model that estimates default probability for SBA 7(a) loan applicants using only information available at the time of application.

## The Dataset

The U.S. Small Business Administration (SBA) releases loan performance data every quarter. We will be working with 1.9 million rows of loan information including borrower names, loan amounts, and interest rates. We will be predicting the status of the loan (paid in full or charged-off).

## The Problem: Two Sources of Leakage

**Things that look almost right are scarier than things that are obviously wrong.**

Before building anything, the features need to pass one test: would this information exist on a loan application? If a field is only known after the loan is issued or after it defaults, it can't be a model input.

**TermInMonths**

This field looks like contracted loan term. This is true for most loans. But for charged-off loans, non-standard values encode something else: the actual time elapsed before the borrower defaulted.

A model trained on this feature learns to recognize default after it already happened. The inflated result comes entirely from this one column.

**BusinessAge**

The SBA changed how it recorded business age in 2018. Before that time, business age was recorded with more specific buckets that differentiated businesses that were two, three, four, or 5+ years old. It moved towards a simpler over/under two years old method.

Training on this feature means the model learns a documentation artifact, not borrower risk. It was dropped and replaced with a binary **IsStartup** flag that is consistently derivable from the application at any point in time.
