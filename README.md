# **Insurance Claims Prediction**

## **EDA Summary**
After doing extensive EDA (or if you may like, a **data reveal party** 😊), we have the following observations:
- The column with most missing data is provider_type with 15.49% missing data points, followed by claim_finalized_date with 9.6%, rest have <3% missing data points. Visual representation in the missing data heatmap and barplot charts.
- Taking serial number as an identity column, the data represents information of 52187 customers.
- To reduce claim status to 2 most important valid categories, marked Resubmitted, Submitted, PartiallyRejected and Rejected categories as Not Approved.  
We have =~ 89.8% Approved claims and =~ 10.2% Not Approved.  
To take care of this imbalance, we might have to explore balancing techniques such as oversampling, undersampling or both.
- Corrected redundancy in provider_type column from 14 to 6 categories. See the provider type distribution chart.
- Also corrected redundancy in program_cover from =~35 to 5 categories. See program cover distribution chart.
- Participants are divided into 56% female and 44% males across gender.
- In item status column, items under 'SUBMITTED' are awaiting approval or rejection.  
Item status is a great feature, that is if it's not a leakage feature, such that the item status is determined at the health provider prior to being pushed to the insurance company (or that the claim is first decided upon, before a customer goes to the health provider).  
To test if this is a leakage feature we investigate to see if there are cases where item status is rejected and claim status is approved and see that there are a number of such cases!(458 in total)  
So we conclude there is no leakage. It's either item status or claim status is determined first.  

**MAJOR:** The Serial Number column is very determining of the claim status label. It looks like we can easily differentiate between a claim to be approved or unapproved based on it's serial number.

-----