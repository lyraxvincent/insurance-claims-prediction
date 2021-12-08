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

## **Feature Engineering preview**

```python

def feature_engineer(data):
    # label encoder object
    le = LabelEncoder()
    
    # new features from dates
    data['participant_age'] = ((pd.to_datetime(datetime.date.today()) - data.participant_date_of_birth).dt.days) / 364
    
    data['participant_yearOB'] = data.participant_date_of_birth.dt.year.astype(int)
    data['participant_monthOB'] = data.participant_date_of_birth.dt.month.astype(int)
    data['participant_dayOB'] = data.participant_date_of_birth.dt.day.astype(int)
    
    data['treat_cr_year'] = data.treatment_created_date.dt.year.astype(int)
    data['treat_cr_month'] = data.treatment_created_date.dt.month.astype(int)
    data['treat_cr_day'] = data.treatment_created_date.dt.day.astype(int)
    data['treat_cr_weekday'] = pd.Series(data.treatment_created_date.dt.weekday).apply(lambda x: 1 if x<5 else 0).astype(int)
    
    data['claim_final_year'] = data.claim_finalized_date.dt.year.astype(int)
    data['claim_final_month'] = data.claim_finalized_date.dt.month.astype(int)
    data['claim_final_day'] = data.claim_finalized_date.dt.day.astype(int)
    data['claim_final_weekday'] = pd.Series(data.claim_finalized_date.dt.weekday).apply(lambda x: 1 if x<5 else 0).astype(int)
    
    # days from treatment creation to claim finalization
    data['treat_claim_diff'] = (data.claim_finalized_date - data.treatment_created_date).dt.days
    
    # categorize some continuous variables from information on plots above
    data['totals_cat'] = np.select([
        data.total_item_amount <= 1200,
        (data.total_item_amount > 1200) & (data.total_item_amount <= 2300),
        (data.total_item_amount > 2300) & (data.total_item_amount <= 4500),
        (data.total_item_amount > 4500) & (data.total_item_amount <= 10000),
        data.total_item_amount > 10000
    ], [3, 5, 1, 2, 4])
    
    
    data['itemq_cat'] = np.select([
        data.item_quantity <= 200,
        data.item_quantity > 200
    ], [0, 1])
    
    
    data['age_cat'] = np.select([
        data.participant_age <= 20,
        (data.participant_age > 20) & (data.participant_age <= 40),
        (data.participant_age > 40) & (data.participant_age <= 60),
        data.participant_age > 60
    ], [3, 0, 2, 1])
    
    # some combination features
    data['prov_typeXreg'] = le.fit_transform(data.provider_type + data.provider_region)
    data['prov_typeXcover'] = le.fit_transform(data.provider_type + data.program_cover)
    data['prov_regXcover'] = le.fit_transform(data.provider_region + data.program_cover)
    data['prov_typeXgen'] = le.fit_transform(data.provider_type + data.participant_gender)
    data['prov_regXgen'] = le.fit_transform(data.provider_region + data.participant_gender)
    data['coverXgen'] = le.fit_transform(data.program_cover + data.participant_gender)
    data['regXstatus'] = le.fit_transform(data.provider_region + data.item_status)
    data['statusXgen'] = le.fit_transform(data.item_status + data.participant_gender)
    data['coverXstatus'] = le.fit_transform(data.program_cover + data.item_status)
    
    # encoding categorical features
    for col in ['provider_type', 'provider_region', 'program_cover', 'participant_gender', 'item_status', 'item_name']:
        data[col] = le.fit_transform(data[col])
    
    # convert 'continuous' columns with < 5 unique values to categorical
    for col in data.select_dtypes(np.number).columns:
        if data[col].nunique() < 5:
            data[col] = le.fit_transform(data[col])
        else:
            pass
        
    return data


# apply function to data
data = pd.read_csv("data/data_clean.csv", parse_dates=['participant_date_of_birth', 'treatment_created_date',
                                                       'claim_finalized_date'])

data = feature_engineer(data)

```

-----

## **Algorithms testing preview**

```python

X = data[data.select_dtypes(np.number).columns[1:]]
y = data.claim_status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)

models = [
    LogisticRegression(),
    SGDClassifier(),
    MultinomialNB(),
    RandomForestClassifier(),
    CatBoostClassifier(verbose=False),
    LGBMClassifier(),
    XGBClassifier()
]

for model in models:
    print(model.__class__.__name__, "\n", "="*40)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(confusion_matrix(y_test, pred), "\n\n", classification_report(y_test, pred))

```

**Output:**

```python

LogisticRegression 
 ========================================
[[74746   382]
 [ 1222  7236]] 

               precision    recall  f1-score   support

    Approved       0.98      0.99      0.99     75128
Not Approved       0.95      0.86      0.90      8458

    accuracy                           0.98     83586
   macro avg       0.97      0.93      0.94     83586
weighted avg       0.98      0.98      0.98     83586

SGDClassifier 
 ========================================
[[75123     5]
 [ 7917   541]] 

               precision    recall  f1-score   support

    Approved       0.90      1.00      0.95     75128
Not Approved       0.99      0.06      0.12      8458

    accuracy                           0.91     83586
   macro avg       0.95      0.53      0.54     83586
weighted avg       0.91      0.91      0.87     83586

MultinomialNB 
 ========================================
[[59361 15767]
 [ 4815  3643]] 

               precision    recall  f1-score   support

    Approved       0.92      0.79      0.85     75128
Not Approved       0.19      0.43      0.26      8458

    accuracy                           0.75     83586
   macro avg       0.56      0.61      0.56     83586
weighted avg       0.85      0.75      0.79     83586

RandomForestClassifier 
 ========================================
[[75073    55]
 [    5  8453]] 

               precision    recall  f1-score   support

    Approved       1.00      1.00      1.00     75128
Not Approved       0.99      1.00      1.00      8458

    accuracy                           1.00     83586
   macro avg       1.00      1.00      1.00     83586
weighted avg       1.00      1.00      1.00     83586

CatBoostClassifier 
 ========================================
[[75060    68]
 [    6  8452]] 

               precision    recall  f1-score   support

    Approved       1.00      1.00      1.00     75128
Not Approved       0.99      1.00      1.00      8458

    accuracy                           1.00     83586
   macro avg       1.00      1.00      1.00     83586
weighted avg       1.00      1.00      1.00     83586

LGBMClassifier 
 ========================================
[[75077    51]
 [    7  8451]] 

               precision    recall  f1-score   support

    Approved       1.00      1.00      1.00     75128
Not Approved       0.99      1.00      1.00      8458

    accuracy                           1.00     83586
   macro avg       1.00      1.00      1.00     83586
weighted avg       1.00      1.00      1.00     83586

XGBClassifier 
 ========================================
[18:46:42] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[[75078    50]
 [   11  8447]] 

               precision    recall  f1-score   support

    Approved       1.00      1.00      1.00     75128
Not Approved       0.99      1.00      1.00      8458

    accuracy                           1.00     83586
   macro avg       1.00      1.00      1.00     83586
weighted avg       1.00      1.00      1.00     83586

```

