import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
import datetime
from datetime import timezone

# load model
model = pickle.load(open('XGBClassifier_model.pkl', 'rb'))

test_data = pd.read_csv("test.csv", parse_dates=['participant_date_of_birth', 'treatment_created_date',
                                                       'claim_finalized_date'])
features = ['provider_type', 'provider_region', 'program_cover',
       'participant_gender', 'item_status', 'item_name', 'item_amount',
       'item_quantity', 'total_item_amount', 'participant_age',
       'participant_yearOB', 'participant_monthOB', 'participant_dayOB',
       'treat_cr_year', 'treat_cr_month', 'treat_cr_day', 'treat_cr_weekday',
       'claim_final_year', 'claim_final_month', 'claim_final_day',
       'claim_final_weekday', 'treat_claim_diff', 'totals_cat', 'itemq_cat',
       'age_cat', 'prov_typeXreg', 'prov_typeXcover', 'prov_regXcover',
       'prov_typeXgen', 'prov_regXgen', 'coverXgen', 'regXstatus',
       'statusXgen', 'coverXstatus']


# feature engineering
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


def claims_projections(claims_data):
    sample_rows = claims_data.sample(10)
    
    # correct dtypes
    for col in ['participant_date_of_birth', 'treatment_created_date', 'claim_finalized_date']:
        sample_rows[col] = pd.to_datetime(sample_rows[col])
        
    test_rows = sample_rows.drop('claim_status', axis=1)
    
    # make predictions
    preds = model.predict(test_rows[features])
    
    # dataframe to compare actual vs predicted values
    compare_vals = pd.DataFrame({'Actual': sample_rows.claim_status,
                                 'Predicted': preds})
    
    sns.countplot(sample_rows.claim_status)
    
    return sample_rows.sample(3).iloc[:,:4], plt.gcf(), compare_vals

iface = gr.Interface(claims_projections, 
    gr.inputs.Dataframe(
        headers=list(test_data.columns),
        default=[list(arr) for arr in test_data.sample(20).values]
    ),
    [
        "dataframe",
        "plot",
        "dataframe"
    ],
    description="Refresh page to auto generate random samples from a preprocessed test set.\
                \nSee comparison of actual vs predicted claim status values in the table generated."
)
iface.launch()