# -----------------------------------------------------------------------------
# Shared covariates definitions for stroke analysis
# -----------------------------------------------------------------------------
# Ensure all columns used in the model are numeric
# X = df.drop(['stroke_status_derived', 'individual_id', 'obs_date'], axis=1, errors='ignore')
COVARIATES = [
    'hpt_status_derived',
    'age',
    'sex_binary',
    'diab_status_derived',
    'bmi_category_Overweight_Obese',
    'site_Nairobi',
    # 'alcohol_use',  # Will return this later
    'tobacco_use',
    'hiv_status_derived',
	'hpt_diab_interaction', # Interraction term
	'hpt_site_interaction',
	'site_diab_interaction',
	'bmi_sex_interaction',
	'hpt_age_interaction',
	
	# site × hypertension, site × diabetes, or BMI × sex
]

# -----------------------------------------------------------------------------
# Predictors - forest plots/compare across sites
# -----------------------------------------------------------------------------
# predictors to display in forest plots/compare across sites
KEY_PREDICTORS = [
    'hpt_status_derived',
    'diab_status_derived',
    'tb_status_derived'
] # , 'site_Nairobi'
# 'sex_binary', 'alcohol_use', 'tobacco_use', 'hpt_status_derived','obese_status_derived', 'diab_status_derived','bmi_refined','hiv_status_derived', 'tb_status_derived'

# -----------------------------------------------------------------------------
# Optional: grouped covariates
# -----------------------------------------------------------------------------
DEMOGRAPHIC_VARS = [
    'sex_binary',
    'site_Nairobi'
]

# -----------------------------------------------------------------------------
# Optional: clinical variables
# -----------------------------------------------------------------------------
CLINICAL_VARS = [
    'hpt_status_derived',
    'diab_status_derived',
    'hiv_status_derived',
    'bmi_category_Overweight_Obese'
]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

