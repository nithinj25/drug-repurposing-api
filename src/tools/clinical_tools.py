def extract_clinical_trial_data(trial_record):
    # Function to extract relevant data from a clinical trial record
    extracted_data = {
        'title': trial_record.get('title'),
        'status': trial_record.get('status'),
        'start_date': trial_record.get('start_date'),
        'end_date': trial_record.get('end_date'),
        'interventions': trial_record.get('interventions'),
        'results': trial_record.get('results'),
    }
    return extracted_data

def normalize_trial_record(trial_record):
    # Function to normalize a clinical trial record for consistency
    normalized_record = {
        'title': trial_record['title'].strip().lower(),
        'status': trial_record['status'].strip().lower(),
        'start_date': trial_record['start_date'],
        'end_date': trial_record['end_date'],
        'interventions': [intervention.strip().lower() for intervention in trial_record['interventions']],
        'results': trial_record.get('results', None),
    }
    return normalized_record

def validate_trial_data(trial_data):
    # Function to validate the extracted clinical trial data
    required_fields = ['title', 'status', 'start_date']
    for field in required_fields:
        if field not in trial_data or not trial_data[field]:
            return False
    return True