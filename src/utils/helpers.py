def load_json(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    import json
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def extract_keys_from_dict_list(dict_list, key):
    return [d[key] for d in dict_list if key in d]

def calculate_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0

def format_date(date_string, input_format="%Y-%m-%d", output_format="%d-%m-%Y"):
    from datetime import datetime
    date_obj = datetime.strptime(date_string, input_format)
    return date_obj.strftime(output_format)