
def get_marital_status(row) -> str:
	if row['marital-status'].startswith('Married'):
		return 'Married'

	return 'Single'