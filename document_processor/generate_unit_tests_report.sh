#!/bin/bash
# For sonarqube to get the unit test coverage report.

# Set environment variable indicating unit testing
export TESTING=True

# Run the tests and generate reports
pytest -v --cov --cov-config=.coveragerc --cov-report xml:reports/coverage.xml --junitxml=reports/pytest.xml

# Save the pytest exit code
PYTEST_EXIT_CODE=$?

# Forward unit tests status to CI
if [[ $PYTEST_EXIT_CODE -ne 0 ]]; then
    exit $PYTEST_EXIT_CODE
fi

# Fix paths in coverage.xml
old_path="src/"
new_path="document_processor\/src\/"

# The coverage.xml filepath
file="reports/coverage.xml"

# Use sed to perform the replacement
sed -i "s#${old_path}#${new_path}#g" "${file}"


# Fix paths in pytest.xml
old_path="src\."
new_path="document_processor\.src\."

# The pytest.xml filepath
file="reports/pytest.xml"

# Use sed to perform the replacement
sed -i "s#${old_path}#${new_path}#g" "${file}"

# Delete environment variable indicating unit testing
unset TESTING


